"""
Step 9 v2 — GlossBERT with Example-Enriched Glosses (Kaggle)
=================================================
Run on Kaggle with GPU T4 x2 accelerator.

Input files (dataset 'dilac-wsd-splits'):
    /kaggle/input/dilac-wsd-splits/gold_train.json
    /kaggle/input/dilac-wsd-splits/gold_dev.json
    /kaggle/input/dilac-wsd-splits/gold_test.json

Outputs saved to /kaggle/working/:
    glossbert_models/glossbert_camelbert_msa/best_model.pt
    step9_glossbert_results.json

Time  : ~40 min on T4 (CAMeLBERT only, MAX_LEN=128)
"""

import os, json, re, random, time
from collections import Counter
from tqdm.auto import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModel,
                          get_linear_schedule_with_warmup)

# ── Paths (Kaggle-specific) ────────────────────────────────────────────────
import glob as _glob
_c=_glob.glob('/kaggle/input/**/gold_test_v2.json',recursive=True)
if not _c: raise FileNotFoundError('gold_test_v2.json not found under /kaggle/input. Add the dilac-project-v2 dataset.')
INPUT_DIR=os.path.dirname(_c[0])
print('INPUT_DIR:',INPUT_DIR)
OUTPUT_DIR = '/kaggle/working'
MODEL_DIR  = f'{OUTPUT_DIR}/glossbert_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU — training will be very slow. Enable T4 GPU in settings.")

# ── Config ─────────────────────────────────────────────────────────────────
FULL_TRAIN = True
EPOCHS     = 3
BATCH_SIZE = 16
LR         = 2e-5
MAX_LEN    = 128    # 128 is 2× faster than 256, ~1% accuracy cost
SEED       = 42
MAX_NEG    = 2      # max negative pairs per instance (prevents imbalance)

# ── Normalization ───────────────────────────────────────────────────────────
_DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
                   r'\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
def normalize_ar(text):
    text = _DIAC.sub('', text)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    return text.replace('ة','ه').replace('ى','ي').strip()

# ── Load splits ─────────────────────────────────────────────────────────────
def load_split(name):
    path = f'{INPUT_DIR}/{name}'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Add dataset 'dilac-wsd-splits' to this notebook via Add Data."
        )
    with open(path, encoding='utf-8') as f:
        return json.load(f)

print("Loading splits ...")
train_data = load_split('gold_train_v2.json')
dev_data   = load_split('gold_dev_v2.json')
test_data  = load_split('gold_test_v2.json')
print(f"  Train={len(train_data):,}  Dev={len(dev_data):,}  "
      f"Test={len(test_data):,}")

if not FULL_TRAIN:
    rng = random.Random(SEED)
    train_data = rng.sample(train_data, 3000)
    dev_data   = rng.sample(dev_data,   500)
    test_data  = rng.sample(test_data,  500)
    print(f"  Quick mode: {len(train_data)} / {len(dev_data)} / {len(test_data)}")

# ── Target word marking ──────────────────────────────────────────────────────
def mark_target(context, lemma):
    nl = normalize_ar(lemma)
    ws = normalize_ar(context).split()
    for i,w in enumerate(ws):
        if w==nl or (len(nl)>=3 and w.startswith(nl[:3])):
            ws[i] = f'[TGT] {w} [/TGT]'; break
    return ' '.join(ws)

# ── Dataset ──────────────────────────────────────────────────────────────────
class GlossBERTDataset(Dataset):
    def __init__(self, instances, tokenizer, max_len=MAX_LEN, seed=SEED):
        self.pairs=[]; self.tok=tokenizer; self.mlen=max_len
        rng=random.Random(seed)
        for inst in instances:
            ctx  = mark_target(inst['context'], inst['lemma'])
            gold = inst['target_sense']
            ctx_norm = normalize_ar(inst['context'])
            neg_kept = 0
            for s in inst['all_senses']:
                # Build enriched gloss = gloss + real examples, EXCLUDING
                # the current context sentence (leave-one-out, no leakage).
                gloss = normalize_ar(s.get('gloss','') or '')
                ex_list = s.get('examples', []) or []
                kept_ex = []
                for ex in ex_list:
                    if normalize_ar(ex) != ctx_norm:   # exclude test sentence
                        kept_ex.append(normalize_ar(ex))
                    if len(kept_ex) >= 2:              # 2 examples is enough
                        break
                enriched = (gloss + ' ' + ' '.join(kept_ex)).strip() or 'معنى'
                label = 1 if s['sense_gkey']==gold else 0
                if label==0:
                    if neg_kept>=MAX_NEG: continue
                    neg_kept+=1
                self.pairs.append((ctx, enriched, label))
        rng.shuffle(self.pairs)
        pos=sum(1 for _,_,l in self.pairs if l==1)
        print(f"    Pairs={len(self.pairs):,}  pos={pos:,}  "
              f"neg={len(self.pairs)-pos:,}  ratio=1:{(len(self.pairs)-pos)//max(pos,1):.1f}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        ctx,gloss,label=self.pairs[i]
        enc=self.tok(ctx, gloss, max_length=self.mlen,
                     truncation=True, padding='max_length',
                     return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc.get('token_type_ids',
                              torch.zeros(self.mlen, dtype=torch.long)
                              ).squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }

# ── Model ────────────────────────────────────────────────────────────────────
class GlossBERT(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert=AutoModel.from_pretrained(model_name)
        self.drop=nn.Dropout(dropout)
        self.classifier=nn.Linear(self.bert.config.hidden_size, 2)
    def forward(self, ids, mask, tids=None):
        out=self.bert(input_ids=ids, attention_mask=mask, token_type_ids=tids)
        return self.classifier(self.drop(out.last_hidden_state[:,0,:]))
    @torch.no_grad()
    def score(self, ids, mask, tids=None):
        return torch.softmax(self.forward(ids,mask,tids), dim=-1)[:,1]

# ── Train ────────────────────────────────────────────────────────────────────
def train_one(model_name, label):
    save_path = f'{MODEL_DIR}/glossbert_{label.lower().replace("-","_")}'
    os.makedirs(save_path, exist_ok=True)
    ckpt_file = f'{save_path}/best_model.pt'

    print(f"\n{'='*52}\nGlossBERT — {label}\n{'='*52}")

    if os.path.exists(ckpt_file):
        print("  ⚡  Checkpoint found — loading ...")
        tok=AutoTokenizer.from_pretrained(save_path)
        mdl=GlossBERT(model_name).to(DEVICE)
        mdl.bert.resize_token_embeddings(len(tok))
        mdl.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
        return mdl, tok, None

    tok=AutoTokenizer.from_pretrained(model_name)
    tok.add_tokens(['[TGT]','[/TGT]'])

    print("  Building datasets ...")
    tr_ds=GlossBERTDataset(train_data, tok)
    dv_ds=GlossBERTDataset(dev_data,   tok, seed=99)
    # num_workers=0 avoids multiprocessing errors on Kaggle/Colab
    tr_ld=DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                     num_workers=0, pin_memory=True)
    dv_ld=DataLoader(dv_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                     num_workers=0, pin_memory=True)

    mdl=GlossBERT(model_name).to(DEVICE)
    mdl.bert.resize_token_embeddings(len(tok))

    no_decay=['bias','LayerNorm.weight']
    params=[
        {'params':[p for n,p in mdl.named_parameters()
                   if not any(nd in n for nd in no_decay)],
         'weight_decay':0.01},
        {'params':[p for n,p in mdl.named_parameters()
                   if any(nd in n for nd in no_decay)],
         'weight_decay':0.0},
    ]
    opt   = torch.optim.AdamW(params, lr=LR)
    total = len(tr_ld)*EPOCHS
    sched = get_linear_schedule_with_warmup(opt, total//10, total)
    crit  = nn.CrossEntropyLoss()
    best_dev=0.0; log=[]

    for epoch in range(1, EPOCHS+1):
        mdl.train()
        tr_loss=tr_cor=tr_tot=0
        for batch in tqdm(tr_ld, desc=f'E{epoch}/train', leave=False):
            ids=batch['input_ids'].to(DEVICE)
            mask=batch['attention_mask'].to(DEVICE)
            tids=batch['token_type_ids'].to(DEVICE)
            labs=batch['label'].to(DEVICE)
            opt.zero_grad()
            logits=mdl(ids,mask,tids)
            loss=crit(logits,labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(),1.0)
            opt.step(); sched.step()
            tr_loss+=loss.item()
            tr_cor+=(logits.argmax(-1)==labs).sum().item()
            tr_tot+=labs.size(0)

        mdl.eval()
        dv_cor=dv_tot=0
        with torch.no_grad():
            for batch in tqdm(dv_ld, desc=f'E{epoch}/dev', leave=False):
                ids=batch['input_ids'].to(DEVICE)
                mask=batch['attention_mask'].to(DEVICE)
                tids=batch['token_type_ids'].to(DEVICE)
                labs=batch['label'].to(DEVICE)
                preds=mdl(ids,mask,tids).argmax(-1)
                dv_cor+=(preds==labs).sum().item()
                dv_tot+=labs.size(0)

        tr_acc=tr_cor/tr_tot*100; dv_acc=dv_cor/dv_tot*100
        print(f"  Epoch {epoch}: loss={tr_loss/len(tr_ld):.4f}  "
              f"train={tr_acc:.2f}%  dev={dv_acc:.2f}%")
        log.append({'epoch':epoch,'train_acc':round(tr_acc,2),
                    'dev_acc':round(dv_acc,2)})

        # Save per-epoch checkpoint for resume capability
        torch.save(mdl.state_dict(), f'{save_path}/epoch_{epoch}.pt')
        if dv_acc>best_dev:
            best_dev=dv_acc
            torch.save(mdl.state_dict(), ckpt_file)
            tok.save_pretrained(save_path)
            print(f"    ✅  Best saved (dev={dv_acc:.2f}%)")

    print(f"  Best dev: {best_dev:.2f}%")
    return mdl, tok, log

# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate_glossbert(instances, mdl, tok, label):
    mdl.eval()
    correct=0; total=0
    tp=Counter(); fp=Counter(); fn=Counter()
    for inst in tqdm(instances, desc=f'{label} test', leave=False):
        ctx=mark_target(inst['context'], inst['lemma'])
        gold=inst['target_sense']
        senses=inst['all_senses']
        ctx_norm=normalize_ar(inst['context'])
        def enrich(s):
            gloss=normalize_ar(s.get('gloss','') or '')
            kept=[]
            for ex in (s.get('examples',[]) or []):
                if normalize_ar(ex)!=ctx_norm:   # leave-one-out
                    kept.append(normalize_ar(ex))
                if len(kept)>=2: break
            return (gloss+' '+' '.join(kept)).strip() or 'معنى'
        pairs=[(ctx, enrich(s)) for s in senses]
        gkeys=[s['sense_gkey'] for s in senses]
        enc=tok([p[0] for p in pairs],[p[1] for p in pairs],
                max_length=256, truncation=True,    # use 256 for best accuracy
                padding=True, return_tensors='pt')
        ids=enc['input_ids'].to(DEVICE)
        mask=enc['attention_mask'].to(DEVICE)
        tids=enc.get('token_type_ids',torch.zeros_like(ids)).to(DEVICE)
        with torch.no_grad():
            scores=mdl.score(ids,mask,tids)
        pred=gkeys[scores.argmax().item()]
        total+=1
        if pred==gold: correct+=1; tp[gold]+=1
        else: fp[pred]+=1; fn[gold]+=1
    acc=correct/total if total else 0
    all_s=set(tp)|set(fn); f1s=[]
    for s in all_s:
        p=tp[s]/(tp[s]+fp[s]) if (tp[s]+fp[s]) else 0
        r=tp[s]/(tp[s]+fn[s]) if (tp[s]+fn[s]) else 0
        f1s.append(2*p*r/(p+r) if (p+r) else 0)
    return {'model':label,'method':'glossbert_finetuned','n':total,
            'correct':correct,'accuracy':round(acc*100,2),
            'macro_f1':round((sum(f1s)/len(f1s) if f1s else 0)*100,2)}

# ── Run ───────────────────────────────────────────────────────────────────────
MODELS=[('CAMeL-Lab/bert-base-arabic-camelbert-msa','CAMeLBERT-MSA')]
all_results=[]

for mname, mlabel in MODELS:
    mdl,tok,log=train_one(mname,mlabel)
    save_path=f'{MODEL_DIR}/glossbert_{mlabel.lower().replace("-","_")}'
    best_tok=AutoTokenizer.from_pretrained(save_path)
    best_mdl=GlossBERT(mname).to(DEVICE)
    best_mdl.bert.resize_token_embeddings(len(best_tok))
    best_mdl.load_state_dict(
        torch.load(f'{save_path}/best_model.pt', map_location=DEVICE))
    res=evaluate_glossbert(test_data, best_mdl, best_tok, mlabel)
    res['training_log']=log or []
    all_results.append(res)
    print(f"\n  {mlabel} test: Acc={res['accuracy']:.2f}%  "
          f"F1={res['macro_f1']:.2f}%")
    del best_mdl, best_tok, mdl, tok
    if DEVICE.type=='cuda': torch.cuda.empty_cache()

# ── Save ──────────────────────────────────────────────────────────────────────
out={'description':'GlossBERT fine-tuned on DiLAC-WSD.','results':all_results}
out_path=f'{OUTPUT_DIR}/step9_glossbert_results.json'
with open(out_path,'w',encoding='utf-8') as f:
    json.dump(out,f,ensure_ascii=False,indent=2)
print(f"\n💾  Saved → {out_path}")
print(f"🎉  Step 9 done")
