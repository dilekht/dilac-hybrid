"""
Error-Example Extractor (Kaggle)
=================================
Runs the trained CAMeLBERT-MSA GlossBERT model on the test set and dumps the
actual misclassified instances, with the gold gloss and the predicted gloss,
so the paper's error-analysis table contains REAL examples (not invented ones).

Reuses the verbatim eval logic from step9_kaggle_v2.py (mark_target, enrich,
leave-one-out, MAX_LEN=256, argmax). Requires the saved checkpoint at CKPT_PATH
(the same one step9/steps10_12 used). If absent, fine-tunes first.

Output: /kaggle/working/error_examples.json
  - a ranked sample of misclassifications, grouped by error type:
    * binary_sense  (lemma has exactly 2 senses)  -> the hardest band
    * high_polysemy (>=7 senses)
    * near_synonym  (gold & predicted glosses share many tokens)
  plus a flat list of 30 examples for manual table selection.
"""
import os, glob, json, re, random
from collections import Counter, defaultdict
import numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

SEED=42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME='CAMeL-Lab/bert-base-arabic-camelbert-msa'
MAX_LEN=256
# step9_kaggle_v2.py saves the CAMeLBERT GlossBERT checkpoint here:
CKPT_PATH='/kaggle/working/glossbert_models/glossbert_camelbert_msa/best_model.pt'

_c=glob.glob('/kaggle/input/**/gold_test_v2.json',recursive=True)
if not _c: raise FileNotFoundError('gold_test_v2.json not found; attach dilac-project-v2.')
INPUT_DIR=os.path.dirname(_c[0]); print('INPUT_DIR:',INPUT_DIR)
test=json.load(open(f'{INPUT_DIR}/gold_test_v2.json',encoding='utf-8'))
print('test instances:',len(test)); assert len(test)==2050

_DIAC=re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
                 r'\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
def normalize_ar(t):
    t=_DIAC.sub('',t); t=re.sub(r'[إأآٱ]','ا',t)
    return t.replace('ة','ه').replace('ى','ي').strip()
def mark_target(context, lemma):
    nl=normalize_ar(lemma); ws=normalize_ar(context).split()
    for i,w in enumerate(ws):
        if w==nl or (len(nl)>=3 and w.startswith(nl[:3])):
            ws[i]=f'[TGT] {w} [/TGT]'; break
    return ' '.join(ws)

class GlossBERT(nn.Module):
    def __init__(self,name):
        super().__init__(); self.bert=AutoModel.from_pretrained(name)
        self.drop=nn.Dropout(0.1)
        self.classifier=nn.Linear(self.bert.config.hidden_size,2)  # matches step9
    def forward(self,ids,mask,tids=None):
        out=self.bert(input_ids=ids,attention_mask=mask,token_type_ids=tids)
        return self.classifier(self.drop(out.last_hidden_state[:,0,:]))
    def score(self,ids,mask,tids=None):
        return torch.softmax(self.forward(ids,mask,tids),-1)[:,1]

# Locate the checkpoint robustly: try known path, else search.
ckpt=CKPT_PATH
if not os.path.exists(ckpt):
    cands=glob.glob('/kaggle/working/**/best_model.pt', recursive=True)
    cands=[c for c in cands if 'camelbert' in c.lower()] or cands
    if not cands:
        found=glob.glob('/kaggle/working/**/*.pt', recursive=True)
        raise FileNotFoundError(
            "No CAMeLBERT checkpoint found.\n"
            f"Looked for: {CKPT_PATH}\n"
            f".pt files present: {found}\n"
            "Run step9_kaggle_v2.py in THIS session first.")
    ckpt=cands[0]
print("Using checkpoint:", ckpt)

# step9 saved the tokenizer (with [TGT]/[/TGT] added) next to the checkpoint.
save_dir=os.path.dirname(ckpt)
try:
    tok=AutoTokenizer.from_pretrained(save_dir)   # has the 2 extra tokens
    print("loaded saved tokenizer (vocab=%d)"%len(tok))
except Exception:
    tok=AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.add_tokens(['[TGT]','[/TGT]'])
    print("rebuilt tokenizer with [TGT]/[/TGT] (vocab=%d)"%len(tok))

mdl=GlossBERT(MODEL_NAME).to(DEVICE)
mdl.bert.resize_token_embeddings(len(tok))   # 30000 -> 30002, matches checkpoint
mdl.load_state_dict(torch.load(ckpt,map_location=DEVICE)); mdl.eval()
print("checkpoint loaded:", ckpt)

def gloss_of(cand):  # short readable gloss
    g=(cand.get('gloss','') or '').strip()
    return g[:80]

def enrich(s, ctx_norm):
    gloss=normalize_ar(s.get('gloss','') or ''); kept=[]
    for ex in (s.get('examples',[]) or []):
        if normalize_ar(ex)!=ctx_norm: kept.append(normalize_ar(ex))
        if len(kept)>=2: break
    return (gloss+' '+' '.join(kept)).strip() or 'معنى'

errors=[]
with torch.no_grad():
    for inst in tqdm(test,desc='scanning'):
        ctx=mark_target(inst['context'],inst['lemma']); gold=inst['target_sense']
        ctx_norm=normalize_ar(inst['context']); senses=inst['all_senses']
        pairs=[(ctx,enrich(s,ctx_norm)) for s in senses]
        gkeys=[s['sense_gkey'] for s in senses]
        enc=tok([p[0] for p in pairs],[p[1] for p in pairs],max_length=MAX_LEN,
                truncation=True,padding=True,return_tensors='pt')
        ids=enc['input_ids'].to(DEVICE); mask=enc['attention_mask'].to(DEVICE)
        tids=enc.get('token_type_ids',torch.zeros_like(ids)).to(DEVICE)
        sc=mdl.score(ids,mask,tids); pi=int(sc.argmax().item())
        pred=gkeys[pi]
        if pred!=gold:
            gold_s=next(s for s in senses if s['sense_gkey']==gold)
            pred_s=senses[pi]
            # DiLAC glosses rarely share exact tokens, so we categorize by the
            # reliable signals: sense count (band) and model confidence.
            gt=set(normalize_ar(gold_s.get('gloss','')).split())
            pt=set(normalize_ar(pred_s.get('gloss','')).split())
            overlap=len(gt&pt)/max(len(gt|pt),1)
            n=len(senses)
            gold_conf=float(sc[gkeys.index(gold)]); pred_conf=float(sc[pi])
            # confident error = model was sure but wrong (most instructive)
            confident = pred_conf>=0.7
            etype=('binary_sense' if n==2 else
                   'high_polysemy' if n>=7 else
                   'mid_polysemy')
            errors.append({
                'lemma':inst['lemma'],'n_senses':n,'error_type':etype,
                'confident_error':confident,
                'context':inst['context'][:90],
                'gold_gloss':gloss_of(gold_s),'pred_gloss':gloss_of(pred_s),
                'gold_conf':round(gold_conf,3),
                'pred_conf':round(pred_conf,3),
                'gloss_overlap':round(overlap,2),
            })

print(f"\nTotal errors: {len(errors)} / {len(test)} "
      f"({100*len(errors)/len(test):.1f}%)")
overall_acc = 100*(len(test)-len(errors))/len(test)
print(f"This model's overall accuracy: {overall_acc:.2f}%  "
      f"(should be ~69.5%, reproducing the locked run)")
by_type=Counter(e['error_type'] for e in errors)
print("By type:", dict(by_type))

# Pick a clean, diverse sample for the paper table.
# Prefer CONFIDENT errors (model sure but wrong) as the most instructive.
def pick(t,k):
    pool=[e for e in errors if e['error_type']==t]
    pool.sort(key=lambda e:-e['pred_conf'])  # most confident mistakes first
    return pool[:k]
table=pick('binary_sense',2)+pick('high_polysemy',1)+pick('mid_polysemy',2)

out={'summary':{'total_errors':len(errors),'by_type':dict(by_type),
                'error_rate':round(100*len(errors)/len(test),1)},
     'table_candidates':table,
     'all_errors_sample':errors[:30]}
with open('/kaggle/working/error_examples.json','w',encoding='utf-8') as _f:
    json.dump(out,_f,ensure_ascii=False,indent=2)
print("\nSaved error_examples.json")
print("\n=== Suggested table rows ===")
for e in table:
    print(f"{e['lemma']:8s} | n={e['n_senses']} | {e['error_type']:13s} | "
          f"conf={e['pred_conf']}")
    print(f"   ctx : {e['context']}")
    print(f"   gold: {e['gold_gloss']}")
    print(f"   pred: {e['pred_gloss']}\n")
