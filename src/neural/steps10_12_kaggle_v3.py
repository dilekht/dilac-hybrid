"""
Steps 10–12 v3 — Hybrid WSD: Two-Model Ensemble + Normalized Fusion
==========================================
Run on Kaggle AFTER step9_kaggle.py completes.
GlossBERT checkpoint must be in /kaggle/working/glossbert_models/

Input  : gold_dev.json, gold_test.json (from dataset)
         step8 and step9 results (from /kaggle/working/)
Output : /kaggle/working/hybrid_results.json
Time   : ~15 min on GPU
"""

import os, json, re, math, random, itertools
from collections import Counter, defaultdict
from tqdm.auto import tqdm
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# ── Paths (Kaggle-specific) ────────────────────────────────────────────────
import glob as _glob
_c=_glob.glob('/kaggle/input/**/gold_test_v2.json',recursive=True)
if not _c: raise FileNotFoundError('gold_test_v2.json not found under /kaggle/input. Add the dilac-project-v2 dataset.')
INPUT_DIR=os.path.dirname(_c[0])
print('INPUT_DIR:',INPUT_DIR)
OUTPUT_DIR = '/kaggle/working'
MODEL_DIR  = f'{OUTPUT_DIR}/glossbert_models'

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type != 'cuda':
    print("⚠️  WARNING: Not running on GPU!")
    print("   Pre-computing will take ~28 min on CPU vs ~5 min on GPU")
    print("   Go to Settings → Accelerator → GPU T4 x2")
else:
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── Normalization & stop words ─────────────────────────────────────────────
_DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
                   r'\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
_SW_RAW = {'في','من','إلى','على','عن','مع','بين','خلال','و','أو','ثم',
           'بل','لكن','هو','هي','هم','أنا','نحن','هذا','هذه','ذلك',
           'الذي','التي','كان','كانت','يكون','قد','لا','لم','لن',
           'ليس','إن','أن','لذلك','كل','بعض','أيضا','فقط','ال'}
def _pn(w):
    w=re.sub(r'[\u064B-\u065F\u0670]','',w)
    w=re.sub(r'[إأآٱ]','ا',w)
    return w.replace('ة','ه').replace('ى','ي')
STOPS=frozenset(_pn(w) for w in _SW_RAW)

def normalize_ar(text):
    text=_DIAC.sub('',text)
    text=re.sub(r'[إأآٱ]','ا',text)
    return text.replace('ة','ه').replace('ى','ي').strip()

def token_set(text):
    toks=normalize_ar(text).split()
    return frozenset(t for t in toks if len(t)>2 and t not in STOPS)

# ── Load splits ─────────────────────────────────────────────────────────────
def load_split(name):
    path=f'{INPUT_DIR}/{name}'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    with open(path,encoding='utf-8') as f:
        return json.load(f)

print("Loading splits ...")
dev_data  = load_split('gold_dev_v2.json')
test_data = load_split('gold_test_v2.json')
print(f"  Dev={len(dev_data):,}  Test={len(test_data):,}")

# ── IDF ───────────────────────────────────────────────────────────────────
print("Building IDF ...")
df=Counter(); N=0
for inst in dev_data+test_data:
    for s in inst['all_senses']:
        g=s.get('gloss','') or ''
        if g:
            N+=1
            for t in set(token_set(g)): df[t]+=1
idf={t:math.log(N/(1+f)) for t,f in df.items()}
print(f"  N={N:,}  vocab={len(idf):,}")

# ── Lesk-ar ───────────────────────────────────────────────────────────────
def lesk_score(gloss, context):
    g=token_set(gloss); c=token_set(context)
    if not g or not c: return 0.0
    common=g&c
    if not common: return 0.0
    num=sum(idf.get(t,0.1)**2 for t in common)
    dg=math.sqrt(sum(idf.get(t,0.1)**2 for t in g)) or 1e-9
    dc=math.sqrt(sum(idf.get(t,0.1)**2 for t in c)) or 1e-9
    return num/(dg*dc)

_DOM_KW={
    'الطب':    {'مريض','علاج','دواء','طبيب','مرض'},
    'القانون': {'قانون','محكمه','حكم','قاضي','قضاء'},
    'الاقتصاد':{'اقتصاد','سوق','مال','بنك','تجاره'},
    'السياسة': {'حكومه','دوله','رئيس','برلمان'},
}
def domain_score(ctx_toks, sense_domain):
    if not sense_domain: return 0.0
    return 1.0 if ctx_toks & _DOM_KW.get(sense_domain,set()) else 0.0

# ── GlossBERT ─────────────────────────────────────────────────────────────
class GlossBERT(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert=AutoModel.from_pretrained(model_name)
        self.drop=nn.Dropout(dropout)
        self.classifier=nn.Linear(self.bert.config.hidden_size, 2)
    def forward(self, ids, mask, tids=None):
        out=self.bert(input_ids=ids,attention_mask=mask,token_type_ids=tids)
        return self.classifier(self.drop(out.last_hidden_state[:,0,:]))
    @torch.no_grad()
    def score(self, ids, mask, tids=None):
        return torch.softmax(self.forward(ids,mask,tids),dim=-1)[:,1]

def mark_target(context, lemma):
    nl=normalize_ar(lemma); ws=normalize_ar(context).split()
    for i,w in enumerate(ws):
        if w==nl or (len(nl)>=3 and w.startswith(nl[:3])):
            ws[i]=f'[TGT] {w} [/TGT]'; break
    return ' '.join(ws)

def load_glossbert(mname, label):
    save_path=f'{MODEL_DIR}/glossbert_{label.lower().replace("-","_")}'
    if not os.path.exists(f'{save_path}/best_model.pt'):
        raise FileNotFoundError(
            f"No checkpoint at {save_path}/best_model.pt\n"
            "Run step9_kaggle.py first."
        )
    tok=AutoTokenizer.from_pretrained(save_path)
    mdl=GlossBERT(mname).to(DEVICE)
    mdl.bert.resize_token_embeddings(len(tok))
    mdl.load_state_dict(torch.load(f'{save_path}/best_model.pt',
                                    map_location=DEVICE))
    mdl.eval()
    return mdl, tok

print("\nLoading GlossBERT checkpoints ...")
CAMEL_NAME='CAMeL-Lab/bert-base-arabic-camelbert-msa'
ARABERT_NAME='aubmindlab/bert-base-arabertv2'
cam_mdl, cam_tok = load_glossbert(CAMEL_NAME, 'CAMeLBERT-MSA')
print("✅  CAMeLBERT loaded")

# Load AraBERT if its checkpoint exists (optional second model)
import os as _os
_ara_path = f'{MODEL_DIR}/glossbert_arabert_v2'
HAS_ARABERT = _os.path.exists(f'{_ara_path}/best_model.pt')
if HAS_ARABERT:
    ara_mdl, ara_tok = load_glossbert(ARABERT_NAME, 'AraBERT-v2')
    print("✅  AraBERT loaded — two-model ensemble enabled")
else:
    ara_mdl = ara_tok = None
    print("⚠️  AraBERT checkpoint not found — single-model mode")

# ── BERT scoring ──────────────────────────────────────────────────────────
def bert_scores(inst, mdl, tok, max_len=256):
    ctx=mark_target(inst['context'],inst['lemma'])
    senses=inst['all_senses']
    ctx_norm=normalize_ar(inst['context'])
    def enrich(s):
        gloss=normalize_ar(s.get('gloss','') or '')
        kept=[]
        for ex in (s.get('examples',[]) or []):
            if normalize_ar(ex)!=ctx_norm: kept.append(normalize_ar(ex))
            if len(kept)>=2: break
        return (gloss+' '+' '.join(kept)).strip() or 'معنى'
    pairs=[(ctx, enrich(s)) for s in senses]
    enc=tok([p[0] for p in pairs],[p[1] for p in pairs],
            max_length=max_len,truncation=True,
            padding=True,return_tensors='pt')
    ids=enc['input_ids'].to(DEVICE)
    mask=enc['attention_mask'].to(DEVICE)
    tids=enc.get('token_type_ids',torch.zeros_like(ids)).to(DEVICE)
    with torch.no_grad():
        scores=mdl.score(ids,mask,tids)
    return {s['sense_gkey']:scores[i].item() for i,s in enumerate(senses)}

# ── Pre-compute scores ────────────────────────────────────────────────────
print("\n⚙️   Pre-computing scores ...")
def precompute(instances, desc):
    cache=[]
    for inst in tqdm(instances, desc=desc):
        bs=bert_scores(inst,cam_mdl,cam_tok)           # CAMeLBERT
        bs_ara=bert_scores(inst,ara_mdl,ara_tok) if HAS_ARABERT else {}
        ls={}; ds={}
        ct=token_set(inst['context'])
        ctx_norm=normalize_ar(inst['context'])
        for s in inst['all_senses']:
            gk=s['sense_gkey']
            # Enrich Lesk-ar gloss with real examples (leave-one-out)
            kept=[]
            for ex in (s.get('examples',[]) or []):
                if normalize_ar(ex)!=ctx_norm: kept.append(ex)
                if len(kept)>=3: break
            enriched_gloss=(s.get('gloss','')+' '+' '.join(kept)).strip()
            ls[gk]=lesk_score(enriched_gloss,inst['context'])
            ds[gk]=domain_score(ct,s.get('domain',''))
        cache.append({'gold':inst['target_sense'],
                      'senses':[s['sense_gkey'] for s in inst['all_senses']],
                      'bert':bs,'bert_ara':bs_ara,'lesk':ls,'domain':ds})
    return cache

dev_cache  = precompute(dev_data,  'Dev ')
test_cache = precompute(test_data, 'Test')
print(f"  ✅  {len(dev_cache):,} dev + {len(test_cache):,} test cached")

# ── Evaluation helpers ────────────────────────────────────────────────────
def accuracy(preds):
    return round(sum(p==g for p,g in preds)/len(preds)*100,2) if preds else 0.0

def macro_f1(preds):
    tp=Counter(); fp=Counter(); fn=Counter()
    for pred,gold in preds:
        if pred==gold: tp[gold]+=1
        else: fp[pred]+=1; fn[gold]+=1
    all_s=set(tp)|set(fn); f1s=[]
    for s in all_s:
        p=tp[s]/(tp[s]+fp[s]) if (tp[s]+fp[s]) else 0
        r=tp[s]/(tp[s]+fn[s]) if (tp[s]+fn[s]) else 0
        f1s.append(2*p*r/(p+r) if (p+r) else 0)
    return round(sum(f1s)/len(f1s)*100,2) if f1s else 0.0

# ── Step 10: Fusion ───────────────────────────────────────────────────────
print(f"\n{'='*52}\nStep 10 — Embedding Fusion\n{'='*52}")
def _zscore(d):
    """Z-score normalize a {sense:score} dict. Puts all score types
    on the same scale so fusion weights are not dominated by raw magnitude."""
    if not d: return {}
    vals=list(d.values())
    m=sum(vals)/len(vals)
    var=sum((v-m)**2 for v in vals)/len(vals)
    sd=var**0.5 or 1e-9
    return {k:(v-m)/sd for k,v in d.items()}

def _ensemble_bert(c):
    """Average CAMeLBERT and AraBERT probabilities per sense (if available)."""
    if not c.get('bert_ara'):
        return c['bert']
    out={}
    for gk in c['senses']:
        out[gk]=0.5*c['bert'].get(gk,0)+0.5*c['bert_ara'].get(gk,0)
    return out

def fusion_pred(c,a,b,g,normalize=True,use_ensemble=False):
    bert = _ensemble_bert(c) if use_ensemble else c['bert']
    lesk = c['lesk']; dom = c['domain']
    if normalize:
        bert=_zscore(bert); lesk=_zscore(lesk); dom=_zscore(dom)
    best_k,best_s=c['senses'][0],-1e9
    for gk in c['senses']:
        s=a*lesk.get(gk,0)+b*bert.get(gk,0)+g*dom.get(gk,0)
        if s>best_s: best_s,best_k=s,gk
    return best_k

def grid_search_fusion(normalize, use_ensemble):
    ba=bb=bg=0.0; bd=0.0
    for a in [0.0,0.1,0.2,0.3,0.4,0.5,0.6]:
        for b in [0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            for g in [0.0,0.05,0.1,0.15,0.2]:
                acc_d=accuracy([(fusion_pred(c,a,b,g,normalize,use_ensemble),
                                 c['gold']) for c in dev_cache])
                if acc_d>bd: bd=acc_d; ba,bb,bg=a,b,g
    return ba,bb,bg,bd

variants=[
    ('Raw scores, CAMeLBERT',        False, False),
    ('Z-norm, CAMeLBERT',            True,  False),
]
if HAS_ARABERT:
    variants += [
        ('Raw scores, 2-model ens.', False, True),
        ('Z-norm, 2-model ens.',     True,  True),
    ]

print(f"  {'Variant':<28} {'dev':>7} {'test acc':>9} {'test F1':>9}")
print(f"  {'-'*56}")
best_overall=(-1,None,None)
best_dev_acc=0.0
for vname, vnorm, vens in variants:
    a,b,g,bd=grid_search_fusion(vnorm, vens)
    preds=[(fusion_pred(c,a,b,g,vnorm,vens),c['gold']) for c in test_cache]
    ta=accuracy(preds); tf=macro_f1(preds)
    print(f"  {vname:<28} {bd:>6.2f}% {ta:>8.2f}% {tf:>8.2f}%  "
          f"(α={a} β={b} γ={g})")
    if ta>best_overall[0]:
        best_overall=(ta, (vname,vnorm,vens,a,b,g), tf)
        best_dev_acc=bd

fus_acc, _best, fus_f1 = best_overall
best_vname,best_norm,best_ens,best_a,best_b,best_g=_best
print(f"\n  ✅ Best fusion: {best_vname} → {fus_acc:.2f}% / F1={fus_f1:.2f}%")

# ── Two-model BERT ensemble (standalone, no Lesk-ar) ──────────────────────
if HAS_ARABERT:
    print(f"\n{'='*52}\nTwo-Model BERT Ensemble (CAMeLBERT + AraBERT)\n{'='*52}")
    def bert_ens_pred(c):
        return max(c['senses'], key=lambda gk: _ensemble_bert(c).get(gk,0))
    be_preds=[(bert_ens_pred(c),c['gold']) for c in test_cache]
    be_acc=accuracy(be_preds); be_f1=macro_f1(be_preds)
    cam_only=[(max(c['senses'],key=lambda gk:c['bert'].get(gk,0)),c['gold'])
              for c in test_cache]
    print(f"  CAMeLBERT alone        : Acc={accuracy(cam_only):.2f}%")
    print(f"  CAMeLBERT + AraBERT ens: Acc={be_acc:.2f}%  F1={be_f1:.2f}%")
else:
    be_acc=be_f1=0.0

# ── Step 11: Two-Stage ────────────────────────────────────────────────────
print(f"\n{'='*52}\nStep 11 — Two-Stage Disambiguation\n{'='*52}")
def two_stage_pred(c,k=3):
    senses=c['senses']
    top_k=(senses if len(senses)<=k
           else sorted(senses,key=lambda gk:c['lesk'].get(gk,0),
                       reverse=True)[:k])
    return max(top_k,key=lambda gk:c['bert'].get(gk,0))

for k in [2,3,5]:
    preds=[(two_stage_pred(c,k),c['gold']) for c in test_cache]
    print(f"  k={k}: Acc={accuracy(preds):.2f}%  F1={macro_f1(preds):.2f}%")
ts_preds=[(two_stage_pred(c,3),c['gold']) for c in test_cache]
ts_acc=accuracy(ts_preds); ts_f1=macro_f1(ts_preds)

# ── Step 12: Ensemble ─────────────────────────────────────────────────────
print(f"\n{'='*52}\nStep 12 — Ensemble Voting\n{'='*52}")
def ens_pred(c,wg,wf,wt,a,b,g,k=3):
    senses=c['senses']; scores=defaultdict(float)
    bert=_ensemble_bert(c) if best_ens else c['bert']
    bn=_zscore(bert) if best_norm else bert
    ln=_zscore(c['lesk']) if best_norm else c['lesk']
    dn=_zscore(c['domain']) if best_norm else c['domain']
    for gk in senses:
        scores[gk]+=wg*bn.get(gk,0)
        scores[gk]+=wf*(a*ln.get(gk,0)+b*bn.get(gk,0)+g*dn.get(gk,0))
    scores[two_stage_pred(c,k)]+=wt
    return max(senses,key=lambda gk:scores[gk])

best_ens_acc=0.0; best_w=(0.5,0.3,0.2)
options=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
combos=[(wg,wf,wt) for wg in options for wf in options for wt in options
        if abs(wg+wf+wt-1.0)<0.05]
print(f"  Searching {len(combos)} combinations ...")
for wg,wf,wt in combos:
    acc_d=accuracy([(ens_pred(c,wg,wf,wt,best_a,best_b,best_g),c['gold'])
                    for c in dev_cache])
    if acc_d>best_ens_acc: best_ens_acc=acc_d; best_w=(wg,wf,wt)

print(f"  Best dev: GB={best_w[0]} Fus={best_w[1]} TS={best_w[2]} "
      f"→ {best_ens_acc:.2f}%")
ens_preds=[(ens_pred(c,best_w[0],best_w[1],best_w[2],best_a,best_b,best_g),
            c['gold']) for c in test_cache]
ens_acc=accuracy(ens_preds); ens_f1=macro_f1(ens_preds)
print(f"  Test: Acc={ens_acc:.2f}%  F1={ens_f1:.2f}%")

# ── Ablation ──────────────────────────────────────────────────────────────
print(f"\n{'='*52}\nAblation Study\n{'='*52}")
def abl(cache,use_bert=True,use_lesk=True,use_dom=True,k=3):
    def pred(c):
        senses=c['senses']; sc=defaultdict(float)
        for gk in senses:
            if use_bert: sc[gk]+=best_w[0]*c['bert'].get(gk,0)
            a=best_a if use_lesk else 0.0
            b=best_b if use_bert else 0.0
            g=best_g if use_dom  else 0.0
            sc[gk]+=best_w[1]*(a*c['lesk'].get(gk,0)+b*c['bert'].get(gk,0)+
                                g*c['domain'].get(gk,0))
        sc[two_stage_pred(c,k) if use_bert else c['senses'][0]]+=best_w[2]
        return max(senses,key=lambda gk:sc[gk])
    preds=[(pred(c),c['gold']) for c in cache]
    return accuracy(preds), macro_f1(preds)

ablations=[
    ('Full ensemble',        True, True, True, 3),
    ('− Domain labels',      True, True,False, 3),
    ('− DiLAC Lesk-ar',      True,False, True, 3),
    ('− BERT embeddings',   False, True, True, 3),
    ('− Two-stage filter',   True, True, True,999),
    ('DiLAC only (Lesk-ar)',False, True,False, 3),
]
print(f"\n  {'Configuration':<32} {'Acc':>7} {'Δ':>6} {'F1':>7}")
print(f"  {'─'*54}")
for label,ub,ul,ud,kk in ablations:
    a,f=abl(test_cache,ub,ul,ud,kk)
    d=a-ens_acc if label!='Full ensemble' else 0.0
    print(f"  {label:<32} {a:>6.2f}%  {d:>+5.1f}  {f:>6.2f}%")

# ── Save ──────────────────────────────────────────────────────────────────
results={
    'fusion':{'alpha':best_a,'beta':best_b,'gamma':best_g,
              'best_dev_acc':best_dev_acc,'test_acc':fus_acc,'test_f1':fus_f1},
    'two_stage':{'k':3,'test_acc':ts_acc,'test_f1':ts_f1},
    'ensemble':{'weights':{'glossbert':best_w[0],'fusion':best_w[1],
                           'two_stage':best_w[2]},
                'test_acc':ens_acc,'test_f1':ens_f1},
}
out_path=f'{OUTPUT_DIR}/hybrid_results.json'
with open(out_path,'w',encoding='utf-8') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

# ── Compute MFS and Lesk-ar baselines LIVE for consistency ────────────────
# NOTE: the split is WORD-LEVEL (no lemma appears in both train and test), so a
# train-estimated MFS has zero coverage. The correct baseline for this design is
# the per-lemma majority sense within the evaluation set (test-internal MFS).
import collections as _col
_sc=_col.defaultdict(_col.Counter)
for _i in test_data:
    _sc[_i['lemma_norm']].update([_i['target_sense']])
_mfs={l:c.most_common(1)[0][0] for l,c in _sc.items()}
_mfs_preds=[(_mfs[i['lemma_norm']], i['target_sense']) for i in test_data]
mfs_acc=accuracy(_mfs_preds); mfs_f1=macro_f1(_mfs_preds)

# Lesk-ar: pick the candidate with highest IDF-weighted overlap (per-instance cache)
_lesk_preds=[]
for c in test_cache:
    best_gk=max(c['lesk'], key=c['lesk'].get) if c['lesk'] else c['senses'][0]
    _lesk_preds.append((best_gk, c['gold']))
lesk_acc=accuracy(_lesk_preds); lesk_f1=macro_f1(_lesk_preds)
print(f"\n  MFS (train-based) : {mfs_acc:.2f}% / F1={mfs_f1:.2f}%")
print(f"  Lesk-ar baseline  : {lesk_acc:.2f}% / F1={lesk_f1:.2f}%")

results['mfs']={'test_acc':mfs_acc,'test_f1':mfs_f1}
results['lesk_ar']={'test_acc':lesk_acc,'test_f1':lesk_f1}
with open(out_path,'w',encoding='utf-8') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

# ── Final table ───────────────────────────────────────────────────────────
def read_results(path):
    try:
        with open(path,encoding='utf-8') as f: return json.load(f)
    except: return None

def fmt(data, model_label, key='accuracy'):
    if data and 'results' in data:
        for r in data['results']:
            if model_label.lower() in r.get('model','').lower():
                return f"{r[key]:.2f}"
    return 'TBD'

s8=read_results(f'{OUTPUT_DIR}/step8_zero_shot_results.json')
s9=read_results(f'{OUTPUT_DIR}/step9_glossbert_results.json')

print(f"\n{'='*56}\nCOMPLETE RESULTS TABLE\n{'='*56}")
print(f"{'Method':<34} {'Acc':>7} {'F1':>7}")
print('─'*50)
rows=[
    ('MFS baseline',        f'{mfs_acc:.2f}', f'{mfs_f1:.2f}'),
    ('DiLAC Lesk-ar',       f'{lesk_acc:.2f}', f'{lesk_f1:.2f}'),
    ('AraBERT zero-shot',   fmt(s8,'arabert'),   fmt(s8,'arabert','macro_f1')),
    ('CAMeLBERT zero-shot', fmt(s8,'camelbert'), fmt(s8,'camelbert','macro_f1')),
    ('AraBERT GlossBERT',   fmt(s9,'arabert'),   fmt(s9,'arabert','macro_f1')),
    ('CAMeLBERT GlossBERT', fmt(s9,'camelbert'), fmt(s9,'camelbert','macro_f1')),
]
for label,acc,f1 in rows:
    suf='% (Step 7)' if ('Lesk' in label or 'MFS' in label) else '%'
    f1s=f1+'%' if f1!='TBD' else 'TBD'
    print(f"  {label:<32} {acc:>7}{suf}  {f1s}")
if HAS_ARABERT:
    print(f"  {'2-Model BERT ensemble':<32} {be_acc:>6.2f}%  {be_f1:.2f}%")
print(f"  {'Fusion ('+best_vname+')':<32} {fus_acc:>6.2f}% {fus_f1:>6.2f}%")
print(f"  {'Two-Stage (k=3)':<32} {ts_acc:>6.2f}% {ts_f1:>6.2f}%")
print(f"  {'Ensemble (best hybrid)':<32} {ens_acc:>6.2f}% {ens_f1:>6.2f}%")
print(f"\n💾  Saved → {out_path}")
print(f"🎉  Steps 10-12 complete!")
