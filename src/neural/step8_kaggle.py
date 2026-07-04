"""
Step 8 — Zero-Shot WSD (Kaggle version)
=========================================
Run on Kaggle with GPU T4 x2 accelerator.

Input files (upload as dataset named 'dilac-wsd-splits'):
    /kaggle/input/dilac-wsd-splits/gold_test.json

Output:
    /kaggle/working/step8_zero_shot_results.json
"""

import os, json, re, time, random
from collections import Counter
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# ── Paths (Kaggle-specific) ────────────────────────────────────────────────
import glob as _glob
_c=_glob.glob('/kaggle/input/**/gold_test_v2.json',recursive=True)
if not _c: raise FileNotFoundError('gold_test_v2.json not found under /kaggle/input. Add the dilac-project-v2 dataset.')
INPUT_DIR=os.path.dirname(_c[0])
print('INPUT_DIR:',INPUT_DIR)
OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU — evaluation will be slow (~2 hours)")

# ── Arabic normalization ───────────────────────────────────────────────────
_DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
                   r'\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
def normalize_ar(text):
    text = _DIAC.sub('', text)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    return text.replace('ة','ه').replace('ى','ي').strip()

# ── Load test set ──────────────────────────────────────────────────────────
test_path = f'{INPUT_DIR}/gold_test_v2.json'
_n=len(json.load(open(test_path,encoding='utf-8')));print('test instances:',_n);assert _n==2050,'NOT v2! got %d'%_n
if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"File not found: {test_path}\n"
        "Make sure you added the dataset 'dilac-wsd-splits' to this notebook.\n"
        "In the right panel: Add Data → Your Datasets → dilac-wsd-splits"
    )

with open(test_path, encoding='utf-8') as f:
    test_data = json.load(f)
print(f"Test set: {len(test_data):,} instances")

EVAL_ALL = True
if not EVAL_ALL:
    test_data = random.Random(42).sample(test_data, 500)
    print(f"Sample mode: {len(test_data)} instances")

# ── Embedding utilities ────────────────────────────────────────────────────
def mean_pool(emb, mask):
    m = mask.unsqueeze(-1).expand(emb.size()).float()
    return torch.sum(emb*m, 1) / torch.clamp(m.sum(1), min=1e-9)

@torch.no_grad()
def encode(text, tok, mdl, max_len=256):
    text = normalize_ar(text) or 'نص'
    enc  = tok(text, return_tensors='pt', max_length=max_len,
               truncation=True, padding=True).to(DEVICE)
    out  = mdl(**enc)
    emb  = mean_pool(out.last_hidden_state, enc['attention_mask'])
    return torch.nn.functional.normalize(emb, p=2, dim=1)[0].cpu()

# ── Zero-shot evaluator ────────────────────────────────────────────────────
def run_zero_shot(instances, tok, mdl, label):
    correct=0; total=0
    tp=Counter(); fp=Counter(); fn=Counter()
    cache={}
    for inst in tqdm(instances, desc=label, leave=False):
        ctx  = encode(inst['context'], tok, mdl)
        gold = inst['target_sense']
        best_key, best_sc = inst['all_senses'][0]['sense_gkey'], -1.0
        for s in inst['all_senses']:
            gk = s['sense_gkey']
            if gk not in cache:
                cache[gk] = encode(s.get('gloss','') or '', tok, mdl)
            sc = torch.dot(ctx, cache[gk]).item()
            if sc > best_sc:
                best_sc, best_key = sc, gk
        total += 1
        if best_key == gold:
            correct += 1; tp[gold] += 1
        else:
            fp[best_key] += 1; fn[gold] += 1
    acc   = correct/total if total else 0
    all_s = set(tp)|set(fn); f1s=[]
    for s in all_s:
        p = tp[s]/(tp[s]+fp[s]) if (tp[s]+fp[s]) else 0
        r = tp[s]/(tp[s]+fn[s]) if (tp[s]+fn[s]) else 0
        f1s.append(2*p*r/(p+r) if (p+r) else 0)
    return {'model':label, 'method':'zero_shot', 'n':total,
            'correct':correct, 'accuracy':round(acc*100,2),
            'macro_f1':round((sum(f1s)/len(f1s) if f1s else 0)*100,2)}

# ── Run both models ────────────────────────────────────────────────────────
MODELS = [
    ('CAMeL-Lab/bert-base-arabic-camelbert-msa', 'CAMeLBERT-MSA'),
    ('aubmindlab/bert-base-arabertv2',            'AraBERT-v2'),
]
results = []
print(f"\n{'Model':<22} {'Acc':>7} {'F1':>7} {'Time':>8}")
print('─'*48)

for mname, mlabel in MODELS:
    print(f"\n  Loading {mlabel} ...")
    tok = AutoTokenizer.from_pretrained(mname)
    mdl = AutoModel.from_pretrained(mname).to(DEVICE); mdl.eval()
    t0  = time.time()
    res = run_zero_shot(test_data, tok, mdl, mlabel)
    elapsed = round(time.time()-t0, 1)
    results.append(res)
    print(f"  {mlabel:<20} {res['accuracy']:>6.2f}% "
          f"{res['macro_f1']:>6.2f}% {elapsed:>7.1f}s")
    del mdl, tok
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()

# ── Save ───────────────────────────────────────────────────────────────────
out = {'eval_mode':'full' if EVAL_ALL else 'sample',
       'n_instances':len(test_data), 'results':results}
out_path = f'{OUTPUT_DIR}/step8_zero_shot_results.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"\n── Step 8 results ───────────────────────────────────")
print(f"{'MFS':<28} 21.64%   9.30%  (Step 7)")
print(f"{'DiLAC Lesk-ar':<28} 27.67%  18.10%  (Step 7)")
for r in results:
    print(f"{r['model']+' zero-shot':<28} "
          f"{r['accuracy']:>5.2f}%  {r['macro_f1']:>5.2f}%")
print(f"\n💾  Saved → {out_path}")
print(f"🎉  Step 8 done")
