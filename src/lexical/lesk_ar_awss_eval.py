"""
DiLAC Lesk-ar — Enhanced Evaluation with Three Improvements
=============================================================
Runs five configurations and prints a side-by-side comparison.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG 0 — Baseline
  Binary token presence, global IDF (current approach)

CONFIG 1 — Gloss Priority
  Gloss tokens weighted W_GLOSS × more than example tokens.
  Rationale: glosses are definitional; examples are contextual.
  A token in the gloss is stronger evidence of word meaning
  than the same token appearing once in a news sentence.

CONFIG 2 — Frequency Weighting
  Token contribution = how many examples contain that token.
  Rationale: a token appearing in 20/25 examples is much
  stronger evidence than one appearing in 1/25 examples.

CONFIG 3 — Combined (Gloss Priority + Frequency)
  Both improvements active simultaneously.

CONFIG 4 — Discriminative IDF
  Tokens that appear in FEWER candidate senses get higher weight.
  Standard IDF is computed over all 57 senses globally.
  Discriminative IDF re-weights per query: tokens unique to
  one or two senses are boosted; ubiquitous tokens are penalised.
  Combined with Config 3 for maximum effect.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run:
    python src\\lexical\\lesk_ar_awss_eval.py

Requires:
    data/dilac/dilac_awss.json
    data/awss/benchmark_data_awss_pairs.xlsx
"""

import json, re, math
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AWSS_JSON    = PROJECT_ROOT / 'data' / 'dilac' / 'dilac_awss.json'
AWSS_XLSX    = PROJECT_ROOT / 'data' / 'awss'  / 'benchmark_data_awss_pairs.xlsx'
OUT_DIR      = PROJECT_ROOT / 'results' / 'tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH     = OUT_DIR / 'lesk_ar_awss_eval.json'

HUMAN_CEILING  = 0.893
AWSS_ALGORITHM = 0.894

# ── Hyperparameters ────────────────────────────────────────────────────────
USE_STEM   = False   # OFF is better for news Arabic (confirmed experimentally)
W_GLOSS    = 3.0    # gloss tokens weighted this many × vs example tokens
DISC_ALPHA = 0.7    # weight of standard IDF in discriminative blend
DISC_BETA  = 0.3    # weight of discriminative boost

# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
_DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
                   r'\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
_PUNC = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟«»]')
_STOPS_RAW = {
    'في','من','إلى','على','عن','مع','بين','خلال','حول','ضد','رغم',
    'منذ','عند','بعد','قبل','حتى','دون','عبر','نحو','و','أو','ثم',
    'بل','لكن','لأن','إذ','إذا','كما','أما','بينما','حيث','عندما',
    'هو','هي','هم','هن','أنت','أنا','نحن','هذا','هذه','ذلك','تلك',
    'الذي','التي','الذين','ما','من','أي','كان','كانت','كانوا',
    'يكون','تكون','ليس','يجب','يمكن','قد','لقد','لا','لم','لن',
    'إن','أن','لذلك','لذا','كل','بعض','جميع','أيضا','فقط','ال',
    'قال','وقال','قالوا','يقول','قيل','ذكر','ورد',
}
def _pn(w):
    w = re.sub(r'[\u064B-\u065F\u0670]','',w)
    w = re.sub(r'[إأآٱ]','ا',w)
    return w.replace('ة','ه').replace('ى','ي')
STOPS = frozenset(_pn(w) for w in _STOPS_RAW)

_PREFIXES = ['وال','فال','بال','كال','لل','وب','وك','فل',
             'ال','و','ف','ب','ك','ل','س']
_SUFFIXES = ['وها','وهم','وهن','تها','تهم','تهن','ونه','ونها',
             'كما','هما','تان','تين','ها','هم','هن','كم','نا',
             'ون','ين','ان','تم','تن','وا','ات','ته','تي',
             'ه','ك','ي','ت','ن','ا']
MIN_STEM = 3

def light_stem(tok):
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok)-len(p)>=MIN_STEM:
            tok=tok[len(p):]; break
    for s in _SUFFIXES:
        if tok.endswith(s) and len(tok)-len(s)>=MIN_STEM:
            tok=tok[:-len(s)]; break
    return tok

def preprocess(text: str) -> list:
    """Returns token LIST (not set) to preserve frequency information."""
    text = _DIAC.sub('', text)
    text = re.sub(r'[إأآٱ]','ا',text)
    text = text.replace('ة','ه').replace('ى','ي')
    text = _PUNC.sub(' ', text)
    text = re.sub(r'\s+',' ',text).strip()
    toks = [t for t in text.split() if len(t)>2 and t not in STOPS]
    if USE_STEM:
        toks = [light_stem(t) for t in toks]
        toks = [t for t in toks if len(t)>=MIN_STEM]
    return toks

def norm_surface(w):
    w = _DIAC.sub('',w)
    w = re.sub(r'[إأآٱ]','ا',w)
    return w.replace('ة','ه').replace('ى','ي').strip()

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
if not AWSS_JSON.exists():
    raise FileNotFoundError(f"Missing: {AWSS_JSON}")
with open(AWSS_JSON, encoding='utf-8') as f:
    data = json.load(f)
entries = data['Dict']['wordEntry']

senses = []
for entry in entries:
    word = entry['word']['idWord']
    pos  = entry.get('property',{}).get('POS','')
    for sense in entry.get('explanation',[]):
        ex_list = []
        for ex in sense.get('example',[]):
            t = _DIAC.sub('', ex.get('defe','')).strip()
            if t: ex_list.append({'text':t,'type':'example'})
        for ae in sense.get('addExamples',[]):
            t = _DIAC.sub('', ae.get('defe','')).strip()
            if t: ex_list.append({'text':t,'type':'addExample'})
        senses.append({
            'sense_id': f"{word}::{sense.get('idx','')}",
            'word': word, 'idx': sense.get('idx',''),
            'gloss': sense.get('defx','').strip(),
            'examples': ex_list,
        })

# ════════════════════════════════════════════════════════════════════════════
#  IDF COMPUTATION
# ════════════════════════════════════════════════════════════════════════════
# Global IDF — over all 57 sense documents (gloss + examples combined)
df_global = Counter(); N_docs = 0
for s in senses:
    parts = [s['gloss']] + [e['text'] for e in s['examples']]
    toks  = set(preprocess(' '.join(p for p in parts if p)))
    if toks:
        N_docs += 1
        for t in toks: df_global[t] += 1
idf_global = {t: math.log(N_docs/(1+f)) for t,f in df_global.items()}

# Discriminative IDF boost:
# For each sense, compute how distinctive each token is relative to
# all other senses. Tokens appearing in fewer senses → higher boost.
# disc_score[sense_id][token] = log(N / (1 + df_excluding_this_sense))
disc_boost = {}
for s in senses:
    parts = [s['gloss']] + [e['text'] for e in s['examples']]
    my_toks = set(preprocess(' '.join(p for p in parts if p)))
    disc = {}
    for t in my_toks:
        # df_others = how many OTHER senses contain this token
        df_others = df_global.get(t, 0) - 1   # subtract self
        df_others = max(0, df_others)
        # Higher boost when fewer others contain this token
        disc[t] = math.log((N_docs - 1) / (1 + df_others))
    disc_boost[s['sense_id']] = disc

# ════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — SENSE REPRESENTATIONS (5 variants)
# ════════════════════════════════════════════════════════════════════════════

def _exclude(text, exclude):
    return _DIAC.sub('',text).strip() != _DIAC.sub('',exclude).strip()

def repr_baseline(sense, exclude=''):
    """Config 0: binary token set, all sources equal weight."""
    parts = [sense['gloss']] + [e['text'] for e in sense['examples']
                                 if _exclude(e['text'], exclude)]
    toks  = preprocess(' '.join(p for p in parts if p))
    return Counter({t:1 for t in toks})   # binary

def repr_gloss_priority(sense, exclude=''):
    """Config 1: gloss tokens weighted W_GLOSS, example tokens weight 1."""
    weights = Counter()
    for tok in preprocess(sense['gloss']):
        weights[tok] += W_GLOSS
    for ex in sense['examples']:
        if _exclude(ex['text'], exclude):
            for tok in preprocess(ex['text']):
                weights[tok] += 1.0
    return weights

def repr_frequency(sense, exclude=''):
    """Config 2: gloss = binary, examples = token frequency count."""
    weights = Counter()
    for tok in preprocess(sense['gloss']):
        weights[tok] += 1.0
    for ex in sense['examples']:
        if _exclude(ex['text'], exclude):
            for tok in preprocess(ex['text']):
                weights[tok] += 1.0   # frequency accumulates
    return weights

def repr_combined(sense, exclude=''):
    """Config 3: gloss priority (W_GLOSS×) + example frequency."""
    weights = Counter()
    for tok in preprocess(sense['gloss']):
        weights[tok] += W_GLOSS       # gloss = high priority
    for ex in sense['examples']:
        if _exclude(ex['text'], exclude):
            for tok in preprocess(ex['text']):
                weights[tok] += 1.0   # examples = frequency
    return weights

# ════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — LESK-AR SCORING (general weighted version)
# ════════════════════════════════════════════════════════════════════════════

def lesk_ar_weighted(ctx_toks: list, sense_weights: Counter,
                     idf: dict) -> float:
    """
    IDF-weighted cosine between context token list and
    weighted sense token Counter.

    score = Σ_t [ w_s(t) × idf(t)² ]   for t in ctx ∩ sense
            ─────────────────────────────────────────────────
            √(Σ idf(t)² for t∈ctx) × √(Σ (w_s(t)×idf(t))² for t∈sense)
    """
    ctx_set = set(ctx_toks)
    common  = ctx_set & set(sense_weights.keys())
    if not common: return 0.0

    num  = sum(sense_weights[t] * idf.get(t,0.1)**2 for t in common)
    da   = math.sqrt(sum(idf.get(t,0.1)**2 for t in ctx_set)) or 1e-9
    db   = math.sqrt(sum((sense_weights[t]*idf.get(t,0.1))**2
                         for t in sense_weights)) or 1e-9
    return num / (da * db)

def lesk_ar_discriminative(ctx_toks: list, sense: dict,
                            combined_weights: Counter,
                            exclude: str = '') -> float:
    """
    Config 4: blend standard IDF score with discriminative IDF score.
    Tokens unique to this sense (low df_others) get higher weight.
    """
    ctx_set = set(ctx_toks)
    disc    = disc_boost.get(sense['sense_id'], {})

    # Standard score (Config 3 representation)
    score_std  = lesk_ar_weighted(ctx_toks, combined_weights, idf_global)

    # Discriminative score — use disc_boost as the IDF
    common_d = ctx_set & set(disc.keys())
    if common_d:
        num_d = sum(combined_weights.get(t,1)*disc[t]**2 for t in common_d)
        da_d  = math.sqrt(sum(disc.get(t,0.1)**2 for t in ctx_set
                              if t in disc)) or 1e-9
        db_d  = math.sqrt(sum((combined_weights.get(t,1)*disc[t])**2
                              for t in disc)) or 1e-9
        score_disc = num_d / (da_d * db_d)
    else:
        score_disc = 0.0

    return DISC_ALPHA * score_std + DISC_BETA * score_disc

# ════════════════════════════════════════════════════════════════════════════
#  STAGE 4a — SENSE IDENTIFICATION EVALUATION
# ════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    ('Baseline (binary)',               repr_baseline,       'standard'),
    ('Gloss Priority (×3)',             repr_gloss_priority, 'standard'),
    ('Frequency Weighting',             repr_frequency,      'standard'),
    ('Combined (Priority + Frequency)', repr_combined,       'standard'),
    ('Discriminative IDF + Combined',   repr_combined,       'discriminative'),
]

print("="*64)
print("DiLAC Lesk-ar — Enhanced Evaluation")
print("="*64)
print(f"\nResource : {len(entries)} entries, {len(senses)} senses")
print(f"Stemming : {'ON' if USE_STEM else 'OFF (better for news Arabic)'}")
print(f"IDF docs : {N_docs}   vocab={len(idf_global):,}")
print(f"Instances: {sum(len(s['examples']) for s in senses):,}")
print(f"Gloss weight : W_GLOSS={W_GLOSS}")
print(f"Disc blend   : α={DISC_ALPHA} (std) + β={DISC_BETA} (disc)")

cls_all = {}   # config_name → list of result dicts

for cfg_name, repr_fn, score_mode in CONFIGS:
    results = []
    for sense in senses:
        gold_id = sense['sense_id']
        for ex in sense['examples']:
            ctx = preprocess(ex['text'])
            best_id='__none__'; best_sc=-1.0

            for cand in senses:
                w = repr_fn(cand, exclude=ex['text'])
                if score_mode == 'discriminative':
                    sc = lesk_ar_discriminative(ctx, cand, w, ex['text'])
                else:
                    sc = lesk_ar_weighted(ctx, w, idf_global)
                if sc > best_sc:
                    best_sc=sc; best_id=cand['sense_id']

            if best_sc == 0.0: best_id='__none__'
            results.append({'gold':gold_id,'pred':best_id,
                            'correct':best_id==gold_id,
                            'score':round(best_sc,4),
                            'ex_type':ex['type'],'word':sense['word']})
    cls_all[cfg_name] = results

def compute_metrics(subset):
    if not subset: return {}
    correct=sum(r['correct'] for r in subset); total=len(subset)
    tp=Counter(); fp=Counter(); fn=Counter()
    for r in subset:
        g=r['gold']; p=r['pred']
        if p==g: tp[g]+=1
        else: fp[p]+=1; fn[g]+=1
    all_c=set(tp)|set(fn); pr=[]; rc=[]; f1=[]
    for c in all_c:
        p_=tp[c]/(tp[c]+fp[c]) if (tp[c]+fp[c]) else 0.0
        r_=tp[c]/(tp[c]+fn[c]) if (tp[c]+fn[c]) else 0.0
        f_=2*p_*r_/(p_+r_) if (p_+r_) else 0.0
        pr.append(p_); rc.append(r_); f1.append(f_)
    zo=sum(1 for r in subset if r['score']==0.0)
    return {'total':total,'correct':correct,
            'accuracy': round(correct/total*100,2),
            'precision':round(sum(pr)/len(pr)*100,2) if pr else 0.0,
            'recall':   round(sum(rc)/len(rc)*100,2) if rc else 0.0,
            'f1':       round(sum(f1)/len(f1)*100,2) if f1 else 0.0,
            'zero_overlap_pct':round(zo/total*100,1)}

# ── Comparison table ─────────────────────────────────────────────────────
print(f"\n── Sense Identification: Configuration Comparison ──────────")
print(f"{'Config':<36} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
print('─'*66)
best_cfg = None; best_f1 = 0.0

for cfg_name, _, _ in CONFIGS:
    m = compute_metrics(cls_all[cfg_name])
    tag = ' ←' if m['f1'] > best_f1 else ''
    if m['f1'] > best_f1:
        best_f1=m['f1']; best_cfg=cfg_name
    print(f"{cfg_name:<36} {m['accuracy']:>6.2f}% "
          f"{m['precision']:>6.2f}% {m['recall']:>6.2f}% "
          f"{m['f1']:>6.2f}%{tag}")

print(f"\n  Random baseline (1/57): 1.75%")
print(f"\n  Best config: {best_cfg}")

# ── Detailed breakdown for best config ───────────────────────────────────
best_results = cls_all[best_cfg]
print(f"\n── Best config breakdown ────────────────────────────────────")
m_all  = compute_metrics(best_results)
m_prim = compute_metrics([r for r in best_results if r['ex_type']=='example'])
m_add  = compute_metrics([r for r in best_results if r['ex_type']=='addExample'])

for label, m in [('Overall', m_all), ('Primary examples', m_prim),
                 ('addExamples', m_add)]:
    print(f"\n  {label}:")
    print(f"    Accuracy  : {m['accuracy']:.2f}%")
    print(f"    Precision : {m['precision']:.2f}%  (macro)")
    print(f"    Recall    : {m['recall']:.2f}%  (macro)")
    print(f"    F1 score  : {m['f1']:.2f}%  (macro)")
    print(f"    Instances : {m['total']:,}")

# ── ظهر breakdown ─────────────────────────────────────────────────────────
zhr=[r for r in best_results if r['word']=='ظهر']
if zhr:
    print(f"\n  ظهر (polysemous — 2 senses) [{best_cfg}]:")
    for gs in sorted(set(r['gold'] for r in zhr)):
        sub=[r for r in zhr if r['gold']==gs]
        nc=sum(1 for r in sub if r['correct'])
        print(f"    {gs.split('::')[1]:22s}: {nc}/{len(sub)} ({nc/len(sub)*100:.1f}%)")

# ── Per-word table for best config ───────────────────────────────────────
wd=defaultdict(list)
for r in best_results: wd[r['word']].append(r)
wa=sorted([(w,sum(r['correct'] for r in rs)/len(rs)*100,len(rs))
           for w,rs in wd.items()],key=lambda x:-x[1])
print(f"\n  Per-word accuracy [{best_cfg}]:")
print(f"    {'Word':14s} {'Acc':>7} {'n':>4}")
print(f"    {'─'*28}")
for w,a,n in wa:
    print(f"    {w:14s} {a:>6.1f}%  {n:>3}  {'█'*int(a/10)}")

# ════════════════════════════════════════════════════════════════════════════
#  STAGE 4b — AWSS SEMANTIC SIMILARITY
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("Stage 4b — AWSS Semantic Similarity (three-way comparison)")
print("="*64)

awss_out = {}
if not AWSS_XLSX.exists():
    print(f"  ⚠️  Not found: {AWSS_XLSX}")
else:
    try:
        import openpyxl
        from scipy.stats import pearsonr

        wb   = openpyxl.load_workbook(AWSS_XLSX, read_only=True)
        rows = list(wb.active.iter_rows(values_only=True))
        wb.close()

        header = rows[0] if rows else ()
        fmt_b  = isinstance(header[0] or '', str) and 'Word' in str(header[0])
        has_machine = True  # new format always has machine col

        pairs=[]
        for i,row in enumerate(rows):
            if i==0: continue
            if fmt_b:
                en1   = str(row[0] or '').strip()
                en2   = str(row[1] or '').strip()
                human = float(row[2]) if row[2] not in (None,'') else None
                mraw  = row[3]
                machine = (None if mraw is None or str(mraw).strip() in ('-','')
                           else float(mraw))
                ar1 = str(row[4] or '').strip().split('/')[0].strip()
                ar2 = str(row[5] or '').strip().split('/')[0].strip()
                if not en1 or human is None: continue
                pairs.append({'w1':ar1,'w2':ar2,'human':round(human,4),
                              'machine':machine,'en1':en1,'en2':en2})
            else:
                if not isinstance(row[0] or '',int): continue
                pairs.append({'w1':str(row[4]).strip(),'w2':str(row[3]).strip(),
                              'human':round(float(row[5]),4),
                              'machine':None,'en1':str(row[1]).strip(),
                              'en2':str(row[2]).strip()})

        print(f"  Pairs: {len(pairs)}  Machine ratings: "
              f"{sum(1 for p in pairs if p['machine'] is not None)}/35")

        # Build word → senses lookup
        word_senses = defaultdict(list)
        for s in senses:
            word_senses[norm_surface(s['word'])].append(s)

        def pair_similarity_cfg(w1, w2, repr_fn, score_mode):
            n1=norm_surface(w1); n2=norm_surface(w2)
            s1=word_senses.get(n1,[]); s2=word_senses.get(n2,[])
            if not s1 or not s2: return 0.0
            best=0.0
            for a in s1:
                wa_=repr_fn(a)
                for b in s2:
                    wb_=repr_fn(b)
                    if score_mode=='discriminative':
                        sc=(DISC_ALPHA*lesk_ar_weighted(
                                list(wa_.keys()), wb_, idf_global) +
                            DISC_BETA*lesk_ar_weighted(
                                list(wa_.keys()), wb_,
                                disc_boost.get(b['sense_id'],{})))
                    else:
                        toks_a=list(wa_.keys())
                        sc=lesk_ar_weighted(toks_a, wb_, idf_global)
                    if sc>best: best=sc
            return best

        # AWSS comparison table
        print(f"\n── AWSS r comparison across configurations ──────────────")
        print(f"{'Config':<36} {'r vs Human':>10} {'r vs Machine':>13}")
        print('─'*62)

        awss_cfg_results = {}
        for cfg_name, repr_fn, score_mode in CONFIGS:
            for p in pairs:
                p[f'dilac_{cfg_name}'] = round(
                    pair_similarity_cfg(p['w1'],p['w2'],repr_fn,score_mode),4)
            hs=[p['human'] for p in pairs]
            ds=[p[f'dilac_{cfg_name}'] for p in pairs]
            r_h,_=pearsonr(hs,ds)

            ms_pairs=[(p['machine'],p[f'dilac_{cfg_name}'])
                      for p in pairs if p['machine'] is not None]
            r_m,_=pearsonr([x[0] for x in ms_pairs],
                           [x[1] for x in ms_pairs]) if ms_pairs else (0,0)

            awss_cfg_results[cfg_name]={'r_human':round(r_h,4),
                                        'r_machine':round(r_m,4)}
            print(f"{cfg_name:<36} {round(r_h,4):>10.4f} {round(r_m,4):>13.4f}")

        # Reproduce AWSS algorithm
        mach_h=[p['human']   for p in pairs if p['machine'] is not None]
        mach_m=[p['machine'] for p in pairs if p['machine'] is not None]
        r_awss_repr,_=pearsonr(mach_h,mach_m)

        print(f"\n  AWSS algorithm (reproduced)  : r = {round(r_awss_repr,4)}"
              f"  (paper: 0.894 ✅)" if abs(r_awss_repr-0.894)<0.01
              else f"\n  AWSS algorithm (reproduced)  : r = {round(r_awss_repr,4)}")
        print(f"  Human ceiling (Almarsoomi)   : r = {HUMAN_CEILING}")

        # Best AWSS config
        best_awss = max(awss_cfg_results.items(),
                        key=lambda x: x[1]['r_human'])
        print(f"\n  Best config for AWSS: {best_awss[0]}")
        print(f"    r vs Human   : {best_awss[1]['r_human']}")
        print(f"    r vs Machine : {best_awss[1]['r_machine']}")

        # Pair-level for best config
        best_col = f"dilac_{best_awss[0]}"
        sorted_p = sorted(pairs, key=lambda p:p['human'])
        print(f"\n── Sample pairs (best config) ───────────────────────────")
        print(f"  {'English pair':<26} {'Arabic pair':<20}"
              f" {'Human':>6} {'AWSS':>6} {'DiLAC':>6}")
        print(f"  {'─'*68}")
        for i,p in enumerate(sorted_p):
            if i not in list(range(5))+list(range(16,19))+list(range(32,35)):
                if i==5: print(f"  ...")
                if i==19: print(f"  ...")
                continue
            ms=f"{p['machine']:>6.3f}" if p['machine'] is not None else "  N/A"
            print(f"  {p['en1']+'/'+p['en2']:<26} "
                  f"{p['w1']+'/'+p['w2']:<20}"
                  f" {p['human']:>6.3f} {ms} {p[best_col]:>6.4f}")

        awss_out = {
            'configs': awss_cfg_results,
            'awss_reproduced_r': round(r_awss_repr,4),
            'best_config': best_awss[0],
            'best_r_human': best_awss[1]['r_human'],
        }

    except ImportError:
        print("  pip install scipy openpyxl")
    except Exception as e:
        import traceback; traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
#  SAVE + FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
cls_metrics = {cfg: compute_metrics(res)
               for cfg, res in cls_all.items()}

output = {
    'hyperparams': {'stemming':USE_STEM,'W_GLOSS':W_GLOSS,
                    'DISC_ALPHA':DISC_ALPHA,'DISC_BETA':DISC_BETA},
    'sense_identification': cls_metrics,
    'awss_similarity': awss_out,
}
with open(OUT_PATH,'w',encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n{'='*64}")
print("FINAL SUMMARY — ALL CONFIGURATIONS")
print("="*64)
print(f"\n{'Config':<36} {'Acc':>7} {'F1':>7}"
      + (f" {'AWSS r':>8}" if awss_out else ''))
print('─'*64)
for cfg_name,_,_ in CONFIGS:
    m = cls_metrics[cfg_name]
    r_str = (f" {awss_out['configs'][cfg_name]['r_human']:>8.4f}"
             if awss_out and cfg_name in awss_out.get('configs',{}) else '')
    best_tag = ' ◀' if cfg_name==best_cfg else ''
    print(f"{cfg_name:<36} {m['accuracy']:>6.2f}% "
          f"{m['f1']:>6.2f}%{r_str}{best_tag}")

print(f"\n  Reference values:")
print(f"    Random baseline (1/57)    :  1.75% accuracy")
print(f"    Human ceiling (AWSS)      :  r = 0.893")
print(f"    AWSS algorithm            :  r = 0.894")
if awss_out:
    print(f"    AWSS reproduced           :  r = {awss_out['awss_reproduced_r']}")
print(f"\n💾  Full results → {OUT_PATH.name}")
