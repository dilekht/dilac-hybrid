"""
Verification script for gold WSD split files.
Run locally before uploading to Google Drive.

Expected:
    gold_train.json : 9,425 instances
    gold_dev.json   : 1,910 instances
    gold_test.json  : 3,285 instances
    Zero word-level leakage between all splits
    Each instance has: lemma, context, target_sense, all_senses
"""

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
DIR  = ROOT / 'data' / 'wsd_dataset'

EXPECTED = {
    'gold_train.json': 9425,
    'gold_dev.json':   1910,
    'gold_test.json':  3285,
}

REQUIRED_KEYS = {'lemma', 'lemma_norm', 'context',
                 'target_sense', 'all_senses', 'n_senses'}

print("="*58)
print("Verifying gold WSD split files")
print("="*58)

all_ok   = True
splits   = {}

for fname, expected_n in EXPECTED.items():
    path = DIR / fname
    print(f"\n── {fname} ──────────────────────────────")

    # 1. File exists?
    if not path.exists():
        print(f"  ❌  NOT FOUND at {path}")
        all_ok = False
        continue
    print(f"  ✅  Found ({path.stat().st_size/1e6:.1f} MB)")

    # 2. Valid JSON?
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌  JSON parse error: {e}")
        all_ok = False
        continue

    # 3. Instance count
    n = len(data)
    ok_n = n == expected_n
    mark = '✅' if ok_n else '⚠️ '
    print(f"  {mark}  Instances: {n:,}  (expected {expected_n:,})")
    if not ok_n:
        all_ok = False

    # 4. Required keys
    if data:
        missing_keys = REQUIRED_KEYS - set(data[0].keys())
        if missing_keys:
            print(f"  ❌  Missing keys in first instance: {missing_keys}")
            all_ok = False
        else:
            print(f"  ✅  All required keys present")

    # 5. No empty contexts or missing gold senses
    empty_ctx  = sum(1 for inst in data if not inst.get('context','').strip())
    empty_gold = sum(1 for inst in data if not inst.get('target_sense',''))
    empty_sens = sum(1 for inst in data if not inst.get('all_senses'))
    if empty_ctx or empty_gold or empty_sens:
        print(f"  ⚠️   Empty context: {empty_ctx}  "
              f"Missing gold: {empty_gold}  Empty senses: {empty_sens}")
    else:
        print(f"  ✅  No empty contexts, gold senses, or sense lists")

    # 6. Sense count distribution
    n_senses = Counter(inst.get('n_senses', len(inst.get('all_senses',[])))
                       for inst in data)
    avg = sum(k*v for k,v in n_senses.items()) / n if n else 0
    print(f"  ✅  Avg senses/word: {avg:.2f}  "
          f"(min={min(n_senses)}, max={max(n_senses)})")

    splits[fname] = {
        'data':    data,
        'lemmas':  set(inst['lemma_norm'] for inst in data),
        'n':       n,
    }

# 7. Zero leakage check
if len(splits) == 3:
    print(f"\n── Leakage check ────────────────────────────────────")
    tr = splits['gold_train.json']['lemmas']
    dv = splits['gold_dev.json']['lemmas']
    te = splits['gold_test.json']['lemmas']

    tr_dv = tr & dv
    tr_te = tr & te
    dv_te = dv & te

    for pair, overlap in [('Train ∩ Dev',  tr_dv),
                           ('Train ∩ Test', tr_te),
                           ('Dev ∩ Test',   dv_te)]:
        if overlap:
            print(f"  ❌  {pair}: {len(overlap)} shared lemmas — LEAKAGE!")
            all_ok = False
        else:
            print(f"  ✅  {pair}: 0 shared lemmas — clean")

    # 8. Unique lemma counts
    print(f"\n── Unique lemmas ────────────────────────────────────")
    for fname, info in splits.items():
        print(f"  {fname:20s}: {len(info['lemmas']):,} unique lemmas")

    # 9. Sample instance from test
    print(f"\n── Sample test instance ─────────────────────────────")
    sample = splits['gold_test.json']['data'][0]
    print(f"  Lemma   : {sample['lemma']}")
    print(f"  Context : {sample['context'][:70]}...")
    print(f"  Gold    : {sample['target_sense']}")
    print(f"  Senses  : {len(sample['all_senses'])} candidates")

# Final verdict
print(f"\n{'='*58}")
if all_ok:
    print("✅  ALL CHECKS PASSED — files are ready to upload to Drive")
    print("\nNext step:")
    print("  Upload these 3 files to MyDrive/dilac_project/ on Google Drive:")
    for fname in EXPECTED:
        p = DIR / fname
        if p.exists():
            print(f"    {p}")
else:
    print("❌  SOME CHECKS FAILED — fix the issues above before uploading")
    print("   If instance counts differ, re-run dataset_builder_fixed.py")
print("="*58)
