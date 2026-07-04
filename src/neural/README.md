# Neural (Hybrid) Track

These scripts run on a GPU (Kaggle T4 or Colab). They were developed and
verified on Kaggle.

## Setup

1. Upload `gold_train_v2.json`, `gold_dev_v2.json`, `gold_test_v2.json` as a
   Kaggle dataset and attach it to the notebook (right panel → Add Data).
2. Set the accelerator to **GPU T4** (Settings → Accelerator).
3. Each script auto-detects the dataset path with:

   ```python
   import os, glob
   _c = glob.glob('/kaggle/input/**/gold_test_v2.json', recursive=True)
   INPUT_DIR = os.path.dirname(_c[0])
   ```

   so it works regardless of the dataset's folder name.

## Run order (one GPU session)

| Step | Script | Output | Time |
|------|--------|--------|------|
| 8  | `step8_kaggle.py`        | zero-shot AraBERT + CAMeLBERT       | ~20 min |
| 9a | `step9_kaggle_v2.py`     | fine-tuned CAMeLBERT checkpoint     | ~40 min |
| 9b | `step9_arabert_v2.py`    | fine-tuned AraBERT checkpoint       | ~40 min |
| 10–12 | `steps10_12_kaggle_v3.py` | fusion, two-stage, ensemble, ablation | ~15 min |

Checkpoints are written to `/kaggle/working/glossbert_models/`. Kaggle wipes
`/kaggle/working/` between sessions, so run 9a → 9b → 10–12 in one session, or
commit the notebook (Save Version → Save & Run All) to persist checkpoints as
output.

## Key design choices (see paper Section 5)

- **GlossBERT** is a binary classifier on `[CLS] context [SEP] gloss+examples [SEP]`,
  not a cosine measure. The target word in the context is marked with
  `[TGT] ... [/TGT]`.
- **Negative subsampling** caps negatives at 2 per instance to prevent the
  1:3.5 class imbalance caused by high-polysemy words.
- **MAX_LEN = 256** (recovers ~1.5 pts over 128).
- **Leave-one-out** excludes the exact test sentence from its sense
  representation in both training and evaluation.
- **CAMeLBERT-MSA** is the primary model; AraBERT is for comparison. The
  two-model ensemble did **not** help (the weaker model drags the stronger),
  a result reported honestly in the paper.

## Verification cells

The repository includes inline checks (in the paper's supplementary notes)
for: zero leakage, the 23.4% context-in-examples rate, and the gold/non-gold
enrichment balance (2.77 vs. 3.00). These can be rerun to confirm the
evaluation is not optimistically biased.
