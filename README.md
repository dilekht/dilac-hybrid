# DiLAC Hybrid WSD

Code and resources for **"DiLAC Hybrid WSD: Integrating Lexical Resources with Transformer Models for Arabic Word Sense Disambiguation."**

This repository provides the code and data to reproduce every number in the paper. It is organized into two evaluation tracks:

1. **Knowledge-based track** — a Lesk-ar measure built only from DiLAC glosses and examples, evaluated on sense identification and on the AWSS semantic-similarity benchmark.
2. **Hybrid WSD track** — zero-shot and fine-tuned (GlossBERT) transformer pipelines, plus fusion/two-stage/ensemble combinations, evaluated on a leakage-controlled WSD test set.

---

## Headline results

| | Result |
|---|---|
| MFS baseline (test-internal) | 28.49% (macro-F1 14.35%) |
| DiLAC Lesk-ar (knowledge-only) | 34.78% (macro-F1 34.91%) |
| CAMeLBERT-MSA zero-shot WSD | 50.39% |
| CAMeLBERT-MSA fine-tuned (GlossBERT) | **69.56%** (+19.2 pts, same test set) |
| Best fusion | **69.56%** (macro-F1 66.28%) |
| Ablation: remove BERT | **−61.3 pts** |

All numbers derive from a single reproducible run on the v2 word-level test set
(2,050 instances) and live in [`results/all_results.json`](results/all_results.json),
the single source of truth for the figures. The best fusion weights are
α=0.0, β=0.4, γ=0.0 — i.e. the runtime lexical and domain terms receive zero
weight, so the "fusion" equals scaled GlossBERT (see the paper, §5–6).

> **Reproducing the results:** the v2 split files are regenerated from the DiLAC
> gold database and are not committed here. See
> [`data/wsd_dataset/README_V2_SPLITS.md`](data/wsd_dataset/README_V2_SPLITS.md)
> for the one-command procedure. Run order for the neural pipeline (on a GPU, e.g.
> Kaggle T4): `step8_kaggle.py` → `step9_kaggle_v2.py` → `step9_arabert_v2.py` →
> `steps10_12_kaggle_v3.py` → `extract_errors.py`.

---

## Repository layout

```
dilac-hybrid/
├── src/
│   ├── lexical/                  # knowledge-based track (runs locally, CPU)
│   │   ├── lesk_ar_awss_eval.py      # Lesk-ar: sense-ID + AWSS, 5 configs
│   │   ├── rebuild_splits_enriched.py# build stratified, enriched WSD splits
│   │   └── verify_splits.py          # leakage + distribution checks
│   └── neural/                   # hybrid track (runs on Kaggle/Colab GPU)
│       ├── step8_kaggle.py           # zero-shot AraBERT + CAMeLBERT
│       ├── step9_kaggle_v2.py        # GlossBERT fine-tuning (CAMeLBERT)
│       ├── step9_arabert_v2.py       # GlossBERT fine-tuning (AraBERT)
│       ├── steps10_12_kaggle_v3.py   # fusion, two-stage, ensemble, ablation
│       └── extract_errors.py         # error analysis (Table 4, real misclassifications)
├── data/
│   ├── dilac/                    # DiLAC v1.0 JSON + AWSS subset (see note)
│   ├── awss/                     # AWSS benchmark pairs
│   └── wsd_dataset/              # gold_*_v2.json splits (see note)
├── results/
│   ├── all_results.json          # canonical verified numbers
│   ├── tables/                   # per-script JSON outputs
│   └── figures/                  # generated PNGs (make_figures.py)
├── scripts/
│   └── make_figures.py           # regenerate all paper figures
├── requirements.txt
└── README.md
```

> **Data note.** The DiLAC v1.0 JSON and the derived `gold_*_v2.json` splits are released under the terms in `data/README.md`. The AWSS benchmark is redistributed per its original license; see `data/awss/README.md`.

---

## Reproducing the results

### 1. Knowledge-based track (local, no GPU)

```bash
pip install -r requirements.txt
# Lesk-ar: sense identification (5 configs) + AWSS similarity + 3-way comparison
python src/lexical/lesk_ar_awss_eval.py
```

This reproduces Tables 2–4 of the paper (sense-ID configurations, AWSS comparison) and writes `results/tables/lesk_ar_awss_eval.json`.

### 2. Build the WSD splits (local, requires the DiLAC database)

```bash
python src/lexical/rebuild_splits_enriched.py   # writes gold_*_v2.json
python src/lexical/verify_splits.py             # confirms zero leakage
```

### 3. Hybrid track (Kaggle/Colab GPU)

Upload the three `gold_*_v2.json` files as a dataset, then run in order:

```text
step8_kaggle.py          # zero-shot baselines      (~20 min)
step9_kaggle_v2.py       # fine-tune CAMeLBERT       (~40 min)
step9_arabert_v2.py      # fine-tune AraBERT         (~40 min)
steps10_12_kaggle_v3.py  # fusion/ensemble/ablation  (~15 min)
```

Each script auto-detects the input path via `glob`. See [`src/neural/README.md`](src/neural/README.md) for the path setup and the leave-one-out verification cells.

### 4. Regenerate figures

```bash
python scripts/make_figures.py   # reads results/all_results.json
```

---

## Methodological controls

This work emphasizes reproducibility and honest evaluation:

- **Word-level split** with verified zero lexical leakage between train/dev/test.
- **Same-test-set** comparison of zero-shot vs. fine-tuned (no cross-dataset claims).
- **Leave-one-out enrichment**: the exact test sentence is excluded from its own sense representation; gold vs. non-gold enrichment is balanced (2.77 vs. 3.00 examples) to rule out an example-count shortcut.
- **Header-sense exclusion** so the most-frequent-sense baseline is well-defined.
- All hyperparameters (fusion weights, ensemble weights) tuned on **dev only**.

Limitations are stated explicitly in the paper: single-source examples, 55 test lemmas, and partial (v1.0) resource coverage.

---

## Citation

```bibtex
@article{dilekh2026dilac,
  title   = {DiLAC Hybrid WSD: Integrating Lexical Resources with
             Transformer Models for Arabic Word Sense Disambiguation},
  author  = {Dilekh, Tahar and Mokeddem, Ayoub and
             Boulahia, Mohamed Abderrahmen and Benharzallah, Saber},
  journal = {ACM Transactions on Asian and Low-Resource Language
             Information Processing},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code: MIT (see `LICENSE`). Data: see `data/README.md`.
