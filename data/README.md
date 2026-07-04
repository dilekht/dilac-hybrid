# Data

## DiLAC v1.0 (`dilac/`)

- `dilac_awss.json` — the AWSS evaluation subset: 56 entries, 57 senses, 741
  human-verified example sentences (avg. 13 per sense). Every example is drawn
  from contemporary Arabic news sources and was verified by annotators.
  Used by `src/lexical/lesk_ar_awss_eval.py`.

The full DiLAC v1.0 subset (613 example-annotated entries, 1,738 senses) and
the SQLite database `dilac_gold.db` used to build the WSD splits are released
on request and via the project release page (too large for the main tree).

**License.** DiLAC v1.0 is released for research use. Redistribution of the
example sentences is subject to the terms of their original news sources;
the annotations and structure are released under CC BY-NC 4.0.

## AWSS benchmark (`awss/`)

- `benchmark_data_awss_pairs.xlsx` — the 35-pair AWSS evaluation table
  (Almarsoomi et al. 2014) with human and machine (Arabic WordNet) ratings.
  Redistributed for reproducibility under the original authors' terms; please
  cite the original paper.

## WSD splits (`wsd_dataset/`)

This directory ships the **v1 splits** used in the pilot experiments:

- `gold_dev_v1.json` (3,307 instances), `gold_test_v1.json` (3,285 instances).
- `gold_train_v1.json` (14,491 instances) — released via the project release
  page due to size (~27 MB).

The **v2 splits** reported as the paper's main results (stratified by
sense-count band, example-enriched; dev 2,018 / test 2,050) are regenerated
deterministically from `dilac_gold.db` with:

```bash
python src/lexical/rebuild_splits_enriched.py   # writes gold_*_v2.json
```

We ship v1 directly and regenerate v2 from source so the splits can be audited
exact split-construction and enrichment logic rather than trusting an opaque
file. Run `src/lexical/verify_splits.py` on either version to confirm zero
leakage and inspect the sense-count distributions.

Each instance carries: `lemma`, `context`, `target_sense`, `all_senses`
(each candidate with `gloss` and `examples`), and `instance_example_id`
(for leave-one-out).

### Leave-one-out note

23.4% of test instances have their own context sentence present among their
gold sense's examples (a consequence of single-source construction). The
evaluation scripts exclude the exact test sentence from its sense
representation at both training and inference time. After exclusion, gold
senses are enriched with 2.77 examples on average versus 3.00 for non-gold
senses, so there is no example-count shortcut favoring the gold answer.
