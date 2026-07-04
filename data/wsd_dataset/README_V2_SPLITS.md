# WSD Splits — IMPORTANT

The paper (Table 2 and all reported results) uses the **v2** word-level splits:

| Split | Instances |
|-------|-----------|
| train | 9,842     |
| dev   | 2,018     |
| test  | 2,050     |

## Files you must add before the results are reproducible

The v2 split files are **not committed to this repository** because they are
regenerated deterministically from the DiLAC gold database. To produce them:

1. Place the DiLAC gold database at `data/dilac/dilac_gold.db`
   (available on request / from the companion resource paper's release).

2. Run the split builder from the project root:

   ```bash
   python src/lexical/rebuild_splits_enriched.py
   ```

   This writes:
   - `data/wsd_dataset/gold_train_v2.json`  (9,842 instances)
   - `data/wsd_dataset/gold_dev_v2.json`    (2,018 instances)
   - `data/wsd_dataset/gold_test_v2.json`   (2,050 instances)

3. Verify zero lexical leakage:

   ```bash
   python src/lexical/verify_splits.py
   ```

## About the v1 files in this directory

`gold_dev_v1.json` and `gold_test_v1.json` are the **earlier** v1 splits
(3,307 / 3,285 instances). They are retained only for provenance and are
**not** the splits used in the paper. Do not use them to reproduce the
reported numbers — use the v2 splits generated as above.
