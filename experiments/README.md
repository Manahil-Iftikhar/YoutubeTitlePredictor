# Offline engagement baseline

This maintained experiment replaces the unusable feature assumptions of the historical scripts with a small, reproducible analysis of the existing CSV. The historical scripts are retained for provenance and are not imported.

## Run from the repository root

Use Python 3.12 in a separate virtual environment:

```bash
python -m pip install -r requirements-ml.txt
python -m unittest discover -s experiments -p 'test_*.py'
python experiments/engagement_baseline.py
```

No API key, network collection, model download, or Django setup is needed after installing dependencies. Outputs go to `reports/engagement/` by default. Use `--data` and `--output` to change paths.

## Recorded result

The existing snapshot contains 34 usable rows: 22 training rows and 12 test rows. Entire publication dates remain on one side of the split. Preprocessing is fitted only on training data. Features are title length and category; views, likes, and comments define the target and are never input features. No invented competition or search-volume values are used.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Training median baseline | 1,554,635.33 | 1,683,005.49 | -2.9407 |
| Ridge with log-transformed target | 2,060,246.68 | 2,205,204.14 | -5.7655 |

**The ridge model underperforms the baseline.** Both negative R² values indicate poor fit relative to the test-set mean reference. The correct conclusion is that these limited features and this sample do not support reliable predictions, not that a useful forecasting model has been established.

Ridge alpha is fixed at 10; no model selection or tuning was performed on the test set. Predictions are clipped at zero. Machine-readable [metrics](../reports/engagement/metrics.json) record the source SHA-256 and package versions; [predictions](../reports/engagement/predictions.csv) show every held-out prediction.

## Interpretation limits

- The target is a snapshot sum of views, likes, and comments, not future engagement.
- Publication date is not the time statistics were observed. A publication-ordered split does not remove exposure-age confounding or create a valid forecasting benchmark.
- The dataset is tiny and selected from trending videos. Channel effects, collection time, and video IDs are unavailable.
- Deduplication uses title and publication timestamp as an imperfect proxy for video identity.
- Results provide no evidence of title-generation quality, SEO improvement, causal effects, or production readiness.
- The CSV is an existing repository asset; this work does not establish its provenance or redistribution rights.

The four regression tests cover invalid data and duplicates, disjoint date partitions, insufficient input, and deterministic reports. This experiment is independent of the metadata-search web app.
