# FraudLens experiment report

**Source:** SYNTHETIC DEMO

**Run time (UTC):** 2026-09-27T19:35:03.498125+00:00

**Interpretation:** If the source is synthetic, these are demonstration results, not Kaggle or real-world performance.

## Experiment settings

- Split: Stratified random
- Seed: 61
- PCA retained variance: 95%
- Decision policy: Balanced F1
- Validated rows: 6,000
- Duplicate rows removed: 0

## Split audit

| Partition | Rows | Fraud | Prevalence |
| --- | ---: | ---: | ---: |
| Train | 3,600 | 122 | 3.39% |
| Validation | 1,200 | 41 | 3.42% |
| Test | 1,200 | 41 | 3.42% |

## Model comparison

The recommendation uses validation average precision. All other metrics below use the held-out test set.

| Model | Validation AP | Test AP | Precision | Recall | F1 | Review rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7529 | 0.7625 | 0.9375 | 0.7317 | 0.8219 | 0.0267 |
| Linear SVM | 0.7594 | 0.7668 | 0.9091 | 0.7317 | 0.8108 | 0.0275 |
| PCA + Logistic Regression | 0.6718 | 0.6168 | 0.7576 | 0.6098 | 0.6757 | 0.0275 |
| PCA + Linear SVM | 0.6704 | 0.6134 | 0.7586 | 0.5366 | 0.6286 | 0.0242 |

## Selected model

**Linear SVM**, selected on validation data.

The frozen threshold is **0.626243**. On the test split, this model achieved precision **90.9%**, recall **73.2%**, and average precision **0.7668**. It produced **3 false alarms** and **11 missed fraud transactions**.

## Reproducibility

Dataset SHA-256: `dc1421bd3dfe2fab6baba317744d3e02221a38dece3210f56e7c4088a5134613`

python 3.9.6, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1

## Warnings and limitations

- No training or small-partition warnings were recorded.
- Decision scores are uncalibrated margins, not probabilities.
- Thresholds and preprocessing use training/validation data only; no post-evaluation refit occurs.
- Repeated configuration selection from test results biases reported performance.
- Feature contributions explain the linear calculation, not the cause of fraud.
- Results from one holdout do not establish production readiness or demographic fairness.
- Predictions support human review and never automatically block payments.
