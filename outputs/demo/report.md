# FraudLens experiment report

**Source:** SYNTHETIC DEMO

**Run time (UTC):** 2026-09-28T18:07:00.826634+00:00

**Interpretation:** If the source is synthetic, these are demonstration results, not Kaggle or real-world performance.

## Experiment settings

- Split: Stratified random
- Seed: 61
- PCA retained variance: 95%
- Decision policy: Balanced F1
- Validated rows: 6,000
- Duplicate rows removed: 0
- Gradient Boosting included: True

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
| Gradient Boosting | 0.8141 | 0.7893 | 0.8611 | 0.7561 | 0.8052 | 0.0300 |

## Selected model

**Gradient Boosting**, selected on validation data.

The frozen threshold is **-0.067223**. On the test split, this model achieved precision **86.1%**, recall **75.6%**, and average precision **0.7893**. It produced **5 false alarms** and **10 missed fraud transactions**.

## Reproducibility

Dataset SHA-256: `dc1421bd3dfe2fab6baba317744d3e02221a38dece3210f56e7c4088a5134613`

python 3.9.6, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.6.1

## Warnings and limitations

- Gradient Boosting: Could not find the number of physical cores for the following reason: invalid literal for int() with base 10: '' Returning the number of logical cores instead. You can silence this warning by setting LOKY\_MAX\_CPU\_COUNT to the number of cores you want to use.
- Decision scores are uncalibrated margins, not probabilities.
- Thresholds and preprocessing use training/validation data only; no post-evaluation refit occurs.
- Repeated configuration selection from test results biases reported performance.
- Linear-model feature contributions explain the calculation, not the cause of fraud; these explanations are unavailable for boosted trees.
- Results from one holdout do not establish production readiness or demographic fairness.
- Predictions support human review and never automatically block payments.

## Uncertainty of selected-model test rates

Approximate 95% Wilson intervals assume independent transactions and a fixed model. They exclude uncertainty from model selection, repeated tuning and data drift.

| Rate | Estimate | Lower 95% | Upper 95% | Denominator |
| --- | ---: | ---: | ---: | ---: |
| Precision | 86.1% | 71.3% | 93.9% | 36 |
| Recall | 75.6% | 60.7% | 86.2% | 41 |
| Review rate | 3.0% | 2.2% | 4.1% | 1200 |
