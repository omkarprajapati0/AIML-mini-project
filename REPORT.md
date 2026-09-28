# EXPERIMENT NO. 10
## Credit Card Fraud Anomaly Detection

**Name:** Omkar M Prajapati\
**Roll Number:** 61\
**Class:** TE IT\
**Date of Conduction:** __________\
**Date of Submission:** __________

## Aim
Design and implement a supervised credit card fraud detection workflow using Logistic Regression, SVM, and PCA, with data validation, reproducible evaluation, model interpretation and batch transaction review.

## Lab outcomes
2345116.5: Evaluate ML models using standard evaluation metrics and validation techniques.\
2345116.6: Analyse ethical issues, bias, fairness, and societal impacts of AI systems.

## Tools
Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, and the Kaggle Credit Card Fraud dataset. Record the actual package versions from the exported experiment JSON.

## Problem statement
Identify potentially fraudulent transactions while measuring the tradeoff between missed fraud and false alarms. Fraud is the positive class. The project uses supervised classification because labelled examples are available.

## Algorithms
Logistic Regression and Linear SVM learn linear decision functions. Balanced class weights account for class imbalance during fitting. PCA compresses standardized features while retaining a configured share of variance. The four compared pipelines are Logistic Regression, Linear SVM, PCA + Logistic Regression, and PCA + Linear SVM. Kaggle V1–V28 are already anonymized PCA components, so additional PCA is evaluated rather than assumed beneficial.

## Implementation
1. Read a strict UTF-8 CSV and validate required numeric fields and binary target labels.
2. Remove exact duplicates and reject identical features with conflicting labels.
3. Create disjoint 60/20/20 partitions using stratification or chronological ordering. Chronological mode keeps equal timestamps in one partition and requires both classes in each period.
4. Fit imputation, scaling, optional PCA and a class-weighted classifier only on training rows.
5. Select each threshold using validation F1, F2, or maximum precision subject to a validation recall target. Select the recommended model using validation average precision.
6. Report performance on held-out test rows using the frozen thresholds.
7. Inspect confusion matrices, precision–recall curves, validation threshold curves, PCA variance and standardized feature weights.
8. Score unseen transactions, preserve their metadata, filter the queue, and export decisions with model and threshold information.
9. Export a run manifest with a dataset fingerprint, split counts, settings, package versions and metrics.

## Experimental record

| Setting | Value from the completed run |
| --- | --- |
| Dataset source and SHA-256 | __________ |
| Validated rows / duplicates removed | __________ |
| Split strategy and train / validation / test counts | __________ |
| Fraud count in each split | __________ |
| PCA retained variance | __________ |
| Threshold policy and recall target, if used | __________ |
| Random seed | 61 |
| Recommended model | __________ |
| Selected model threshold | __________ |

## Results and interpretation
Insert the exported model comparison table from your real Kaggle run. The bundled synthetic results verify the workflow and must not be presented as real-data performance.

Precision measures the fraction of alerts that are fraud. Recall measures the fraction of actual fraud that is caught. F1 balances precision and recall; F2 gives recall more weight. Average precision summarizes fraud-ranking quality, while ROC-AUC measures discrimination across thresholds. Review rate estimates how much of the held-out batch is flagged, not the workload of an actual deployment. Accuracy alone is misleading for rare fraud.

Discuss the validation-selected model’s test results, false alarms, missed fraud, and whether additional PCA helped. If using a recall target, compare achieved validation and test recall and explain why the target is not guaranteed on unseen data. If comparing split strategies, report them as separate experiments and record their settings. Avoid choosing a final configuration repeatedly from test performance.

## Verification
Run `python -m unittest discover -v` and `python -m pip check`. Tests cover data boundaries, training-only preprocessing, disjoint splits, chronological boundaries, validation-only decisions, missing values, prediction order, artifact roundtrips, CLI execution and dashboard interactions.

## Screenshots to attach
1. Overview with the real source, class distribution and data quality visible.
2. Four-model test comparison and validation-selected recommendation.
3. Confusion matrix and precision–recall curves.
4. Validation decision policy, PCA explained variance, and feature coefficients.
5. Split audit and experiment settings.
6. Batch review queue and export options.

## Ethics and limitations
False positives can inconvenience genuine customers; false negatives can allow fraud. Human review and secure data handling are necessary. Model scores are uncalibrated margins, not fraud probabilities. Feature coefficients describe model associations and do not establish causes. Anonymized features limit demographic fairness analysis. A chronological holdout better separates earlier and later periods but still requires future-period validation and drift monitoring. Small fraud counts produce unstable estimates. The application does not make automatic payment decisions.

## Conclusion template — complete after the real run
On the held-out ________ split, the validation-selected model ________ achieved precision ________, recall ________, F1 ________, and average precision ________. The decision policy ________ produced ________ false alarms and ________ missed fraud transactions. Additional PCA ________. These results demonstrate the effect of dimensionality reduction and threshold selection on measured fraud detection and review workload.
