# FraudLens · Credit Card Fraud Detection

Experiment 10 · AI/ML mini project\
Omkar M Prajapati · Roll No. 61 · TE IT

A local, interactive workspace for studying credit card fraud. Compare Logistic Regression and Linear SVM with and without PCA, choose a validation-based decision policy, inspect held-out results, and score a transaction batch for human review.

## What is included

- **Overview:** data quality, class imbalance, transaction amount distributions, and sample CSV downloads.
- **Model lab:** four comparable pipelines, confusion matrices, precision–recall curves, validation threshold tradeoffs, PCA variance, and feature coefficients.
- **Evaluation choices:** stratified random or chronological holdout, with F1, recall-focused F2, or a minimum validation recall target.
- **Transaction review:** model selection, amount and decision filters, sorting, pagination, per-record inspection, and full or filtered CSV exports.
- **Explain each prediction:** exact feature contributions that add up to the transaction score, including the baseline and any imputed fields.
- **Find transaction IDs:** literal, case-insensitive metadata search and configurable metadata columns. Leading zeros and text values such as `NA` stay intact.
- **Share results:** download a readable Markdown experiment report with measured results, settings, split counts and limitations.
- **Reproducible experiments:** dataset fingerprints, settings, split counts, package versions, timing, and downloadable JSON manifests.
- **CLI workflows:** training, persisted model bundles, and batch predictions without opening the dashboard.
- **Regression checks:** data validation, preprocessing isolation, threshold selection, chronological boundaries, artifacts, CLI execution, and dashboard workflows.

Synthetic demo data is available immediately. The Kaggle dataset is not bundled. Demo scores must not be reported as real-world or Kaggle performance.

## Quick start

Use Python 3.11 or 3.12 for a new environment.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address=127.0.0.1
```

### Windows (Command Prompt)

```bat
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address=127.0.0.1
```

Open [localhost:8501](http://localhost:8501). Stop with Ctrl+C. On later runs, activate the environment and run the final command. If dependencies were changed while the app was running, restart it.

The existing Python 3.9 environment also works with the compatible dependency versions selected by pip. On macOS with Python below 3.10, requirements use NumPy 1.26.4 to avoid the observed Apple Accelerate matrix-multiplication warnings in NumPy 2.0.2. This is a dependency fix; the application does not suppress numerical warnings. See the [upstream NumPy issue](https://github.com/numpy/numpy/issues/28687).

### Docker

With Docker Desktop running Linux containers:

```bash
docker compose up --build
```

Open [localhost:8502](http://localhost:8502). Stop with Ctrl+C; restart with `docker compose up`. Rebuild after code or dependency changes. The service binds to localhost, runs as a non-root user, includes a health check, and mounts `data/` read-only.

## Try the complete workflow

1. Leave **Explore demo** selected.
2. Choose an evaluation split and decision policy in the sidebar.
3. Select **Train all models**. The dashboard opens Model lab when training finishes.
4. Compare the models and inspect the validation threshold curves and split audit.
5. Open **Transaction review** and enable **Try 20 sample transactions**, or upload a feature CSV.
6. Filter the queue and export all predictions or only the matching rows.

Changes in the experiment form apply when you train again. The existing results retain their original settings, shown in Model lab. Changing datasets clears the previous experiment. Parsed datasets and scored batches are cached within the current browser session, not in a shared global data cache.

## Bring your own dataset

Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Select **Upload dataset**, or place the file in `data/creditcard.csv` and choose **Local dataset**.

| Field | Training | Prediction | Requirements |
| --- | --- | --- | --- |
| `Time` | Required | Required | Numeric, nonnegative; every row needs a value for chronological splitting |
| `V1` … `V28` | Required | Required | Numeric anonymized features |
| `Amount` | Required | Required | Numeric, nonnegative, in dataset units |
| `Class` | Required | Ignored | 0 = genuine, 1 = fraud; no missing labels |
| Extra columns | Ignored | Preserved | Useful for transaction IDs and other metadata |

CSV files must be UTF-8 (an Excel UTF-8 BOM is accepted) with one header row. Header whitespace is trimmed. Duplicate column names, malformed records, invalid numbers, infinities, negative amounts/times, and missing required columns produce specific errors.

Training requires at least **20 distinct rows of each class**. Exact duplicates are removed before splitting; identical feature rows with conflicting labels are rejected. Missing feature values are filled using training medians. Training fails if any feature has no observed training values. A prediction batch can have an entirely missing feature column; its values are still filled using the trained imputer.

Batch scoring preserves row order, duplicates and extra columns. `Class` is ignored even if present. The output adds `Fraud_score`, `Predicted_class`, `Decision`, `Decision_threshold`, and `Scoring_model`. Input columns with those names are rejected to prevent accidental overwrites. Text metadata that could become spreadsheet formulas is escaped on CSV export; numeric values remain numeric.

Extra metadata columns are read as text so identifiers such as `000123` and literal values such as `NA` are preserved. In Transaction review, search across metadata columns and choose which fields to display alongside each prediction. Open **Inspect a transaction** to see the exact score decomposition and download all 30 feature contributions. Contributions are measured relative to zero standardized inputs; they explain this model’s arithmetic, not causal evidence of fraud.

V1–V28 are required: a card number and amount alone are not sufficient inputs.

## Evaluation methodology

1. **Separate the data.** Default: fixed-seed stratified 60/20/20 train/validation/test split. Chronological mode sorts by `Time`, trains on the earliest period, and holds out later periods. Equal timestamps never cross boundaries, so proportions can shift. Every partition must contain both classes.
2. **Fit on training only.** Median imputation, standardization, optional PCA, and balanced-class Logistic Regression / Linear SVM all fit exclusively on training rows. No resampling occurs before splitting.
3. **Choose the decision threshold on validation.**
   - `f1`: maximize validation F1 (default).
   - `f2`: maximize validation F2, giving recall more weight.
   - `recall`: maximize precision among validation thresholds meeting the requested recall target.
   - Ties favor the higher threshold and smaller review queue. A validation target is not a guarantee on test or future data.
4. **Select the model using validation average precision.** The recommendation never uses test metrics.
5. **Report held-out test results.** Precision, recall, F1, ROC-AUC, average precision, review rate, false alarms, and missed fraud are measured using the frozen validation threshold.
6. **Keep the original fitted pipeline.** No refit changes the preprocessing, model or decision threshold after evaluation.

Decision scores are uncalibrated margins, **not fraud probabilities**, and cannot be compared across different models. Feature weights represent associations in standardized input coordinates; PCA coefficients are mapped back to those coordinates. They do not establish causality. Kaggle V1–V28 are already PCA components, so additional PCA is an experiment, not a guaranteed improvement.

Repeatedly tuning configurations based on the test table biases the final result. Small fraud counts make results unstable; the app records a warning when any partition has fewer than 10 examples of either class. Random holdout does not estimate future-period performance as directly as chronological holdout. Chronological evaluation still needs sufficient fraud examples and does not replace deployment monitoring.

## Command-line usage

Train a synthetic experiment:

```bash
python train.py --output outputs/demo-v2
```

Train with real data and a chronological recall target:

```bash
python train.py --csv data/creditcard.csv --output outputs/kaggle --split chronological --policy recall --target-recall 0.90 --retained 0.95
```

Each run writes:

- `metrics.csv`: all four models and their measured test results.
- `run.json`: source, UTC timestamp, normalized-dataset SHA-256, split audit, seed, policy, package versions, warnings, timing and metrics.
- `report.md`: readable experiment summary, selected-model outcomes, comparison table, provenance and limitations.
- `models.joblib`: fitted pipelines, thresholds, validation operating points and held-out labels/scores.

Score a batch using the validation-selected model:

```bash
python predict.py --model outputs/demo-v2/models.joblib --csv transactions.csv --output outputs/predictions.csv
```

Choose another model with `--name "PCA + Linear SVM"`. Both commands protect existing output files; pass `--overwrite` explicitly to replace them. Prediction output cannot replace the input CSV or model. Models from the previous artifact format must be retrained.

Only load model files you created or trust: **joblib loading can execute code**. Use the same dependency environment for training and prediction. The CLI rejects incompatible scikit-learn versions; this check is not a security mechanism. The dashboard does not accept uploaded model artifacts.

## Verification

```bash
python -m unittest discover -v
python -m pip check
```

Tests include malformed CSVs, conflicting labels, complete and missing-feature batches, disjoint and chronological splits, validation-only selection, coefficient reconstruction, model persistence, CLI roundtrips, and Streamlit session interactions. The GitHub Actions workflow runs the suite on Python 3.11 and 3.12 when the project is placed in a GitHub repository.

See [REPORT.md](REPORT.md) for the academic write-up and screenshot checklist. The checked-in `outputs/demo/` results are generated from the synthetic demo; `run.json` records the environment that produced them.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Dashboard screens and session management |
| `ui.py` | Shared styling and chart presentation |
| `ml.py` | Validation, splits, training, thresholds, scoring, provenance |
| `data_io.py` | Strict CSV loading and spreadsheet-safe exports |
| `reporting.py` | Downloadable reports generated from measured experiment results |
| `train.py` / `predict.py` | Command-line training and batch prediction |
| `test_project.py` / `test_app.py` | Model, data, CLI and dashboard regression tests |
| `Dockerfile` / `compose.yaml` | Local container setup |

## Scope

This is supervised binary classification, despite the original “anomaly detection” assignment title. Flags support human review and never automatically block payments. Real use needs secure hosting, access controls, independent future-period validation and drift monitoring. Anonymized inputs cannot establish demographic fairness.

The app processes CSVs on the machine running Streamlit. It does not send transaction data to an external prediction service. Browser uploads go to that app server, so host placement matters. Session data and model artifacts should be handled as sensitive. Streamlit usage telemetry is disabled, and the interface uses system fonts without an external font request.

## References

- [Scikit-learn pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Decision threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [Model evaluation](https://scikit-learn.org/stable/model_selection.html)
- [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing)
