"""Reproducible fraud experiments with isolated training, validation and test data."""
import hashlib
import platform
import warnings
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

FEATURES = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
SEED = 61
ARTIFACT_VERSION = 2
POLICIES = {'f1': 'Balanced F1', 'f2': 'Recall-focused F2', 'recall': 'Minimum recall target'}
SPLITS = {'stratified': 'Stratified random', 'chronological': 'Chronological'}
PREDICTION_COLUMNS = ['Fraud_score', 'Predicted_class', 'Decision', 'Decision_threshold', 'Scoring_model']


def demo_data(n=6000):
    """Deterministic synthetic data, never a substitute for measured real-data results."""
    x, y = make_classification(
        n_samples=n, n_features=28, n_informative=10, n_redundant=6,
        weights=[.97, .03], class_sep=1.5, random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    frame = pd.DataFrame(x, columns=FEATURES[1:-1])
    frame.insert(0, 'Time', np.arange(n) * 20)
    frame['Amount'] = rng.lognormal(3, 1.2, n).round(2)
    frame['Class'] = y
    return frame


def validate(df, labelled=True):
    """Normalize feature order; deduplicate only labelled training datasets."""
    if df.empty:
        raise ValueError('The CSV contains no rows.')
    if df.columns.duplicated().any():
        raise ValueError('Column names must be unique.')
    required = FEATURES + (['Class'] if labelled else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    out = df[required].copy()
    for column in required:
        try:
            out[column] = pd.to_numeric(out[column], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{column} must contain numbers or blank cells.') from exc
        if np.iscomplexobj(out[column]) or np.isinf(out[column].to_numpy(dtype=float)).any():
            raise ValueError(f'{column} contains infinite or complex values; use finite numbers.')
    for column in ('Time', 'Amount'):
        if (out[column] < 0).any():
            raise ValueError(f'{column} cannot contain negative values.')
    if labelled:
        empty = out[FEATURES].columns[out[FEATURES].isna().all()].tolist()
        if empty:
            raise ValueError('Entirely empty training features: ' + ', '.join(empty))
        if out.Class.isna().any() or set(out.Class.unique()) != {0, 1}:
            raise ValueError('Class must contain both 0 (genuine) and 1 (fraud), without missing labels.')
        out = out.drop_duplicates().reset_index(drop=True)
        if out.duplicated(subset=FEATURES, keep=False).any():
            raise ValueError('Identical feature rows have conflicting Class labels. Correct these labels before training.')
        if out.Class.value_counts().min() < 20:
            raise ValueError('Provide at least 20 distinct rows of each class for reliable splitting.')
        out['Class'] = out.Class.astype(int)
    # An all-missing column in a prediction batch is valid: use training medians.
    return out


def dataset_fingerprint(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def split_data(df, strategy='stratified'):
    """Return disjoint 60/20/20 partitions; never split a timestamp across periods."""
    if strategy not in SPLITS:
        raise ValueError('Unknown split strategy: ' + str(strategy))
    if strategy == 'stratified':
        dev, test = train_test_split(df, test_size=.2, stratify=df.Class, random_state=SEED)
        training, validation = train_test_split(dev, test_size=.25, stratify=dev.Class, random_state=SEED)
    else:
        if df.Time.isna().any():
            raise ValueError('Chronological splitting requires Time on every row.')
        ordered = df.sort_values('Time', kind='stable')
        times = ordered.Time.to_numpy()
        first = int(np.searchsorted(times, times[int(len(times) * .6)], side='left'))
        second = int(np.searchsorted(times, times[int(len(times) * .8)], side='left'))
        training, validation, test = ordered.iloc[:first], ordered.iloc[first:second], ordered.iloc[second:]
    partitions = (training, validation, test)
    for name, partition in zip(('Training', 'Validation', 'Test'), partitions):
        if partition.Class.nunique() != 2:
            raise ValueError(f'{name} split needs both classes. Use more time periods or choose a stratified split.')
    empty = training[FEATURES].columns[training[FEATURES].isna().all()].tolist()
    if empty:
        raise ValueError('No observed training values for: ' + ', '.join(empty))
    return partitions


def threshold_curve(labels, scores):
    """Validation operating points, including tied-score groups exactly once."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('The model produced non-finite scores. Check feature scales and numerical dependencies.')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]
    return pd.DataFrame({
        'Threshold': thresholds, 'Precision': precision, 'Recall': recall,
        'F1': 2 * precision * recall / np.maximum(precision + recall, 1e-12),
        'F2': 5 * precision * recall / np.maximum(4 * precision + recall, 1e-12),
        'Flag rate': (len(scores) - np.searchsorted(np.sort(scores), thresholds)) / len(scores),
    })


def select_threshold(curve, policy='f1', target_recall=.8):
    if policy not in POLICIES:
        raise ValueError('Unknown threshold policy: ' + str(policy))
    if not 0 < target_recall <= 1:
        raise ValueError('Target recall must be greater than 0 and at most 1.')
    if policy == 'recall':
        candidates = curve[curve.Recall >= target_recall]
        metric = 'Precision'
    else:
        candidates, metric = curve, policy.upper()
    # In a tie, prefer the higher threshold (a smaller review queue).
    return float(candidates.sort_values([metric, 'Threshold'], ascending=False).iloc[0].Threshold)


def evaluate(labels, scores, threshold):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('The model produced non-finite scores. Check feature scales and numerical dependencies.')
    predictions = (scores >= threshold).astype(int)
    confusion = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    return {
        'Precision': float(precision_score(labels, predictions, zero_division=0)),
        'Recall': float(recall_score(labels, predictions, zero_division=0)),
        'F1': float(f1_score(labels, predictions, zero_division=0)),
        'ROC-AUC': float(roc_auc_score(labels, scores)),
        'Average precision': float(average_precision_score(labels, scores)),
        'Flag rate': float(predictions.mean()),
        'False alarms': int(fp), 'Missed fraud': int(fn),
    }, confusion


def train(df, retained=.95, split='stratified', policy='f1', target_recall=.8, progress=None):
    """Select models and thresholds on validation only; preserve test data for reporting."""
    if isinstance(retained, bool) or not 0 < retained < 1:
        raise ValueError('PCA retained variance must be between 0 and 1, exclusive.')
    if policy not in POLICIES:
        raise ValueError('Unknown threshold policy: ' + str(policy))
    if not 0 < target_recall <= 1:
        raise ValueError('Target recall must be greater than 0 and at most 1.')
    started = perf_counter()
    raw_rows = len(df)
    df = validate(df)
    training, validation, test = split_data(df, split)
    x_train, y_train = training[FEATURES], training.Class
    x_val, y_val = validation[FEATURES], validation.Class
    x_test, y_test = test[FEATURES], test.Class
    models, rows, notes = {}, [], []
    for use_pca in (False, True):
        for kind in ('Logistic Regression', 'Linear SVM'):
            name = ('PCA + ' if use_pca else '') + kind
            if progress:
                progress(len(models), 4, name)
            classifier = (
                LogisticRegression(class_weight='balanced', max_iter=2500, random_state=SEED)
                if kind == 'Logistic Regression' else
                LinearSVC(class_weight='balanced', dual=False, max_iter=10000, random_state=SEED)
            )
            steps = [('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]
            if use_pca:
                steps.append(('pca', PCA(n_components=retained, svd_solver='full')))
            model = Pipeline(steps + [('classifier', classifier)])
            fit_started = perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x_train, y_train)
            notes.extend(f'{name}: {warning.message}' for warning in caught)
            fit_seconds = perf_counter() - fit_started
            val_scores = model.decision_function(x_val)
            curve = threshold_curve(y_val, val_scores)
            threshold = select_threshold(curve, policy, target_recall)
            scores = model.decision_function(x_test)
            metrics, confusion = evaluate(y_test, scores, threshold)
            validation_metrics, _ = evaluate(y_val, val_scores, threshold)
            rows.append({
                'Model': name, 'Validation AP': validation_metrics['Average precision'], **metrics,
                'Threshold': threshold,
                'Components': int(model.named_steps['pca'].n_components_) if use_pca else len(FEATURES),
                'Fit seconds': fit_seconds,
            })
            models[name] = {
                'pipeline': model, 'threshold': threshold, 'scores': scores, 'confusion': confusion,
                'validation_curve': curve, 'validation_metrics': validation_metrics,
            }
    table = pd.DataFrame(rows)
    summary = []
    for name, partition in zip(('Train', 'Validation', 'Test'), (training, validation, test)):
        summary.append({
            'Split': name, 'Rows': len(partition), 'Fraud': int(partition.Class.sum()),
            'Prevalence': float(partition.Class.mean()),
            'Time start': float(partition.Time.min()) if partition.Time.notna().any() else None,
            'Time end': float(partition.Time.max()) if partition.Time.notna().any() else None,
        })
        if partition.Class.value_counts().min() < 10:
            notes.append(f'{name} has fewer than 10 examples of one class; metrics may be unstable.')
    return {
        'artifact_version': ARTIFACT_VERSION, 'models': models, 'metrics': table,
        'best': str(table.loc[table['Validation AP'].idxmax(), 'Model']), 'y_test': y_test,
        'notes': list(dict.fromkeys(notes)), 'sizes': [len(part) for part in (training, validation, test)],
        'rows': len(df), 'duplicates_removed': raw_rows - len(df), 'split_summary': summary,
        'config': {'retained': retained, 'split': split, 'policy': policy, 'target_recall': target_recall, 'seed': SEED},
        'fingerprint': dataset_fingerprint(df), 'created_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': perf_counter() - started,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scikit-learn': sklearn.__version__},
    }


def feature_weights(result, name):
    """Linear coefficients in standardized original-feature coordinates, even after PCA."""
    model = result['models'][name]['pipeline']
    coefficients = model.named_steps['classifier'].coef_[0]
    if 'pca' in model.named_steps:
        coefficients = model.named_steps['pca'].components_.T @ coefficients
    return pd.DataFrame({'Feature': FEATURES, 'Weight': coefficients}).sort_values(
        'Weight', key=lambda values: values.abs(), ascending=False,
    )


def predict(result, name, df):
    """Keep every original row and metadata field; apply the frozen validation threshold."""
    if name not in result['models']:
        raise ValueError('Unknown model: ' + str(name))
    collisions = sorted(set(df.columns) & set(PREDICTION_COLUMNS))
    if collisions:
        raise ValueError('Remove existing prediction columns before scoring: ' + ', '.join(collisions))
    x = validate(df, labelled=False)
    entry = result['models'][name]
    scores = entry['pipeline'].decision_function(x)
    if not np.isfinite(scores).all():
        raise ValueError('Prediction produced non-finite scores. Check feature scales and numerical dependencies.')
    output = df.copy()
    output['Fraud_score'] = scores
    output['Predicted_class'] = (scores >= entry['threshold']).astype(int)
    output['Decision'] = np.where(output.Predicted_class == 1, 'Review for fraud', 'Predicted genuine')
    output['Decision_threshold'] = entry['threshold']
    output['Scoring_model'] = name
    return output


def run_metadata(result, source='unspecified'):
    """JSON-compatible provenance and metrics without transaction records."""
    return {
        'artifact_version': result['artifact_version'], 'source': source,
        'created_at': result['created_at'], 'dataset_sha256': result['fingerprint'],
        'rows': result['rows'], 'duplicates_removed': result['duplicates_removed'],
        'recommended': result['best'], 'config': result['config'],
        'split_sizes': result['sizes'], 'splits': result['split_summary'],
        'elapsed_seconds': result['elapsed_seconds'], 'versions': result['versions'],
        'warnings': result['notes'], 'metrics': result['metrics'].to_dict(orient='records'),
    }
