"""Regression coverage for data boundaries, leakage, decisions, and CLI artifacts."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score

from data_io import export_csv, read_csv
from ml import (
    FEATURES, dataset_fingerprint, demo_data, feature_weights, predict, run_metadata,
    select_threshold, split_data, threshold_curve, train, validate,
)

ROOT = Path(__file__).parent


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = demo_data(2000)

    def test_missing_columns_are_actionable(self):
        with self.assertRaisesRegex(ValueError, 'Amount'):
            validate(self.df.drop(columns='Amount'))

    def test_bad_numbers_name_the_column(self):
        broken = self.df.copy()
        broken['Amount'] = broken.Amount.astype(object)
        broken.loc[0, 'Amount'] = 'not a number'
        with self.assertRaisesRegex(ValueError, 'Amount must contain numbers'):
            validate(broken)

    def test_infinite_and_negative_features(self):
        for column, value in [('Amount', np.inf), ('V1', -np.inf), ('Amount', -1), ('Time', -1)]:
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate(self.df.assign(**{column: value}))

    def test_binary_labels_required(self):
        for labels in [0, 2, np.nan]:
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'Class'):
                validate(self.df.assign(Class=labels))

    def test_too_few_distinct_minority_examples(self):
        small = pd.concat([self.df[self.df.Class == 0], self.df[self.df.Class == 1].head(19)])
        with self.assertRaisesRegex(ValueError, 'at least 20'):
            validate(small)

    def test_deduplication_and_conflicting_labels(self):
        repeated = pd.concat([self.df, self.df.iloc[:20]])
        pd.testing.assert_frame_equal(validate(repeated), validate(self.df))
        contradictory = pd.concat([self.df, self.df.iloc[:1].assign(Class=1 - self.df.Class.iloc[0])])
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            validate(contradictory)

    def test_training_rejects_empty_features(self):
        with self.assertRaisesRegex(ValueError, 'empty training features'):
            validate(self.df.assign(V3=np.nan))

    def test_csv_rejects_duplicate_headers_before_pandas_renames_them(self):
        for text in ['Amount,Amount\n1,2\n', 'Amount, Amount \n1,2\n']:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'unique'):
                read_csv(text.encode())

    def test_csv_empty_malformed_and_bad_encoding(self):
        for content in [b'', b'  ', b'a,b\n1,2,3\n', b'a,b\n1\n', b'a,\n1,2\n', b'a,b\n"unclosed,2']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                read_csv(content)
        with self.assertRaises(UnicodeError):
            read_csv(b'\xff\xff')

    def test_csv_bom_whitespace_and_reusable_stream(self):
        stream = io.BytesIO('\ufeff Amount , V1\n2,3\n'.encode())
        first = read_csv(stream)
        self.assertEqual(list(first), ['Amount', 'V1'])
        pd.testing.assert_frame_equal(first, read_csv(stream))

    def test_empty_dataframe_and_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, 'no rows'):
            validate(self.df.head(0))
        broken = pd.concat([self.df, self.df[['V1']]], axis=1)
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate(broken)

    def test_spreadsheet_formula_text_is_escaped(self):
        frame = pd.DataFrame({'ID': ['=1+1', ' @SUM(A1)', 'safe', '-formula'], 'Score': [-1., 2., 3., 4.]})
        exported = read_csv(export_csv(frame).encode())
        self.assertEqual(exported.ID.tolist(), ["'=1+1", "' @SUM(A1)", 'safe', "'-formula"])
        np.testing.assert_array_equal(exported.Score, frame.Score)
        self.assertEqual(frame.ID.iloc[0], '=1+1')


class SplitAndThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = demo_data(2000)

    def test_stratified_splits_are_disjoint_and_exhaustive(self):
        parts = split_data(self.df)
        ids = [set(part.index) for part in parts]
        self.assertEqual([len(part) for part in parts], [1200, 400, 400])
        self.assertEqual(set.union(*ids), set(self.df.index))
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for part in parts:
            self.assertEqual(part.Class.nunique(), 2)

    def test_chronological_splits_keep_timestamps_separate(self):
        data = self.df.sample(frac=1, random_state=12).copy()
        data['Time'] = (data.index // 7).astype(float)
        a, b, c = split_data(data, 'chronological')
        self.assertLess(a.Time.max(), b.Time.min())
        self.assertLess(b.Time.max(), c.Time.min())
        self.assertEqual(len(a) + len(b) + len(c), len(data))

    def test_chronological_split_rejects_missing_times_and_classless_periods(self):
        for data in [self.df.assign(Time=1), self.df.assign(Time=np.nan), self.df.sort_values('Class').assign(Time=np.arange(len(self.df)))]:
            with self.subTest(), self.assertRaises(ValueError):
                split_data(data, 'chronological')

    def test_empty_feature_in_training_partition_is_rejected(self):
        data = self.df.copy()
        training, _, _ = split_data(data)
        data.loc[training.index, 'V1'] = np.nan
        with self.assertRaisesRegex(ValueError, 'No observed training values for: V1'):
            split_data(data)

    def test_threshold_policies_match_exhaustive_decisions_with_ties(self):
        y = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        scores = np.array([-2, -1, -.5, 0, 0, .2, .8, 1.])
        curve = threshold_curve(y, scores)
        for beta in [1, 2]:
            selected = select_threshold(curve, f'f{beta}')
            expected = max((fbeta_score(y, scores >= t, beta=beta), t) for t in np.unique(scores))
            self.assertEqual(selected, expected[1])
        threshold = select_threshold(curve, 'recall', .75)
        self.assertGreaterEqual(recall_score(y, scores >= threshold), .75)
        eligible = [t for t in np.unique(scores) if recall_score(y, scores >= t) >= .75]
        self.assertEqual(threshold, max((precision_score(y, scores >= t), t) for t in eligible)[1])
        for row in curve.itertuples(index=False, name=None):
            self.assertEqual(row[-1], np.mean(scores >= row[0]))

    def test_nonfinite_scores_and_invalid_settings_fail_early(self):
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            threshold_curve([0, 1], [np.nan, 1])
        for settings in [{'retained': 1}, {'retained': 0}, {'policy': 'bad'}, {'target_recall': 0}, {'split': 'bad'}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                train(self.df, **settings)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = demo_data(2000)
        cls.result = train(cls.df)

    def test_four_models_train_preprocessing_on_training_only(self):
        self.assertEqual(len(self.result['models']), 4)
        training, _, _ = split_data(self.df)
        for model in self.result['models'].values():
            self.assertEqual(model['confusion'].sum(), self.result['sizes'][2])
            scale = model['pipeline'].named_steps['scale']
            self.assertEqual(scale.n_samples_seen_, self.result['sizes'][0])
            np.testing.assert_allclose(scale.mean_, training[FEATURES].mean())
            imputer = model['pipeline'].named_steps['imputer']
            np.testing.assert_allclose(imputer.statistics_, training[FEATURES].median())

    def test_model_and_threshold_selection_use_validation(self):
        table = self.result['metrics']
        self.assertEqual(self.result['best'], table.loc[table['Validation AP'].idxmax(), 'Model'])
        for model in self.result['models'].values():
            self.assertEqual(model['threshold'], select_threshold(model['validation_curve']))
        # Changing only test labels must not alter model or threshold selection.
        training, validation, test = split_data(self.df)
        altered_test = test.assign(Class=1 - test.Class)
        with patch('ml.split_data', return_value=(training, validation, altered_test)):
            altered = train(self.df)
        self.assertEqual(altered['best'], self.result['best'])
        for name, model in self.result['models'].items():
            self.assertEqual(model['threshold'], altered['models'][name]['threshold'])

    def test_prediction_preserves_order_duplicates_and_metadata(self):
        x = pd.concat([self.df.head(12), self.df.head(2)]).assign(TransactionID=range(14))
        a = predict(self.result, self.result['best'], x)
        b = predict(self.result, self.result['best'], x[list(reversed(FEATURES))])
        np.testing.assert_allclose(a.Fraud_score, b.Fraud_score)
        self.assertEqual(a.TransactionID.tolist(), list(range(14)))
        self.assertEqual(a.index.tolist(), x.index.tolist())
        np.testing.assert_array_equal(a.Predicted_class, a.Fraud_score >= a.Decision_threshold)
        self.assertEqual(len(a), len(x))

    def test_prediction_ignores_class_and_imputes_wholly_missing_batch_columns(self):
        x = self.df.head(10).assign(Amount=np.nan, Class='ignored')
        output = predict(self.result, self.result['best'], x)
        self.assertTrue(np.isfinite(output.Fraud_score).all())
        self.assertEqual(output.Class.tolist(), ['ignored'] * 10)

    def test_prediction_does_not_overwrite_existing_decisions(self):
        with self.assertRaisesRegex(ValueError, 'existing prediction columns'):
            predict(self.result, self.result['best'], self.df.head().assign(Fraud_score=0))
        with self.assertRaisesRegex(ValueError, 'Unknown model'):
            predict(self.result, 'missing', self.df.head())

    def test_pca_feature_coefficients_reconstruct_model_margin(self):
        x = self.df[FEATURES].head(15)
        for name, entry in self.result['models'].items():
            pipeline = entry['pipeline']
            imputed = pipeline.named_steps['imputer'].transform(x)
            scaled = pipeline.named_steps['scale'].transform(imputed)
            weights = feature_weights(self.result, name).set_index('Feature').loc[FEATURES, 'Weight'].to_numpy()
            intercept = pipeline.named_steps['classifier'].intercept_[0]
            if 'pca' in pipeline.named_steps:
                intercept -= pipeline.named_steps['pca'].mean_ @ weights
            np.testing.assert_allclose(scaled @ weights + intercept, pipeline.decision_function(x), atol=1e-10)

    def test_manifest_serializes_and_fingerprint_tracks_data(self):
        manifest = json.loads(json.dumps(run_metadata(self.result), allow_nan=False))
        self.assertEqual(manifest['rows'], len(self.df))
        self.assertEqual(len(manifest['dataset_sha256']), 64)
        self.assertEqual(manifest['config']['seed'], 61)
        changed = self.df.copy()
        changed.loc[0, 'Amount'] += 1
        self.assertNotEqual(dataset_fingerprint(changed), self.result['fingerprint'])

    def test_artifact_roundtrip_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'models.joblib'
            joblib.dump(self.result, path)
            loaded = joblib.load(path)
            expected = predict(self.result, self.result['best'], self.df.head())
            actual = predict(loaded, loaded['best'], self.df.head())
            pd.testing.assert_frame_equal(expected, actual)

    def test_chronological_recall_experiment(self):
        result = train(self.df, split='chronological', policy='recall', target_recall=.9)
        for model in result['models'].values():
            self.assertGreaterEqual(model['validation_metrics']['Recall'], .9)
        self.assertLess(result['split_summary'][0]['Time end'], result['split_summary'][1]['Time start'])


class CommandLineTests(unittest.TestCase):
    def test_train_and_predict_roundtrip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / 'training.csv'
            batch = root / 'batch.csv'
            out = root / 'run'
            predictions = root / 'predictions.csv'
            data = demo_data(2000)
            data.to_csv(csv, index=False)
            data[FEATURES].head(13).to_csv(batch, index=False)
            args = [sys.executable, str(ROOT / 'train.py'), '--csv', str(csv), '--output', str(out)]
            trained = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(trained.returncode, 0, trained.stderr)
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 2)
            manifest = json.loads((out / 'run.json').read_text())
            self.assertEqual(manifest['rows'], 2000)
            scored = subprocess.run([sys.executable, str(ROOT / 'predict.py'), '--model', str(out / 'models.joblib'), '--csv', str(batch), '--output', str(predictions)], capture_output=True, text=True)
            self.assertEqual(scored.returncode, 0, scored.stderr)
            output = read_csv(predictions)
            self.assertEqual(len(output), 13)
            self.assertEqual(output.Scoring_model.unique().tolist(), [manifest['recommended']])


if __name__ == '__main__':
    unittest.main()
