"""Dashboard workflow tests using Streamlit's actual session and widget runtime."""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).with_name('app.py')


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assert_clean()

    def assert_clean(self):
        self.assertFalse(self.app.exception, [exc.message for exc in self.app.exception])
        self.assertFalse(self.app.error, [error.value for error in self.app.error])

    def widget(self, kind, label):
        return next(widget for widget in getattr(self.app, kind) if widget.label == label)

    def train(self):
        self.widget('button', 'Train all models').click().run()
        self.assert_clean()
        self.assertEqual(self.widget('radio', 'Workspace').value, 'Model lab')

    def test_overview_and_all_untrained_pages(self):
        self.assertEqual(self.app.metric[0].value, '6,000')
        for page in ['Model lab', 'Transaction review', 'Project guide', 'Overview']:
            self.widget('radio', 'Workspace').set_value(page).run()
            self.assert_clean()

    def test_train_explore_models_and_review_sample(self):
        self.train()
        for name in ['Logistic Regression', 'PCA + Linear SVM']:
            self.widget('selectbox', 'Explore model').set_value(name).run()
            self.assert_clean()
        self.widget('radio', 'Workspace').set_value('Transaction review').run()
        self.widget('checkbox', 'Try 20 sample transactions').check().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state['scored_batch']['frame'].shape[0], 20)
        for choice in ['Flagged', 'Predicted genuine', 'All transactions']:
            self.widget('selectbox', 'Queue filter').set_value(choice).run()
            self.assert_clean()
        self.widget('number_input', 'Minimum amount').set_value(1_000_000.).run()
        self.assert_clean()
        self.assertTrue(any('No transactions match' in info.value for info in self.app.info))

    def test_chronological_recall_training(self):
        self.widget('selectbox', 'Evaluation split').set_value('chronological')
        self.widget('selectbox', 'Decision policy').set_value('recall')
        self.widget('slider', 'Recall target').set_value(.9)
        self.train()
        result = self.app.session_state['result']
        self.assertEqual(result['config']['split'], 'chronological')
        self.assertEqual(result['config']['policy'], 'recall')
        self.assertGreaterEqual(result['models'][result['best']]['validation_metrics']['Recall'], .9)

    def test_boosting_review_and_optional_four_model_mode(self):
        self.train()
        self.assertEqual(len(self.app.session_state['result']['models']), 5)
        self.widget('selectbox', 'Explore model').set_value('Gradient Boosting').run()
        self.assert_clean()
        self.widget('radio', 'Workspace').set_value('Transaction review').run()
        self.widget('selectbox', 'Scoring model').set_value('Gradient Boosting').run()
        self.widget('checkbox', 'Try 20 sample transactions').check().run()
        self.assert_clean()
        self.assertTrue(any('nonlinear trees' in info.value for info in self.app.info))
        self.widget('checkbox', 'Include Gradient Boosting').uncheck()
        self.train()
        self.assertEqual(len(self.app.session_state['result']['models']), 4)

    def test_metadata_search_is_literal_and_preserves_user_row_column(self):
        self.train()
        self.widget('radio', 'Workspace').set_value('Transaction review').run()
        self.widget('checkbox', 'Try 20 sample transactions').check().run()
        cached = self.app.session_state['scored_batch']
        cached['frame']['Input row'] = ['0001[abc]'] + ['other'] * 19
        self.app.session_state['scored_batch'] = cached
        self.app.run()
        self.widget('text_input', 'Search transaction metadata').set_value('[ABC]').run()
        self.assert_clean()
        queue = self.app.dataframe[0].value
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue['Input row'].iloc[0], '0001[abc]')
        self.assertEqual(queue['Input row (reference)'].iloc[0], 1)

    def test_changing_dataset_clears_stale_results(self):
        self.train()
        self.widget('selectbox', 'Dataset source').set_value('Upload dataset').run()
        self.assert_clean()
        self.assertNotIn('result', self.app.session_state)
        self.widget('selectbox', 'Dataset source').set_value('Explore demo').run()
        self.assert_clean()
        self.assertNotIn('result', self.app.session_state)


if __name__ == '__main__':
    unittest.main()
