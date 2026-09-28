"""FraudLens: a local workspace for measured, reviewable fraud experiments."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve

from data_io import export_csv, read_csv
from ml import (
    FEATURES, POLICIES, SPLITS, PREDICTION_COLUMNS, demo_data, explain_prediction, feature_weights, predict,
    run_metadata, train, validate,
)
from ui import COLORS, callout, configure_page, empty_state, section, show_figure
from reporting import experiment_report

ROOT = Path(__file__).parent
PAGES = ['Overview', 'Model lab', 'Transaction review', 'Project guide']


def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><span>◈</span> FraudLens</div><div class="brand-sub">TRANSACTION INTELLIGENCE</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('**01 / Data workspace**')
        source = st.selectbox('Dataset source', ['Explore demo', 'Upload dataset', 'Local dataset'])
        upload = st.file_uploader('Training CSV', type='csv', help='Time, V1–V28, Amount and Class.') if source == 'Upload dataset' else None
        st.caption('Synthetic data · 6,000 rows' if source == 'Explore demo' else 'Your data stays on the app server.')
        st.divider()
        st.markdown('**02 / Experiment setup**')
        with st.form('experiment'):
            split = st.selectbox('Evaluation split', list(SPLITS), format_func=SPLITS.get,
                                 help='Chronological trains on earlier transactions and tests on later ones.')
            policy = st.selectbox('Decision policy', list(POLICIES), format_func=POLICIES.get)
            target = st.slider('Recall target', .50, 1.0, .80, .05,
                               help='Used by Minimum recall target. This is a validation target, not a guarantee on future data.')
            retained = st.slider('PCA variance retained', .80, .99, .95, .01)
            run = st.form_submit_button('Train all models', type='primary', width='stretch')
        st.caption('Four models · 60 / 20 / 20 split\n\nThresholds chosen on validation data.')
        st.divider()
        st.markdown('[Get the Kaggle dataset ↗](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)')
        st.caption('Experiment 10 · TE IT\n\nOmkar M Prajapati · Roll 61')
    config = {'retained': retained, 'split': split, 'policy': policy, 'target_recall': target}
    return source, upload, config, run


def load_dataset(source, upload):
    """Cache only in this browser session; avoid reparsing the full CSV on every click."""
    if source == 'Explore demo':
        token, origin = 'demo-v2', 'SYNTHETIC DEMO'
        loader = demo_data
    elif source == 'Upload dataset':
        if upload is None:
            st.session_state.pop('result', None)
            empty_state('A fresh workspace for your data', 'Upload a labelled CSV in the sidebar, or choose Explore demo to try the complete workflow.')
            st.stop()
        payload = upload.getvalue()
        token, origin = hashlib.sha256(payload).hexdigest(), upload.name
        loader = lambda: read_csv(payload)
    else:
        path = ROOT / 'data' / 'creditcard.csv'
        if not path.exists():
            st.session_state.pop('result', None)
            empty_state('Connect your local dataset', 'Place creditcard.csv in the project’s data folder, then refresh this page.')
            st.stop()
        stat = path.stat()
        token, origin = f'{stat.st_mtime_ns}-{stat.st_size}', 'data/creditcard.csv'
        loader = lambda: read_csv(path)
    key = source + ':' + token
    if st.session_state.get('dataset_key') != key:
        # Invalidate first: failed or changed input can never retain an old experiment.
        for item in ('result', 'scored_batch', 'dataset', 'dataset_key'):
            st.session_state.pop(item, None)
        raw = loader()
        clean = validate(raw)
        st.session_state.dataset = {
            'frame': clean, 'raw_rows': len(raw), 'duplicates': len(raw) - len(clean),
            'missing': int(clean[FEATURES].isna().sum().sum()), 'origin': origin,
        }
        st.session_state.dataset_key = key
    return st.session_state.dataset


def overview(dataset):
    df = dataset['frame']
    left, right = st.columns([1.35, 1])
    with left, st.container(border=True):
        section('Class balance', 'A small signal in a large dataset', 'Both classes are shown as counts. The imbalance is why accuracy alone is misleading.')
        fig, ax = plt.subplots(figsize=(7, 2.8))
        counts = [int((df.Class == 0).sum()), int(df.Class.sum())]
        ax.barh(['Genuine', 'Fraud'], counts, color=[COLORS[0], COLORS[2]], height=.45)
        for i, count in enumerate(counts):
            ax.text(count + len(df) * .018, i, f'{count:,}  ·  {count / len(df):.1%}', va='center', fontsize=10)
        ax.set_xlim(0, max(counts) * 1.32)
        ax.set_xlabel('Transactions')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.6)
        ax.set_axisbelow(True)
        show_figure(fig)
    with right, st.container(border=True):
        section('Data readiness', 'Ready to investigate')
        st.markdown(f'**{len(FEATURES)} numeric features** · binary fraud labels')
        st.markdown(f'**{dataset["duplicates"]:,}** duplicate rows removed')
        st.markdown(f'**{dataset["missing"]:,}** missing feature values · imputed from training medians')
        st.markdown(f'**{df.Time.nunique():,}** distinct transaction timestamps')
        st.caption('Conflicting labels, invalid numbers and incomplete schemas are checked before training.')
        st.download_button('Download sample CSV', export_csv(df[FEATURES].head(20)), 'sample_transactions.csv', 'text/csv', width='stretch')
        st.caption('Sample rows demonstrate the format and may overlap training data.')
    with st.container(border=True):
        section('Transaction patterns', 'How transaction amounts differ', 'Amounts use the units supplied by your dataset; the horizontal axis is logarithmic.')
        a, b = st.columns([1.7, 1])
        with a:
            fig, ax = plt.subplots(figsize=(8, 2.7))
            amounts = np.log1p(df.Amount.dropna())
            bins = np.linspace(float(amounts.min()), float(amounts.max()) + .01, 36)
            for value, label, color in ((0, 'Genuine', COLORS[0]), (1, 'Fraud', COLORS[2])):
                ax.hist(np.log1p(df.loc[df.Class == value, 'Amount'].dropna()), bins=bins,
                        density=True, alpha=.5, color=color, label=label)
            ax.set(xlabel='log(1 + amount)', ylabel='Density')
            ax.legend(frameon=False)
            show_figure(fig)
        with b:
            profile = df.groupby('Class').Amount.agg(['median', 'mean', 'max'])
            profile.index = ['Genuine', 'Fraud']
            profile.columns = ['Median', 'Mean', 'Maximum']
            st.dataframe(profile.round(2), width='stretch')
            st.caption('The chart normalizes each class separately to compare their shapes.')
    with st.expander('Inspect transaction records'):
        choice = st.selectbox('Show records', ['All transactions', 'Fraud only', 'Genuine only'])
        view = df if choice == 'All transactions' else df[df.Class == (1 if choice == 'Fraud only' else 0)]
        st.dataframe(view.head(200), hide_index=True, width='stretch')
        st.caption(f'{len(view):,} matching rows · first 200 shown.')


def model_lab(result, source):
    section('Experiment results', 'Evidence before decisions', 'Recommendation uses validation average precision. Performance below is measured on the held-out test split.')
    if result is None:
        empty_state('Your first experiment starts here', 'Choose a split and decision policy in the sidebar, then select Train all models. All four pipelines use the same partitions.')
        return
    config = result['config']
    callout(result['best'], f'Validation-selected model · {SPLITS[config["split"]]} split · {POLICIES[config["policy"]]} · {result["elapsed_seconds"]:.1f}s total')
    if result['notes']:
        st.warning('This experiment has training or sample-size warnings. Read the split audit before interpreting the results.')
    names = list(result['models'])
    selected = st.selectbox('Explore model', names, index=names.index(result['best']))
    entry = result['models'][selected]
    metrics = result['metrics'].set_index('Model').loc[selected]
    columns = st.columns(4)
    for column, title, value, help_text in zip(columns,
        ['Average precision', 'Fraud recall', 'Flag precision', 'Review rate'],
        [metrics['Average precision'], metrics['Recall'], metrics['Precision'], metrics['Flag rate']],
        ['Precision–recall ranking summary.', 'Share of known fraud caught.', 'Share of alerts that are actual fraud.', 'Share of test transactions flagged.']):
        column.metric(title, f'{value:.1%}', help=help_text)
    st.write('')
    with st.container(border=True):
        st.markdown('#### Compare all four models')
        visible = ['Model', 'Validation AP', 'Average precision', 'Precision', 'Recall', 'F1', 'Flag rate', 'Components']
        st.dataframe(result['metrics'][visible].style.format({name: '{:.3f}' for name in visible[1:-1]}), hide_index=True, width='stretch')
        a, b, c = st.columns(3)
        a.download_button('Export comparison CSV', export_csv(result['metrics']), 'model_metrics.csv', 'text/csv', width='stretch')
        b.download_button('Export experiment JSON', json.dumps(run_metadata(result, source), indent=2, allow_nan=False), 'experiment.json', 'application/json', width='stretch')
        c.download_button('Download experiment report', experiment_report(result, source), 'fraudlens_report.md', 'text/markdown', width='stretch')
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown('#### Where the model gets it right')
        tn, fp, fn, tp = entry['confusion'].ravel()
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        ax.imshow(entry['confusion'], cmap='BuGn')
        for i in range(2):
            for j in range(2):
                value = entry['confusion'][i, j]
                ax.text(j, i, f'{value:,}', ha='center', va='center', fontsize=23,
                        color='white' if value > entry['confusion'].max() / 2 else '#174536')
        ax.set_xticks([0, 1], ['Genuine', 'Fraud'])
        ax.set_yticks([0, 1], ['Genuine', 'Fraud'])
        ax.set(xlabel='Predicted class', ylabel='Actual class')
        show_figure(fig)
        st.caption(f'{tp:,} fraud caught · {fn:,} fraud missed · {fp:,} false alarms · {tn:,} genuine cleared')
    with right, st.container(border=True):
        st.markdown('#### Precision and recall, together')
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for (name, model), color in zip(result['models'].items(), COLORS):
            precision, recall, _ = precision_recall_curve(result['y_test'], model['scores'])
            ax.step(recall, precision, where='post', label=name, color=color, lw=1.8)
        ax.axhline(result['y_test'].mean(), ls='--', color='#92a4ae', label='Class prevalence')
        ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(alpha=.5)
        show_figure(fig)
        st.caption('Higher curves indicate stronger fraud ranking on this test split.')
    with st.expander('Decision policy · validation operating points', expanded=True):
        curve = entry['validation_curve']
        # Plot a bounded sample but retain every operating point for exact selection.
        preview = curve.iloc[np.unique(np.linspace(0, len(curve) - 1, min(len(curve), 400)).astype(int))]
        fig, ax = plt.subplots(figsize=(10, 2.8))
        for metric, color in zip(['Precision', 'Recall', 'Flag rate'], COLORS):
            ax.plot(preview.Threshold, preview[metric], label=metric, color=color, lw=1.8)
        ax.axvline(entry['threshold'], color='#253b4a', ls='--', label='Selected threshold')
        ax.set(xlabel='Decision threshold (model margin)', ylabel='Validation fraction', ylim=(0, 1.03))
        ax.legend(frameon=False, ncol=4, fontsize=8)
        ax.grid(alpha=.5)
        show_figure(fig)
        validation = entry['validation_metrics']
        st.caption(f'Frozen threshold {entry["threshold"]:.4f} · validation precision {validation["Precision"]:.1%} · recall {validation["Recall"]:.1%} · review rate {validation["Flag rate"]:.1%}.')
        st.info('Choose your policy before evaluating test results. A validation recall target does not guarantee the same recall on unseen transactions.')
    left, right = st.columns(2)
    with left, st.expander('What influences the model?'):
        weights = feature_weights(result, selected).head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.barh(weights.Feature, weights.Weight, color=[COLORS[2] if value > 0 else COLORS[0] for value in weights.Weight])
        ax.axvline(0, color='#9cabb5', lw=.8)
        ax.set_xlabel('Coefficient per standardized feature unit')
        show_figure(fig)
        st.caption('Positive weights increase the fraud margin; negative weights decrease it. PCA weights are mapped back to the standardized input features. These are model associations, not causal explanations.')
    with right, st.expander('What did PCA retain?'):
        pca = result['models']['PCA + Logistic Regression']['pipeline'].named_steps['pca']
        st.write(f'**{pca.n_components_} components** retain **{pca.explained_variance_ratio_.sum():.1%}** of training variance.')
        st.line_chart(pd.DataFrame({'Cumulative variance': np.cumsum(pca.explained_variance_ratio_)}, index=np.arange(1, pca.n_components_ + 1)), color=COLORS[0])
        st.caption('V1–V28 are already anonymized PCA features. More retained variance does not guarantee stronger fraud detection.')
    with st.expander('Reproducibility and split audit'):
        st.dataframe(pd.DataFrame(result['split_summary']).style.format({'Prevalence': '{:.2%}'}), hide_index=True, width='stretch')
        st.caption('Timestamp boundaries remain separate in chronological mode; split proportions can shift to keep equal timestamps together.')
        st.json({'created_at': result['created_at'], 'config': config, 'dataset_sha256': result['fingerprint'], 'versions': result['versions']})
        for note in result['notes']:
            st.warning(note)


def transaction_review(result, df):
    section('Transaction review', 'Turn scores into a review queue', 'Rank a batch by the selected model’s score, inspect the flags, and export the results.')
    if result is None:
        empty_state('Train once. Review a whole batch.', 'Train all models in the sidebar to unlock transaction scoring. Then upload unseen transactions or try a sample.')
        return
    names = list(result['models'])
    left, right = st.columns([1, 1.5])
    with left:
        name = st.selectbox('Scoring model', names, index=names.index(result['best']))
        use_sample = st.checkbox('Try 20 sample transactions', value=False)
    with right:
        batch = st.file_uploader('Transaction CSV', type='csv', key='batch', help='Time, V1–V28 and Amount. Class is optional and ignored. Extra metadata is preserved.')
    st.caption('Scores are uncalibrated model margins, not fraud probabilities. They are only comparable within the selected model.')
    if batch is None and not use_sample:
        empty_state('Your review queue is ready', 'Upload a CSV with the 30 feature columns. Include your own transaction ID column to preserve it in the exported results.')
        return
    payload = batch.getvalue() if batch is not None else None
    token = hashlib.sha256(payload).hexdigest() if payload is not None else 'sample'
    key = (result['created_at'], name, token)
    try:
        if st.session_state.get('scored_batch', {}).get('key') != key:
            data = read_csv(payload) if payload is not None else df[FEATURES].head(20)
            with st.spinner('Scoring transactions…'):
                scored = predict(result, name, data)
            st.session_state.scored_batch = {'key': key, 'frame': scored}
        scored = st.session_state.scored_batch['frame']
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(f'Check your transaction CSV: {exc}')
        return
    if batch is None:
        st.warning('Sample demonstration: these rows can overlap training data. Use unseen transactions for independent evaluation.')
    a, b, c, d = st.columns(4)
    flagged = scored.Predicted_class == 1
    missing_cells = int(scored[FEATURES].isna().sum().sum())
    if missing_cells:
        st.info(f'{missing_cells:,} missing feature values were filled using training medians. Transaction explanations identify the imputed fields.')
    a.metric('Rows scored', f'{len(scored):,}')
    b.metric('Flagged for review', f'{flagged.sum():,}')
    c.metric('Flag rate', f'{flagged.mean():.1%}')
    d.metric('Flagged amount', f'{pd.to_numeric(scored.loc[flagged, "Amount"]).sum():,.2f}', help='Sum of known amounts in flagged rows, in dataset units. This is not an estimate of fraud loss.')
    st.write('')
    with st.container(border=True):
        a, b, c = st.columns([1.5, 1, 1])
        choice = a.selectbox('Queue filter', ['All transactions', 'Flagged', 'Predicted genuine'])
        minimum = b.number_input('Minimum amount', min_value=0.0, value=0.0, step=10.0)
        order = c.selectbox('Sort by', ['Highest score', 'Largest amount', 'Original order'])
        metadata = [column for column in scored if column not in FEATURES + PREDICTION_COLUMNS + ['Class']]
        search = st.text_input('Search transaction metadata', placeholder='Search IDs or other metadata; literal text, case-insensitive') if metadata else ''
        shown_metadata = st.multiselect('Metadata columns to display', metadata, default=metadata[:2]) if metadata else []
        positions = np.arange(len(scored))
        mask = np.ones(len(scored), dtype=bool)
        if search:
            matches = np.zeros(len(scored), dtype=bool)
            for column in metadata:
                matches |= scored[column].astype('string').str.contains(search, case=False, regex=False, na=False).to_numpy(dtype=bool)
            mask &= matches
        if choice != 'All transactions':
            mask &= scored.Predicted_class.to_numpy() == (1 if choice == 'Flagged' else 0)
        if minimum > 0:
            mask &= pd.to_numeric(scored.Amount).fillna(-1).to_numpy() >= minimum
        positions = positions[mask]
        if order != 'Original order':
            field = 'Fraud_score' if order == 'Highest score' else 'Amount'
            values = pd.to_numeric(scored.iloc[positions][field]).to_numpy()
            positions = positions[np.argsort(-np.nan_to_num(values, nan=-np.inf), kind='stable')]
        queue = scored.iloc[positions]
        if queue.empty:
            st.info('No transactions match these filters. Try another filter or a lower minimum amount.')
        else:
            pages = max(1, math.ceil(len(queue) / 100))
            page = int(st.number_input('Page', min_value=1, max_value=pages, value=1, step=1))
            start = (page - 1) * 100
            visible = queue.iloc[start:start + 100][['Decision', 'Fraud_score', 'Amount', 'Time'] + shown_metadata].copy()
            row_label = 'Input row'
            while row_label in visible.columns:
                row_label += ' (reference)'
            visible.insert(0, row_label, positions[start:start + 100] + 1)
            st.dataframe(visible, hide_index=True, width='stretch',
                         column_config={'Fraud_score': st.column_config.NumberColumn('Fraud score', format='%.4f')})
            st.caption(f'{len(queue):,} matches · page {page} of {pages} · input rows are numbered from 1 after the CSV header.')
        st.caption(f'Fixed decision threshold: {result["models"][name]["threshold"]:.4f}. Missing amounts remain included when the minimum is zero.')
        a, b = st.columns(2)
        a.download_button('Export all predictions', export_csv(scored), 'fraud_predictions.csv', 'text/csv', width='stretch')
        b.download_button('Export filtered queue', export_csv(queue), 'review_queue.csv', 'text/csv', width='stretch')
        st.caption('Exports preserve input metadata and include the model, threshold, decision and score for each row.')
    with st.expander('Inspect a transaction'):
        row = int(st.number_input('Input row number', min_value=1, max_value=len(scored), value=1, step=1))
        record = scored.iloc[row - 1]
        st.write(f'**{record.Decision}** · score {record.Fraud_score:.4f} · threshold {record.Decision_threshold:.4f}')
        explanation = explain_prediction(result, name, scored.iloc[[row - 1]][FEATURES])
        contributions = explanation['contributions']
        st.markdown('#### Why did this transaction receive this score?')
        top = contributions.head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.barh(top.Feature, top.Contribution, color=[COLORS[2] if value > 0 else COLORS[0] for value in top.Contribution])
        ax.axvline(0, color='#9cabb5', lw=.8)
        ax.set_xlabel('Contribution to this transaction’s fraud score')
        show_figure(fig)
        st.caption(f'Baseline {explanation["baseline"]:.4f} + all feature contributions {contributions.Contribution.sum():.4f} = score {explanation["score"]:.4f}. The chart shows the 10 largest absolute contributions.')
        st.caption('Orange pushes the score toward fraud; green pushes it toward genuine. Contributions are relative to zero standardized inputs. This explains the model calculation, not the cause of fraud.')
        st.dataframe(contributions, hide_index=True, width='stretch', column_config={
            'Contribution': st.column_config.NumberColumn(format='%.4f'),
            'Imputed': st.column_config.CheckboxColumn('Filled from training median'),
        })
        st.download_button('Export transaction explanation', export_csv(contributions), f'transaction_{row}_explanation.csv', 'text/csv')
        st.markdown('#### Original record and prediction')
        st.dataframe(record.rename('Value').astype(str).to_frame(), width='stretch')


def guide():
    section('Project guide', 'A transparent path from data to decision')
    for title, text in [
        ('01 / Prepare', 'Validate numeric fields and binary labels, reject conflicting labels, and remove exact duplicates before splitting.'),
        ('02 / Separate', 'Use a reproducible stratified split, or train on earlier transactions and evaluate later periods with a chronological split. Equal timestamps stay in one partition.'),
        ('03 / Learn', 'Fit median imputation, scaling, optional PCA, and class-weighted Logistic Regression or Linear SVM on training rows only.'),
        ('04 / Choose', 'Select a decision threshold using validation F1, recall-focused F2, or the highest precision satisfying a validation recall target. Select the model by validation average precision.'),
        ('05 / Evaluate', 'Report held-out precision, recall, F1, average precision, ROC-AUC, and review rate. Export the settings, split audit, package versions and dataset fingerprint with the results.'),
        ('06 / Review', 'Score unseen batches with a frozen model and threshold, preserve transaction IDs, filter the review queue and export the decisions for human review.'),
    ]:
        with st.container(border=True):
            st.markdown(f'**{title}**')
            st.write(text)
    st.info('Synthetic demo results demonstrate the workflow. They are not results on the Kaggle credit card dataset.')
    st.markdown('#### Read the metrics correctly')
    st.dataframe(pd.DataFrame([
        ('Precision', 'Of the flagged transactions, how many were fraud?'),
        ('Recall', 'Of the actual fraud transactions, how many were flagged?'),
        ('Average precision', 'How well does the model rank fraud across decision thresholds?'),
        ('F1 / F2', 'F1 balances precision and recall; F2 gives recall more weight.'),
        ('Review rate', 'What share of transactions enters the review queue?'),
    ], columns=['Metric', 'Interpretation']), hide_index=True, width='stretch')
    st.markdown('#### Scope and limitations')
    st.write('This is a supervised educational prototype. Anonymized features limit causal explanations and demographic fairness analysis. Chronological holdout is a useful check but does not replace future-period validation, drift monitoring or access controls. Repeated tuning against test results biases the reported performance. Flags support human review and do not automatically block payments.')
    st.markdown('[Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [Threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html) · [Streamlit documentation](https://docs.streamlit.io/)')
    st.caption('The local README.md includes setup, CLI commands and data requirements. REPORT.md provides the academic report template.')


def main():
    configure_page()
    source, upload, config, run = sidebar()
    st.markdown('''<div class="hero"><span class="eyebrow">Fraud intelligence / Research workspace</span>
    <h1>Find the signal.<br>Make an informed decision.</h1>
    <p>Understand your transactions, compare the evidence, and bring suspicious activity into focus.</p>
    <span class="tag">4 model pipelines</span><span class="tag">Validation-led decisions</span><span class="tag">Human review</span></div>''', unsafe_allow_html=True)
    try:
        dataset = load_dataset(source, upload)
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(f'Dataset needs attention: {exc}')
        st.stop()
    df = dataset['frame']
    if source == 'Explore demo':
        st.caption('● SYNTHETIC DEMO  /  Explore the full workflow with generated transactions. These are not Kaggle results.')
    else:
        st.caption(f'DATASET  /  {dataset["origin"]}')
    if run:
        progress = st.progress(0, text='Preparing experiment…')
        try:
            result = train(df, **config, progress=lambda done, total, name: progress.progress(done / total, text=f'Fitting {name}…'))
            result['duplicates_removed'] = dataset['duplicates']
            st.session_state.result = result
            st.session_state.pop('scored_batch', None)
            st.session_state.page = 'Model lab'
            progress.progress(1.0, text='Experiment complete')
        except (ValueError, MemoryError) as exc:
            st.session_state.pop('result', None)
            st.error(f'Training needs attention: {exc}')
        finally:
            progress.empty()
    result = st.session_state.get('result')
    columns = st.columns(4)
    columns[0].metric('Transactions', f'{len(df):,}', help='Validated, deduplicated rows.')
    columns[1].metric('Known fraud', f'{int(df.Class.sum()):,}')
    columns[2].metric('Fraud prevalence', f'{df.Class.mean():.2%}')
    columns[3].metric('Experiment', 'Complete' if result else 'Ready', help='Train all models to generate an evaluation.')
    st.write('')
    page = st.radio('Workspace', PAGES, horizontal=True, key='page', label_visibility='collapsed')
    st.divider()
    if page == 'Overview':
        overview(dataset)
    elif page == 'Model lab':
        model_lab(result, dataset['origin'])
    elif page == 'Transaction review':
        transaction_review(result, df)
    else:
        guide()
    st.divider()
    st.caption('FraudLens · Built for investigation and learning. Predictions support human judgment.')


if __name__ == '__main__':
    main()
