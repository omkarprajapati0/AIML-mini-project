"""Human-readable experiment reports generated only from measured run results."""
import re

from ml import POLICIES, SPLITS


def _cell(value):
    text = str(value).replace('\n', ' ').replace('\r', ' ')
    return re.sub(r'([\\`*_{}\[\]<>|])', r'\\\1', text)


def experiment_report(result, source):
    """Export a self-contained Markdown report without transaction-level records."""
    config = result['config']
    selected = result['metrics'].set_index('Model').loc[result['best']]
    entry = result['models'][result['best']]
    lines = [
        '# FraudLens experiment report', '',
        f'**Source:** {_cell(source)}', '',
        f'**Run time (UTC):** {_cell(result["created_at"])}', '',
        '**Interpretation:** If the source is synthetic, these are demonstration results, '
        'not Kaggle or real-world performance.', '',
        '## Experiment settings', '',
        f'- Split: {SPLITS[config["split"]]}',
        f'- Seed: {config["seed"]}',
        f'- PCA retained variance: {config["retained"]:.0%}',
        f'- Decision policy: {POLICIES[config["policy"]]}',
        f'- Validated rows: {result["rows"]:,}',
        f'- Duplicate rows removed: {result["duplicates_removed"]:,}',
    ]
    if config['policy'] == 'recall':
        lines.append(f'- Validation recall target: {config["target_recall"]:.0%} (not a test or future-data guarantee)')
    lines += ['', '## Split audit', '', '| Partition | Rows | Fraud | Prevalence |', '| --- | ---: | ---: | ---: |']
    for split in result['split_summary']:
        lines.append(f'| {split["Split"]} | {split["Rows"]:,} | {split["Fraud"]:,} | {split["Prevalence"]:.2%} |')
    lines += ['', '## Model comparison', '',
              'The recommendation uses validation average precision. All other metrics below use the held-out test set.', '',
              '| Model | Validation AP | Test AP | Precision | Recall | F1 | Review rate |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['metrics'].to_dict(orient='records'):
        values = [f'{row[key]:.4f}' for key in ['Validation AP', 'Average precision', 'Precision', 'Recall', 'F1', 'Flag rate']]
        lines.append('| ' + _cell(row['Model']) + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Selected model', '',
              f'**{_cell(result["best"])}**, selected on validation data.', '',
              f'The frozen threshold is **{entry["threshold"]:.6f}**. On the test split, this model '
              f'achieved precision **{selected["Precision"]:.1%}**, recall **{selected["Recall"]:.1%}**, '
              f'and average precision **{selected["Average precision"]:.4f}**. '
              f'It produced **{int(selected["False alarms"]):,} false alarms** and '
              f'**{int(selected["Missed fraud"]):,} missed fraud transactions**.', '',
              '## Reproducibility', '',
              f'Dataset SHA-256: `{result["fingerprint"]}`', '',
              ', '.join(f'{_cell(name)} {_cell(version)}' for name, version in result['versions'].items()), '',
              '## Warnings and limitations', '']
    lines += [f'- {_cell(note)}' for note in result['notes']] or ['- No training or small-partition warnings were recorded.']
    lines += ['- Decision scores are uncalibrated margins, not probabilities.',
              '- Thresholds and preprocessing use training/validation data only; no post-evaluation refit occurs.',
              '- Repeated configuration selection from test results biases reported performance.',
              '- Feature contributions explain the linear calculation, not the cause of fraud.',
              '- Results from one holdout do not establish production readiness or demographic fairness.',
              '- Predictions support human review and never automatically block payments.', '']
    return '\n'.join(lines)
