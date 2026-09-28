"""Train a reproducible experiment and export models, metrics, and a run manifest."""
import argparse
import json
from pathlib import Path

import joblib

from data_io import export_csv, read_csv
from ml import POLICIES, SPLITS, demo_data, run_metadata, train


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='Labelled creditcard.csv; omit for a synthetic demo.')
    parser.add_argument('--output', type=Path, default=Path('outputs/latest'))
    parser.add_argument('--split', choices=SPLITS, default='stratified')
    parser.add_argument('--policy', choices=POLICIES, default='f1')
    parser.add_argument('--target-recall', type=float, default=.8)
    parser.add_argument('--retained', type=float, default=.95, help='PCA variance fraction between 0 and 1.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing experiment in the output folder.')
    args = parser.parse_args(argv)
    paths = [args.output / name for name in ('metrics.csv', 'models.joblib', 'run.json')]
    if not args.overwrite and any(path.exists() for path in paths):
        parser.error('Output already contains an experiment. Choose a new folder or pass --overwrite.')
    try:
        data = read_csv(args.csv) if args.csv else demo_data()
        result = train(data, retained=args.retained, split=args.split, policy=args.policy, target_recall=args.target_recall)
        manifest = run_metadata(result, str(args.csv) if args.csv else 'SYNTHETIC DEMO')
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'metrics.csv').write_text(export_csv(result['metrics']), encoding='utf-8')
        joblib.dump(result, args.output / 'models.joblib', compress=3)
        (args.output / 'run.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (ValueError, OSError, UnicodeError, MemoryError) as exc:
        parser.error(str(exc))
    print(f'Recommended: {result["best"]}')
    print(result['metrics'].to_string(index=False))
    print(f'Experiment saved to {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
