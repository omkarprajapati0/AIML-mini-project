"""Score transactions with a trusted locally trained FraudLens artifact."""
import argparse
import warnings
from pathlib import Path

import joblib
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from data_io import export_csv, read_csv
from ml import ARTIFACT_VERSION, predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Only load models you created or trust: joblib loading can execute code.')
    parser.add_argument('--model', type=Path, required=True, help='Trusted models.joblib from train.py.')
    parser.add_argument('--csv', type=Path, required=True, help='Transaction CSV; labels optional.')
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    parser.add_argument('--name', help='Model name; defaults to the validation-selected model.')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.csv.resolve(), args.model.resolve()):
        parser.error('Prediction output must not replace the input CSV or model.')
    if args.output.exists() and not args.overwrite:
        parser.error('Output exists. Choose a new path or pass --overwrite.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', InconsistentVersionWarning)
            result = joblib.load(args.model)
        if not isinstance(result, dict) or result.get('artifact_version') != ARTIFACT_VERSION:
            raise ValueError('Unsupported artifact. Retrain using the current train.py.')
        if result.get('versions', {}).get('scikit-learn') != sklearn.__version__:
            raise ValueError('scikit-learn version differs from training. Use the original environment or retrain.')
        name = args.name or result['best']
        scored = predict(result, name, read_csv(args.csv))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(export_csv(scored), encoding='utf-8')
    except InconsistentVersionWarning:
        parser.error('Model package versions differ. Use the training environment or retrain.')
    except (ValueError, OSError, UnicodeError, KeyError, EOFError) as exc:
        parser.error(str(exc))
    print(f'{len(scored):,} rows scored; {scored.Predicted_class.sum():,} flagged with {name}.')
    print(f'Predictions saved to {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
