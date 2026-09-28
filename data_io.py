"""CSV boundaries shared by the dashboard and command-line tools."""
import csv
import io
from pathlib import Path

import pandas as pd


def read_csv(source):
    """Read UTF-8 CSV without silently accepting duplicate or ambiguous headers."""
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding='utf-8-sig')
    elif isinstance(source, bytes):
        text = source.decode('utf-8-sig')
    else:
        source.seek(0)
        content = source.read()
        text = content.decode('utf-8-sig') if isinstance(content, bytes) else content
    text = text.lstrip('\ufeff')
    if not text.strip():
        raise ValueError('The CSV is empty. Include a header and transaction rows.')
    try:
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        names = [name.strip() for name in header]
        if not names or any(not name for name in names):
            raise ValueError('Every CSV column needs a name.')
        if len(set(names)) != len(names):
            raise ValueError('Column names must be unique, including after trimming spaces.')
        for line, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(f'CSV record {line} has {len(row)} fields; expected {len(header)}.')
        numeric = {'Time', 'Amount', 'Class', 'Fraud_score', 'Predicted_class', 'Decision_threshold'}
        numeric.update(f'V{i}' for i in range(1, 29))
        # Identifiers such as 000123 and literal NA must survive ingestion intact.
        converters = {name: str for name in names if name not in numeric}
        return pd.read_csv(io.StringIO(text), header=0, names=names, converters=converters)
    except (csv.Error, pd.errors.ParserError) as exc:
        raise ValueError(f'Could not parse the CSV: {exc}') from exc


def export_csv(frame):
    """Escape spreadsheet formulas in text metadata while leaving numbers numeric."""
    output = frame.copy()
    for column in output.select_dtypes(include=['object', 'string']).columns:
        output[column] = output[column].map(
            lambda value: "'" + value
            if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r', '\n'))
            else value
        )
    return output.to_csv(index=False)
