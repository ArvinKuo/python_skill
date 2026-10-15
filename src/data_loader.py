"""Load the raw Telco Customer Churn CSV and perform basic input checks.

This module also defines the dataset's column names, which every other
module imports from here, so they are written down in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

REQUIRED_COLUMNS = [
    ID_COLUMN,
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COLUMN,
]


class DataLoadError(ValueError):
    """Raised when the input file is missing, unreadable or not the expected dataset."""


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV unchanged, after checking it exists, has rows and has every required column.

    Values are not converted or cleaned here; content checks are done by
    `validator.validate()` and conversions by `preprocessing.clean()`.
    """
    path = Path(path)
    if not path.is_file():
        raise DataLoadError(f"Input file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise DataLoadError(f"Input file is empty: {path}") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise DataLoadError(f"Input file is not a readable CSV: {path} ({error})") from error

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataLoadError(f"Input file is missing required column(s): {', '.join(missing)}")
    if df.empty:
        raise DataLoadError(f"Input file has column headers but no data rows: {path}")
    return df
