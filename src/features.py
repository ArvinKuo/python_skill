"""Task 6: define the model's features (X) and target (y).

`get_features_and_target()` is the only place that splits cleaned data into
X, y and customer IDs. It accepts either the result of `clean()` or the
cleaned CSV (which contains customerID, the 41 features and Churn side by
side) and refuses to return an X that contains the identifier, the target or
any column that is not one of the expected features.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api import types as ptypes

from src.cleaning import CleanedData, feature_columns
from src.validation import ID_COLUMN, TARGET_COLUMN

EXCLUDED_COLUMNS = {
    ID_COLUMN: "unique identifier with no relationship to churn; kept separately for the prediction output",
    TARGET_COLUMN: "the target the model predicts; using it as an input would leak the answer",
}


class FeatureError(ValueError):
    """Raised when the features or target are not safe to give to a model."""


@dataclass
class ModelData:
    X: pd.DataFrame
    y: pd.Series | None
    customer_ids: pd.Series

    def summary_lines(self) -> list[str]:
        lines = [
            f"Features (X): {self.X.shape[0]:,} rows x {self.X.shape[1]} columns, all numeric, no missing values",
        ]
        if self.y is not None:
            churned = int(self.y.sum())
            lines.append(
                f"Target (y): {TARGET_COLUMN} -> {churned:,} churned (1), "
                f"{len(self.y) - churned:,} stayed (0)"
            )
        else:
            lines.append(f"Target (y): none ({TARGET_COLUMN} not present)")
        for col, reason in EXCLUDED_COLUMNS.items():
            lines.append(f"Excluded from X: {col} ({reason})")
        return lines


def validate_features(X: pd.DataFrame) -> None:
    """Raise FeatureError unless X has exactly the expected numeric feature columns."""
    leaked = [col for col in X.columns if col in EXCLUDED_COLUMNS]
    if leaked:
        raise FeatureError(
            f"X must not contain {', '.join(leaked)}: "
            + "; ".join(f"{col} is {EXCLUDED_COLUMNS[col]}" for col in leaked)
        )

    expected = feature_columns()
    actual = list(X.columns)
    if actual != expected:
        unexpected = [c for c in actual if c not in expected]
        missing = [c for c in expected if c not in actual]
        problems = []
        if unexpected:
            problems.append(f"unexpected column(s): {', '.join(map(str, unexpected))}")
        if missing:
            problems.append(f"missing column(s): {', '.join(missing)}")
        if not problems:
            problems.append("columns are in a different order than expected")
        raise FeatureError("X does not match the expected feature list - " + "; ".join(problems))

    non_numeric = [c for c in X.columns if not ptypes.is_numeric_dtype(X[c])]
    if non_numeric:
        raise FeatureError(f"X contains non-numeric column(s): {', '.join(non_numeric)}")

    with_missing = X.columns[X.isna().any()].tolist()
    if with_missing:
        raise FeatureError(f"X contains missing values in: {', '.join(with_missing)}")


def _validate_target(y: pd.Series) -> pd.Series:
    if y.isna().any():
        raise FeatureError(f"{TARGET_COLUMN} contains {int(y.isna().sum())} missing value(s).")
    invalid = sorted(set(y.unique()) - {0, 1}, key=str)
    if invalid:
        raise FeatureError(f"{TARGET_COLUMN} must contain only 0 and 1, found: {invalid}")
    return y.astype(int)


def get_features_and_target(
    data: CleanedData | pd.DataFrame, require_target: bool = True
) -> ModelData:
    """Split cleaned data into X, y and customer IDs.

    `data` is either the result of `clean()` or a cleaned table with
    customerID, the feature columns and (optionally) Churn, e.g. the CSV
    saved in data/processed/.
    """
    cleaned_df = data.to_frame() if isinstance(data, CleanedData) else data

    if ID_COLUMN not in cleaned_df.columns:
        raise FeatureError(f"{ID_COLUMN} is required to match predictions back to customers.")
    customer_ids = cleaned_df[ID_COLUMN]
    if customer_ids.isna().any():
        raise FeatureError(f"{ID_COLUMN} contains {int(customer_ids.isna().sum())} missing value(s).")

    if TARGET_COLUMN in cleaned_df.columns:
        y = _validate_target(cleaned_df[TARGET_COLUMN])
    elif require_target:
        raise FeatureError(f"{TARGET_COLUMN} is required to train or evaluate a model.")
    else:
        y = None

    X = cleaned_df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in cleaned_df.columns])
    validate_features(X)
    return ModelData(X=X, y=y, customer_ids=customer_ids)
