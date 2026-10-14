"""Data cleaning and encoding for the Telco Customer Churn dataset.

`clean()` never modifies the DataFrame it receives; it returns a new feature
matrix, the binary target, the customer IDs (kept separately for the final
prediction output) and a plain-language log of every change it made.
See README.md, "Task 4 - Data cleaning", for the reasoning behind each rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.validation import EXPECTED_CATEGORIES, ID_COLUMN, TARGET_COLUMN

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_MAPPING = {"Yes": 1, "No": 0}
YES_NO_MAPPING = {"Yes": 1, "No": 0}

ALREADY_BINARY_COLUMNS = ["SeniorCitizen"]
YES_NO_COLUMNS = [
    col for col, allowed in EXPECTED_CATEGORIES.items() if allowed == set(YES_NO_MAPPING)
]
# Category order is fixed (sorted) so the encoded columns are always identical,
# regardless of which values happen to appear in a given file.
ONE_HOT_COLUMNS = {
    col: sorted(allowed)
    for col, allowed in EXPECTED_CATEGORIES.items()
    if col not in YES_NO_COLUMNS and col not in ALREADY_BINARY_COLUMNS
}


class DataCleaningError(ValueError):
    """Raised when the data cannot be cleaned without guessing."""


@dataclass
class CleanedData:
    features: pd.DataFrame
    target: pd.Series | None
    customer_ids: pd.Series
    log: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """customerID + encoded features (+ Churn) in one table, for saving to disk."""
        parts = [self.customer_ids, self.features]
        if self.target is not None:
            parts.append(self.target)
        return pd.concat(parts, axis=1)


def _strip_text(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v.strip() if isinstance(v, str) else v)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataCleaningError(f"Required column(s) missing: {', '.join(missing)}")


def _clean_numeric(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        raw = _strip_text(df[col]).replace("", pd.NA)
        converted = pd.to_numeric(raw, errors="coerce")
        bad_text = converted.isna() & raw.notna()
        if bad_text.any():
            examples = ", ".join(repr(v) for v in raw[bad_text].unique()[:3])
            raise DataCleaningError(
                f"{col} contains {int(bad_text.sum())} non-numeric value(s) ({examples})."
            )
        numeric[col] = converted.astype(float)

    for col in ("tenure", "MonthlyCharges"):
        missing = int(numeric[col].isna().sum())
        if missing:
            raise DataCleaningError(
                f"{col} has {missing} missing value(s); there is no reliable way to fill them."
            )

    blanks = int(df["TotalCharges"].map(lambda v: isinstance(v, str) and not v.strip()).sum())
    if blanks:
        log.append(
            f"TotalCharges: converted from text to a number; {blanks} blank value(s) found."
        )
    else:
        log.append("TotalCharges: converted from text to a number.")

    missing_total = numeric["TotalCharges"].isna()
    new_customers = missing_total & (numeric["tenure"] == 0)
    if new_customers.any():
        numeric.loc[new_customers, "TotalCharges"] = 0.0
        log.append(
            f"TotalCharges: {int(new_customers.sum())} missing value(s) for new customers "
            "(tenure = 0) set to 0, because they have not been billed yet."
        )

    existing_customers = missing_total & (numeric["tenure"] > 0)
    if existing_customers.any():
        numeric.loc[existing_customers, "TotalCharges"] = (
            numeric.loc[existing_customers, "tenure"]
            * numeric.loc[existing_customers, "MonthlyCharges"]
        )
        log.append(
            f"TotalCharges: {int(existing_customers.sum())} missing value(s) for existing "
            "customers estimated as tenure x MonthlyCharges."
        )

    return numeric


def _encode_categorical(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    encoded = pd.DataFrame(index=df.index)

    for col in ALREADY_BINARY_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise DataCleaningError(f"{col} must contain only 0 and 1.")
        encoded[col] = values.astype(int)

    for col in YES_NO_COLUMNS:
        mapped = _strip_text(df[col]).map(YES_NO_MAPPING)
        _raise_on_unknown(col, df[col], mapped)
        encoded[col] = mapped.astype(int)

    for col, categories in ONE_HOT_COLUMNS.items():
        values = _strip_text(df[col])
        as_category = pd.Categorical(values, categories=categories)
        _raise_on_unknown(col, df[col], pd.Series(as_category, index=df.index))
        dummies = pd.get_dummies(as_category, prefix=col, prefix_sep="_", dtype=int)
        dummies.index = df.index
        encoded = pd.concat([encoded, dummies], axis=1)

    log.append(f"{', '.join(ALREADY_BINARY_COLUMNS)}: already 0/1, kept as is.")
    log.append(f"Yes/No columns converted to 1/0: {', '.join(YES_NO_COLUMNS)}.")
    log.append(
        "One-hot encoded (one 0/1 column per category): "
        + ", ".join(f"{col} ({len(cats)})" for col, cats in ONE_HOT_COLUMNS.items())
        + "."
    )
    return encoded


def _raise_on_unknown(col: str, original: pd.Series, mapped: pd.Series) -> None:
    unknown = mapped.isna()
    if unknown.any():
        examples = ", ".join(repr(v) for v in original[unknown].unique()[:3])
        raise DataCleaningError(
            f"{col} contains {int(unknown.sum())} unexpected or missing value(s): {examples}."
        )


def _encode_target(df: pd.DataFrame, log: list[str]) -> pd.Series | None:
    if TARGET_COLUMN not in df.columns:
        log.append(f"{TARGET_COLUMN}: column not present (prediction-only data); no target created.")
        return None
    target = _strip_text(df[TARGET_COLUMN]).map(TARGET_MAPPING)
    _raise_on_unknown(TARGET_COLUMN, df[TARGET_COLUMN], target)
    log.append(f"{TARGET_COLUMN}: converted to 1 = Yes (left), 0 = No (stayed).")
    return target.astype(int).rename(TARGET_COLUMN)


def clean(df: pd.DataFrame) -> CleanedData:
    """Return encoded features, binary target and customer IDs. `df` is not modified."""
    _require_columns(df, [ID_COLUMN, *NUMERIC_COLUMNS, *EXPECTED_CATEGORIES])
    df = df.copy()
    log: list[str] = []

    customer_ids = _strip_text(df[ID_COLUMN]).rename(ID_COLUMN)
    log.append(f"{ID_COLUMN}: removed from the features and kept separately for the prediction output.")

    numeric = _clean_numeric(df, log)
    categorical = _encode_categorical(df, log)
    target = _encode_target(df, log)

    features = pd.concat([numeric, categorical], axis=1)
    log.append(f"Rows removed: 0. Final feature matrix: {features.shape[0]} rows x {features.shape[1]} columns.")
    return CleanedData(features=features, target=target, customer_ids=customer_ids, log=log)


def feature_columns() -> list[str]:
    """The exact, fixed list of feature columns `clean()` produces."""
    columns = [*NUMERIC_COLUMNS, *ALREADY_BINARY_COLUMNS, *YES_NO_COLUMNS]
    for col, categories in ONE_HOT_COLUMNS.items():
        columns += [f"{col}_{cat}" for cat in categories]
    return columns
