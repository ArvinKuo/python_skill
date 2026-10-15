"""Prepare model inputs from the raw Telco Customer Churn data.

1. clean()                    convert TotalCharges, encode the categorical columns
                              and Churn; returns features, target and customer IDs
                              separately and never modifies its input
2. get_features_and_target()  X, y and customer IDs, with checks that no identifier
                              or target column can reach the model
3. split_data()               reproducible, stratified train/test split
4. build_preprocessor()       scikit-learn transformer that scales the numeric
                              columns; fitted later, on the training set only

The reasoning behind every cleaning rule is documented in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as ptypes
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import ID_COLUMN, TARGET_COLUMN
from src.validator import EXPECTED_CATEGORIES

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

EXCLUDED_COLUMNS = {
    ID_COLUMN: "unique identifier with no relationship to churn; kept separately for the prediction output",
    TARGET_COLUMN: "the target the model predicts; using it as an input would leak the answer",
}

TEST_SIZE = 0.20
RANDOM_STATE = 42


class DataCleaningError(ValueError):
    """Raised when the data cannot be cleaned without guessing."""


class FeatureError(ValueError):
    """Raised when the features or target are not safe to give to a model."""


# --- 1. Cleaning and encoding -------------------------------------------------


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


def strip_text(series: pd.Series) -> pd.Series:
    """Remove surrounding spaces from text values; other values are unchanged."""
    return series.map(lambda v: v.strip() if isinstance(v, str) else v)


def _raise_on_unknown(col: str, original: pd.Series, mapped: pd.Series) -> None:
    unknown = mapped.isna()
    if unknown.any():
        examples = ", ".join(repr(v) for v in original[unknown].unique()[:3])
        raise DataCleaningError(
            f"{col} contains {int(unknown.sum())} unexpected or missing value(s): {examples}."
        )


def encode_target(churn: pd.Series) -> pd.Series:
    """'Yes' -> 1, 'No' -> 0; any other or missing value raises DataCleaningError."""
    target = strip_text(churn).map(TARGET_MAPPING)
    _raise_on_unknown(TARGET_COLUMN, churn, target)
    return target.astype(int).rename(TARGET_COLUMN)


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DataCleaningError(f"Required column(s) missing: {', '.join(missing)}")


def _clean_numeric(df: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    numeric = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLUMNS:
        raw = strip_text(df[col]).replace("", pd.NA)
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

    negative = {col: int((numeric[col] < 0).sum()) for col in NUMERIC_COLUMNS}
    negative = {col: n for col, n in negative.items() if n}
    if negative:
        raise DataCleaningError(
            "Negative values are not valid for months or charges: "
            + ", ".join(f"{col} ({n})" for col, n in negative.items())
            + "."
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
        mapped = strip_text(df[col]).map(YES_NO_MAPPING)
        _raise_on_unknown(col, df[col], mapped)
        encoded[col] = mapped.astype(int)

    for col, categories in ONE_HOT_COLUMNS.items():
        values = strip_text(df[col])
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


def clean(df: pd.DataFrame) -> CleanedData:
    """Return encoded features, binary target and customer IDs. `df` is not modified."""
    _require_columns(df, [ID_COLUMN, *NUMERIC_COLUMNS, *EXPECTED_CATEGORIES])
    df = df.copy()
    log: list[str] = []

    customer_ids = strip_text(df[ID_COLUMN]).rename(ID_COLUMN)
    log.append(f"{ID_COLUMN}: removed from the features and kept separately for the prediction output.")

    numeric = _clean_numeric(df, log)
    categorical = _encode_categorical(df, log)

    if TARGET_COLUMN in df.columns:
        target = encode_target(df[TARGET_COLUMN])
        log.append(f"{TARGET_COLUMN}: converted to 1 = Yes (left), 0 = No (stayed).")
    else:
        target = None
        log.append(f"{TARGET_COLUMN}: column not present (prediction-only data); no target created.")

    features = pd.concat([numeric, categorical], axis=1)
    log.append(f"Rows removed: 0. Final feature matrix: {features.shape[0]} rows x {features.shape[1]} columns.")
    return CleanedData(features=features, target=target, customer_ids=customer_ids, log=log)


def feature_columns() -> list[str]:
    """The exact, fixed list of feature columns `clean()` produces."""
    columns = [*NUMERIC_COLUMNS, *ALREADY_BINARY_COLUMNS, *YES_NO_COLUMNS]
    for col, categories in ONE_HOT_COLUMNS.items():
        columns += [f"{col}_{cat}" for cat in categories]
    return columns


# --- 2. Features (X) and target (y) -------------------------------------------


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


# --- 3. Train/test split ------------------------------------------------------


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series
    test_size: float
    random_state: int

    def summary_lines(self) -> list[str]:
        lines = [
            f"Settings: test_size={self.test_size}, random_state={self.random_state}, "
            "stratify=Churn (same churn share in both sets)"
        ]
        for name, y in (("Training set", self.y_train), ("Test set", self.y_test)):
            lines.append(
                f"{name}: {len(y):,} customers, {int(y.sum()):,} churned ({y.mean() * 100:.2f}%)"
            )
        return lines


def split_data(
    data: ModelData, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> DataSplit:
    """Stratified, reproducible split; customer IDs are split together with X and y."""
    if data.y is None:
        raise FeatureError("A target (Churn) is required to split data for training.")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        data.X,
        data.y,
        data.customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=data.y,
    )
    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ids_train=ids_train,
        ids_test=ids_test,
        test_size=test_size,
        random_state=random_state,
    )


# --- 4. Scaling (fitted inside the model pipeline) -----------------------------


def build_preprocessor() -> ColumnTransformer:
    """Unfitted transformer: StandardScaler on the numeric columns, 0/1 columns unchanged.

    It learns the mean and standard deviation only when the model pipeline is
    fitted, i.e. from the training set; the test set reuses those values.
    """
    return ColumnTransformer(
        transformers=[("scale_numeric", StandardScaler(), NUMERIC_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
