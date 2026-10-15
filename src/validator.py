"""Validate the raw Telco Customer Churn data.

`validate()` checks required columns, missing values, duplicate IDs, data
types, value ranges, target labels and category values, and returns a
`ValidationResult`. `format_report()` turns it into a plain-language report.
Nothing is changed here; problems are only reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as ptypes

from src.data_loader import ID_COLUMN, REQUIRED_COLUMNS, TARGET_COLUMN

EXPECTED_TARGET_LABELS = {"Yes", "No"}
NON_NEGATIVE_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
TENURE_RANGE = (0, 120)  # months; 10 years is a generous upper bound

_YES_NO = {"Yes", "No"}
_PHONE_ADDON = {"Yes", "No", "No phone service"}
_INTERNET_ADDON = {"Yes", "No", "No internet service"}

EXPECTED_CATEGORIES: dict[str, set] = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": _PHONE_ADDON,
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    },
}

TEXT = "text"
WHOLE_NUMBER = "whole number"
DECIMAL_NUMBER = "decimal number"

EXPECTED_TYPES: dict[str, str] = {
    ID_COLUMN: TEXT,
    **{col: TEXT for col in EXPECTED_CATEGORIES},
    "SeniorCitizen": WHOLE_NUMBER,
    "tenure": WHOLE_NUMBER,
    "MonthlyCharges": DECIMAL_NUMBER,
    "TotalCharges": DECIMAL_NUMBER,
    TARGET_COLUMN: TEXT,
}


@dataclass
class ValidationResult:
    rows: int
    columns: int
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    column_types: dict[str, str] = field(default_factory=dict)
    type_mismatches: dict[str, tuple[str, str]] = field(default_factory=dict)
    duplicate_ids: int = 0
    duplicate_id_examples: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    total_charges_unconvertible: int = 0
    total_charges_blank: int = 0
    total_charges_blank_with_zero_tenure: int = 0
    tenure_out_of_range: int = 0
    tenure_not_whole: int = 0
    tenure_observed: tuple[float, float] | None = None
    negative_values: dict[str, int] = field(default_factory=dict)
    invalid_target: dict[str, int] = field(default_factory=dict)
    unexpected_categories: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def issues(self) -> list[str]:
        found = []
        if self.missing_columns:
            found.append(f"Expected columns are missing: {', '.join(self.missing_columns)}.")
        if self.duplicate_ids:
            found.append(f"{self.duplicate_ids} customer ID(s) appear more than once.")
        for col, count in self.missing_values.items():
            found.append(f"{col} has {count} empty value(s).")
        for col, (actual, expected) in self.type_mismatches.items():
            found.append(f"{col} is stored as {actual} but should be a {expected}.")
        if self.total_charges_unconvertible:
            found.append(
                f"{self.total_charges_unconvertible} TotalCharges value(s) cannot be read as a number."
            )
        if self.tenure_out_of_range or self.tenure_not_whole:
            found.append(
                f"{self.tenure_out_of_range + self.tenure_not_whole} tenure value(s) are outside "
                f"{TENURE_RANGE[0]}-{TENURE_RANGE[1]} months or are not whole months."
            )
        for col, count in self.negative_values.items():
            found.append(f"{col} has {count} negative value(s).")
        if self.invalid_target:
            found.append(f"{TARGET_COLUMN} has {sum(self.invalid_target.values())} invalid label(s).")
        for col, values in self.unexpected_categories.items():
            found.append(f"{col} has unexpected value(s): {_format_counts(values)}.")
        return found

    @property
    def status(self) -> str:
        return "REVIEW" if self.issues else "PASS"


def _describe_type(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series):
        return "yes/no flag"
    if ptypes.is_integer_dtype(series):
        return WHOLE_NUMBER
    if ptypes.is_float_dtype(series):
        return DECIMAL_NUMBER
    return TEXT


def _is_blank_value(value: object) -> bool:
    """True for NaN/None and for strings that are empty or whitespace only."""
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))


def _is_blank(series: pd.Series) -> pd.Series:
    return series.map(_is_blank_value).astype(bool)


def _type_matches(actual: str, expected: str) -> bool:
    # A whole-number column read with blanks becomes float; still numeric, so accept it.
    return actual == expected or (expected == DECIMAL_NUMBER and actual == WHOLE_NUMBER)


def validate(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult(rows=len(df), columns=df.shape[1])
    result.missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    result.extra_columns = [c for c in df.columns if c not in REQUIRED_COLUMNS]

    for col in df.columns:
        actual = _describe_type(df[col])
        result.column_types[col] = actual
        expected = EXPECTED_TYPES.get(col)
        if expected and not _type_matches(actual, expected):
            result.type_mismatches[col] = (actual, expected)

        blanks = int(_is_blank(df[col]).sum())
        if blanks:
            result.missing_values[col] = blanks

    if ID_COLUMN in df.columns:
        dup_mask = df[ID_COLUMN].duplicated(keep=False) & df[ID_COLUMN].notna()
        result.duplicate_ids = int(df.loc[dup_mask, ID_COLUMN].nunique())
        result.duplicate_id_examples = df.loc[dup_mask, ID_COLUMN].drop_duplicates().head(5).tolist()

    numeric = {
        col: pd.to_numeric(df[col], errors="coerce")
        for col in NON_NEGATIVE_COLUMNS
        if col in df.columns
    }

    if "TotalCharges" in df.columns:
        raw = df["TotalCharges"]
        blank = _is_blank(raw)
        unconvertible = numeric["TotalCharges"].isna() & raw.notna()
        result.total_charges_unconvertible = int(unconvertible.sum())
        result.total_charges_blank = int((blank & raw.notna()).sum())
        if "tenure" in numeric:
            result.total_charges_blank_with_zero_tenure = int(
                (unconvertible & blank & (numeric["tenure"] == 0)).sum()
            )

    if "tenure" in numeric:
        tenure = numeric["tenure"].dropna()
        low, high = TENURE_RANGE
        result.tenure_out_of_range = int(((tenure < low) | (tenure > high)).sum())
        result.tenure_not_whole = int((tenure % 1 != 0).sum())
        if not tenure.empty:
            result.tenure_observed = (float(tenure.min()), float(tenure.max()))

    for col, values in numeric.items():
        negatives = int((values < 0).sum())
        if negatives:
            result.negative_values[col] = negatives

    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN]
        bad = target[~target.isin(EXPECTED_TARGET_LABELS)]
        result.invalid_target = {
            ("(empty)" if _is_blank_value(v) else str(v)): int(n)
            for v, n in bad.value_counts(dropna=False).items()
        }

    for col, allowed in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            continue
        values = df[col][~_is_blank(df[col])]
        unexpected = values[~values.isin(allowed)]
        if not unexpected.empty:
            result.unexpected_categories[col] = {
                str(v): int(n) for v, n in unexpected.value_counts().items()
            }

    return result


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f'"{value}" x{n}' for value, n in counts.items())


def _fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def format_report(result: ValidationResult, source: str | None = None) -> str:
    none = "None found"
    lines = ["DATA VALIDATION REPORT", "----------------------"]
    if source:
        lines.append(f"Dataset: {source}")
    lines += [f"Rows: {result.rows}", f"Columns: {result.columns}"]
    if result.missing_columns:
        lines.append(f"Missing expected columns: {', '.join(result.missing_columns)}")
    if result.extra_columns:
        lines.append(f"Additional columns (not checked): {', '.join(result.extra_columns)}")

    if result.duplicate_ids:
        examples = ", ".join(result.duplicate_id_examples)
        lines.append(f"Duplicate customer IDs: {result.duplicate_ids} (e.g. {examples})")
    else:
        lines.append(f"Duplicate customer IDs: {none} - every customer appears once")

    if result.missing_values:
        fields = ", ".join(f"{c} ({n} empty)" for c, n in result.missing_values.items())
        lines.append(f"Missing-value fields: {fields}")
    else:
        lines.append(f"Missing-value fields: {none}")

    if result.invalid_target:
        lines.append(f"Invalid target values: {_format_counts(result.invalid_target)}")
    elif TARGET_COLUMN in result.column_types:
        lines.append(f'Invalid target values: {none} - {TARGET_COLUMN} only contains "Yes" / "No"')
    else:
        lines.append(f"Invalid target values: not checked ({TARGET_COLUMN} column missing)")

    if result.total_charges_unconvertible:
        total = result.total_charges_unconvertible

        def share(n: int) -> str:
            return f"all {n}" if n == total else f"{n} of them"

        detail = f"{total} value(s) cannot be read as a number"
        if result.total_charges_blank:
            detail += f"; {share(result.total_charges_blank)} are blank"
        if result.total_charges_blank_with_zero_tenure:
            detail += (
                f" and {share(result.total_charges_blank_with_zero_tenure)} belong to brand-new "
                "customers (tenure = 0) who have not been billed yet"
            )
        lines.append(f"TotalCharges conversion issues: {detail}")
    elif "TotalCharges" in result.column_types:
        lines.append(f"TotalCharges conversion issues: {none}")
    else:
        lines.append("TotalCharges conversion issues: not checked (column missing)")

    if result.negative_values:
        fields = ", ".join(f"{c} ({n})" for c, n in result.negative_values.items())
        lines.append(f"Negative numeric values: {fields}")
    else:
        lines.append(f"Negative numeric values: {none}")

    low, high = TENURE_RANGE
    tenure_label = f"Tenure outside {low}-{high} months"
    if result.tenure_observed is None:
        lines.append(f"{tenure_label}: not checked (no usable tenure values)")
    else:
        observed = f"observed {_fmt_number(result.tenure_observed[0])}-{_fmt_number(result.tenure_observed[1])}"
        bad = result.tenure_out_of_range + result.tenure_not_whole
        lines.append(f"{tenure_label}: {bad if bad else none} ({observed})")

    if result.unexpected_categories:
        lines.append("Unexpected categorical values:")
        for col, values in result.unexpected_categories.items():
            lines.append(f"  - {col}: {_format_counts(values)}")
    else:
        lines.append(f"Unexpected categorical values: {none}")

    lines.append("")
    lines.append("Column data types:")
    width = max((len(c) for c in result.column_types), default=0)
    for col, kind in result.column_types.items():
        note = ""
        if col in result.type_mismatches:
            note = f"  <-- expected {result.type_mismatches[col][1]}"
        lines.append(f"  {col.ljust(width)}  {kind}{note}")

    issues = result.issues
    if issues:
        lines.append("")
        lines.append("Items to review:")
        lines += [f"  {i}. {issue}" for i, issue in enumerate(issues, start=1)]

    lines.append("")
    lines.append(f"Overall status: {result.status}")
    return "\n".join(lines)
