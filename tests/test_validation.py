import pandas as pd
import pytest

from src.validation import EXPECTED_CATEGORIES, format_report, validate


def _row(customer_id: str, **overrides) -> dict:
    row = {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 358.2,
        "Churn": "No",
    }
    row.update(overrides)
    return row


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return pd.DataFrame([_row("A"), _row("B", Churn="Yes", tenure=0, TotalCharges=0.0)])


def test_clean_data_passes(clean_df):
    result = validate(clean_df)

    assert result.rows == 2
    assert result.columns == 21
    assert result.issues == []
    assert result.status == "PASS"
    assert "Overall status: PASS" in format_report(result)


def test_duplicate_customer_ids_detected(clean_df):
    df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)

    result = validate(df)

    assert result.duplicate_ids == 1
    assert result.duplicate_id_examples == ["A"]
    assert result.status == "REVIEW"


def test_blank_total_charges_reported_as_missing_and_unconvertible():
    df = pd.DataFrame([_row("A", TotalCharges="10.5"), _row("B", tenure=0, TotalCharges=" ")])

    result = validate(df)

    assert result.missing_values == {"TotalCharges": 1}
    assert result.type_mismatches["TotalCharges"] == ("text", "decimal number")
    assert result.total_charges_unconvertible == 1
    assert result.total_charges_blank == 1
    assert result.total_charges_blank_with_zero_tenure == 1
    assert "TotalCharges conversion issues: 1 value(s)" in format_report(result)


def test_non_numeric_total_charges_counted():
    df = pd.DataFrame([_row("A", TotalCharges="10.5"), _row("B", TotalCharges="abc")])

    result = validate(df)

    assert result.total_charges_unconvertible == 1
    assert result.total_charges_blank == 0


def test_tenure_out_of_range_and_negative_values():
    df = pd.DataFrame(
        [
            _row("A", tenure=-1),
            _row("B", tenure=500),
            _row("C", MonthlyCharges=-5.0, TotalCharges=-5.0),
        ]
    )

    result = validate(df)

    assert result.tenure_out_of_range == 2
    assert result.negative_values == {"tenure": 1, "MonthlyCharges": 1, "TotalCharges": 1}
    assert result.status == "REVIEW"


def test_invalid_target_labels_detected():
    df = pd.DataFrame([_row("A", Churn="yes"), _row("B", Churn=None), _row("C")])

    result = validate(df)

    assert result.invalid_target == {"yes": 1, "(empty)": 1}


def test_unexpected_categorical_values_detected():
    df = pd.DataFrame([_row("A", gender="F"), _row("B", Contract="Three year"), _row("C")])

    result = validate(df)

    assert result.unexpected_categories == {"gender": {"F": 1}, "Contract": {"Three year": 1}}
    report = format_report(result)
    assert '"Three year" x1' in report


def test_missing_columns_flagged(clean_df):
    result = validate(clean_df.drop(columns=["Churn", "TotalCharges"]))

    assert result.missing_columns == ["TotalCharges", "Churn"]
    assert result.status == "REVIEW"
    report = format_report(result)
    assert "Invalid target values: not checked" in report
    assert "TotalCharges conversion issues: not checked" in report


def test_raw_dataset_matches_expected_categories():
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

    result = validate(df)

    assert result.rows == 7043
    assert result.columns == 21
    assert result.duplicate_ids == 0
    assert result.unexpected_categories == {}
    assert set(EXPECTED_CATEGORIES) <= set(df.columns)
    assert result.total_charges_unconvertible == 11
    assert result.status == "REVIEW"
