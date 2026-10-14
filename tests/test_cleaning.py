import pandas as pd
import pytest

from src.cleaning import DataCleaningError, clean, feature_columns


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
        "MonthlyCharges": 30.0,
        "TotalCharges": "360.5",
        "Churn": "No",
    }
    row.update(overrides)
    return row


def test_input_dataframe_is_not_modified():
    df = pd.DataFrame([_row("A"), _row("B", tenure=0, TotalCharges=" ")])
    original = df.copy()

    clean(df)

    pd.testing.assert_frame_equal(df, original)


def test_customer_id_kept_separately_not_in_features():
    df = pd.DataFrame([_row("A"), _row("B")])

    result = clean(df)

    assert "customerID" not in result.features.columns
    assert result.customer_ids.tolist() == ["A", "B"]
    assert result.to_frame().columns[0] == "customerID"


def test_total_charges_converted_to_number():
    result = clean(pd.DataFrame([_row("A", TotalCharges="360.5")]))

    assert result.features["TotalCharges"].dtype == float
    assert result.features.loc[0, "TotalCharges"] == 360.5


def test_blank_total_charges_for_new_customer_becomes_zero():
    df = pd.DataFrame([_row("A", tenure=0, MonthlyCharges=52.55, TotalCharges=" ")])

    result = clean(df)

    assert result.features.loc[0, "TotalCharges"] == 0.0
    assert any("set to 0" in step for step in result.log)


def test_blank_total_charges_for_existing_customer_is_estimated():
    df = pd.DataFrame([_row("A", tenure=10, MonthlyCharges=20.0, TotalCharges="")])

    result = clean(df)

    assert result.features.loc[0, "TotalCharges"] == pytest.approx(200.0)
    assert any("estimated" in step for step in result.log)


def test_non_numeric_total_charges_raises():
    with pytest.raises(DataCleaningError, match="non-numeric"):
        clean(pd.DataFrame([_row("A", TotalCharges="abc")]))


def test_missing_monthly_charges_raises():
    with pytest.raises(DataCleaningError, match="MonthlyCharges"):
        clean(pd.DataFrame([_row("A", MonthlyCharges=None)]))


def test_churn_mapped_to_binary():
    df = pd.DataFrame([_row("A", Churn="Yes"), _row("B", Churn="No")])

    result = clean(df)

    assert result.target.tolist() == [1, 0]


def test_unexpected_churn_label_raises():
    with pytest.raises(DataCleaningError, match="Churn"):
        clean(pd.DataFrame([_row("A", Churn="Maybe")]))


def test_missing_churn_column_allowed_for_prediction_data():
    result = clean(pd.DataFrame([_row("A")]).drop(columns=["Churn"]))

    assert result.target is None
    assert "Churn" not in result.to_frame().columns


def test_yes_no_and_one_hot_encoding():
    df = pd.DataFrame([_row("A", Partner="Yes", Dependents="No", Contract="Two year")])

    features = clean(df).features

    assert features.loc[0, "Partner"] == 1
    assert features.loc[0, "Dependents"] == 0
    assert features.loc[0, "Contract_Two year"] == 1
    assert features.loc[0, "Contract_Month-to-month"] == 0
    assert features.loc[0, "Contract_One year"] == 0


def test_encoded_columns_are_fixed_even_if_category_absent():
    one_row = clean(pd.DataFrame([_row("A")])).features
    other_row = clean(pd.DataFrame([_row("B", Contract="One year", gender="Male")])).features

    assert list(one_row.columns) == feature_columns()
    assert list(other_row.columns) == feature_columns()


def test_unexpected_category_raises():
    with pytest.raises(DataCleaningError, match="Contract"):
        clean(pd.DataFrame([_row("A", Contract="Three year")]))


def test_surrounding_whitespace_is_ignored():
    df = pd.DataFrame([_row(" A ", Partner=" Yes ", Contract=" One year", Churn="Yes ")])

    result = clean(df)

    assert result.customer_ids.tolist() == ["A"]
    assert result.features.loc[0, "Partner"] == 1
    assert result.features.loc[0, "Contract_One year"] == 1
    assert result.target.tolist() == [1]


def test_raw_dataset_cleans_completely():
    raw = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

    result = clean(raw)

    assert result.features.shape == (7043, len(feature_columns()))
    assert result.features.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in result.features.dtypes)
    assert set(result.target.unique()) == {0, 1}
    assert result.customer_ids.is_unique
    new_customers = raw["tenure"] == 0
    assert (result.features.loc[new_customers, "TotalCharges"] == 0).all()
