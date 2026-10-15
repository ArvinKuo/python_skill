"""Shared test data.

`make_row` builds one valid raw customer record, so each test only has to
state the values it cares about. The `raw_df` ... `holdout_predictions`
fixtures run the real dataset through the pipeline once per test session;
tests must not modify them (raw_df is a fresh copy for every test).
"""

from pathlib import Path

import pytest

from src.data_loader import load_raw_data
from src.model import predict_churn, train_model
from src.preprocessing import clean, get_features_and_target, split_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

VALID_ROW = {
    "customerID": "A",
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


def _make_row(customer_id: str = "A", **overrides) -> dict:
    return {**VALID_ROW, "customerID": customer_id, **overrides}


@pytest.fixture
def make_row():
    """make_row("B", Churn="Yes") -> a valid raw record with the given changes."""
    return _make_row


@pytest.fixture(scope="session")
def raw_data_path():
    return RAW_DATA_PATH


@pytest.fixture(scope="session")
def _raw_data():
    return load_raw_data(RAW_DATA_PATH)


@pytest.fixture
def raw_df(_raw_data):
    return _raw_data.copy()


@pytest.fixture(scope="session")
def cleaned(_raw_data):
    return clean(_raw_data)


@pytest.fixture(scope="session")
def model_data(cleaned):
    return get_features_and_target(cleaned)


@pytest.fixture(scope="session")
def split(model_data):
    return split_data(model_data)


@pytest.fixture(scope="session")
def trained_model(split):
    return train_model(split.X_train, split.y_train)


@pytest.fixture(scope="session")
def holdout_predictions(trained_model, split):
    return predict_churn(trained_model, split.X_test)
