import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.model import build_model, coefficient_table, format_model_report, predict_churn, train_model
from src.preprocessing import NUMERIC_COLUMNS, FeatureError, clean, feature_columns, get_features_and_target


@pytest.fixture
def small_sample(make_row):
    """20 valid customers: 10 short-tenure month-to-month churners, 10 long-tenure stayers."""
    rows = []
    for i in range(10):
        rows.append(
            make_row(f"C{i:02d}", tenure=1 + i, Contract="Month-to-month", InternetService="Fiber optic",
                     MonthlyCharges=80.0 + i, TotalCharges=(1 + i) * (80.0 + i), Churn="Yes")
        )
        rows.append(
            make_row(f"S{i:02d}", tenure=40 + i, Contract="Two year", PaymentMethod="Mailed check",
                     MonthlyCharges=25.0 + i, TotalCharges=(40 + i) * (25.0 + i), Churn="No")
        )
    return get_features_and_target(clean(pd.DataFrame(rows)))


def test_pipeline_stages():
    model = build_model()

    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["preprocessor", "classifier"]
    assert model.named_steps["classifier"].max_iter == 1000


def test_model_can_be_trained_on_small_valid_sample(small_sample):
    model = train_model(small_sample.X, small_sample.y)

    predicted = predict_churn(model, small_sample.X)["predicted_churn"]
    assert len(predicted) == 20
    assert (predicted == small_sample.y).all()  # clearly separated groups are learned


def test_predictions_contain_only_valid_classes(trained_model, split):
    result = predict_churn(trained_model, split.X_test)

    assert set(result["predicted_churn"].unique()) <= {0, 1}
    assert result["churn_probability"].between(0, 1).all()
    expected = (result["churn_probability"] >= 0.5).astype(int)
    assert (result["predicted_churn"] == expected).all()
    assert result.index.equals(split.X_test.index)


def test_scaler_learns_from_training_data_only(trained_model, split, model_data):
    scaler = trained_model.named_steps["preprocessor"].named_transformers_["scale_numeric"]

    np.testing.assert_allclose(scaler.mean_, split.X_train[NUMERIC_COLUMNS].mean())
    assert not np.allclose(scaler.mean_, model_data.X[NUMERIC_COLUMNS].mean())


def test_model_converges_without_warning(split):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        train_model(split.X_train, split.y_train)


def test_training_is_reproducible(trained_model, split):
    again = train_model(split.X_train, split.y_train)

    np.testing.assert_array_equal(
        again.named_steps["classifier"].coef_, trained_model.named_steps["classifier"].coef_
    )


def test_train_rejects_identifier_in_features(split):
    X = split.X_train.assign(customerID="x")

    with pytest.raises(FeatureError, match="customerID"):
        train_model(X, split.y_train)


def test_coefficients_cover_every_feature(trained_model):
    table = coefficient_table(trained_model)

    assert sorted(table.index) == sorted(feature_columns())
    assert table["coefficient"].is_monotonic_decreasing


def test_model_report_lists_split_and_pipeline(trained_model, split):
    report = format_model_report(split, trained_model)

    assert "test_size=0.2, random_state=42" in report
    assert "Training set: 5,634 customers" in report
    assert "Test set: 1,409 customers" in report
    assert "LogisticRegression" in report
    assert "not causes" in report
