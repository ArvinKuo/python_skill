import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from src.cleaning import NUMERIC_COLUMNS, clean, feature_columns
from src.features import FeatureError, get_features_and_target
from src.model import (
    build_model,
    coefficient_table,
    format_model_report,
    predict_churn,
    split_data,
    train_model,
)


@pytest.fixture(scope="module")
def model_data():
    return get_features_and_target(clean(pd.read_csv("data/raw/Telco-Customer-Churn.csv")))


@pytest.fixture(scope="module")
def split(model_data):
    return split_data(model_data)


@pytest.fixture(scope="module")
def model(split):
    return train_model(split.X_train, split.y_train)


# --- Task 7: split -----------------------------------------------------------


def test_split_sizes_are_80_20(split):
    assert len(split.X_train) == 5634
    assert len(split.X_test) == 1409


def test_split_is_stratified(split, model_data):
    overall = model_data.y.mean()
    assert split.y_train.mean() == pytest.approx(overall, abs=0.001)
    assert split.y_test.mean() == pytest.approx(overall, abs=0.001)


def test_split_is_reproducible(model_data, split):
    again = split_data(model_data)

    assert again.X_test.index.equals(split.X_test.index)
    assert again.X_train.index.equals(split.X_train.index)


def test_different_seed_gives_different_split(model_data, split):
    other = split_data(model_data, random_state=7)

    assert not other.X_test.index.equals(split.X_test.index)


def test_train_and_test_do_not_overlap(split):
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert set(split.ids_train).isdisjoint(split.ids_test)


def test_ids_stay_aligned_with_rows(split, model_data):
    for X, y, ids in (
        (split.X_train, split.y_train, split.ids_train),
        (split.X_test, split.y_test, split.ids_test),
    ):
        assert X.index.equals(y.index)
        assert X.index.equals(ids.index)
        assert ids.equals(model_data.customer_ids.loc[X.index])


def test_split_requires_target(model_data):
    no_target = type(model_data)(X=model_data.X, y=None, customer_ids=model_data.customer_ids)

    with pytest.raises(FeatureError, match="target"):
        split_data(no_target)


# --- Task 8: pipeline --------------------------------------------------------


def test_pipeline_stages():
    model = build_model()

    assert isinstance(model, Pipeline)
    assert [name for name, _ in model.steps] == ["preprocessor", "classifier"]
    assert model.named_steps["classifier"].max_iter == 1000


def test_scaler_learns_from_training_data_only(model, split, model_data):
    scaler = model.named_steps["preprocessor"].named_transformers_["scale_numeric"]

    np.testing.assert_allclose(scaler.mean_, split.X_train[NUMERIC_COLUMNS].mean())
    assert not np.allclose(scaler.mean_, model_data.X[NUMERIC_COLUMNS].mean())


def test_only_numeric_columns_are_scaled(model, split):
    transformed = model[:-1].transform(split.X_test)
    names = list(model[:-1].get_feature_names_out())

    assert names == feature_columns()
    binary = [c for c in names if c not in NUMERIC_COLUMNS]
    passthrough = pd.DataFrame(transformed, columns=names)[binary]
    np.testing.assert_array_equal(passthrough.to_numpy(), split.X_test[binary].to_numpy())


def test_model_converges_without_warning(split):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        train_model(split.X_train, split.y_train)


def test_training_is_reproducible(model, split):
    again = train_model(split.X_train, split.y_train)

    np.testing.assert_array_equal(
        again.named_steps["classifier"].coef_, model.named_steps["classifier"].coef_
    )


def test_train_rejects_identifier_in_features(split):
    X = split.X_train.assign(customerID="x")

    with pytest.raises(FeatureError, match="customerID"):
        train_model(X, split.y_train)


def test_predictions_have_label_and_probability(model, split):
    result = predict_churn(model, split.X_test)

    assert result.index.equals(split.X_test.index)
    assert set(result["predicted_churn"].unique()) <= {0, 1}
    assert result["churn_probability"].between(0, 1).all()
    expected = (result["churn_probability"] >= 0.5).astype(int)
    assert (result["predicted_churn"] == expected).all()


def test_coefficients_cover_every_feature(model):
    table = coefficient_table(model)

    assert sorted(table.index) == sorted(feature_columns())
    assert table["coefficient"].is_monotonic_decreasing


def test_model_report_lists_split_and_pipeline(model, split):
    report = format_model_report(split, model)

    assert "test_size=0.2, random_state=42" in report
    assert "Training set: 5,634 customers" in report
    assert "Test set: 1,409 customers" in report
    assert "LogisticRegression" in report
    assert "not causes" in report
