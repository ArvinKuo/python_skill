import pandas as pd
import pytest

from src.cleaning import clean, feature_columns
from src.features import FeatureError, get_features_and_target, validate_features


@pytest.fixture(scope="module")
def cleaned():
    return clean(pd.read_csv("data/raw/Telco-Customer-Churn.csv"))


def test_x_excludes_identifier_and_target(cleaned):
    data = get_features_and_target(cleaned)

    assert "customerID" not in data.X.columns
    assert "Churn" not in data.X.columns
    assert list(data.X.columns) == feature_columns()
    assert data.X.shape == (7043, 41)


def test_y_and_customer_ids_line_up_with_x(cleaned):
    data = get_features_and_target(cleaned)

    assert data.y.tolist() == cleaned.target.tolist()
    assert data.customer_ids.tolist() == cleaned.customer_ids.tolist()
    assert data.X.index.equals(data.y.index)
    assert data.X.index.equals(data.customer_ids.index)


def test_saved_csv_gives_same_result_as_clean_output(cleaned, tmp_path):
    path = tmp_path / "clean.csv"
    cleaned.to_frame().to_csv(path, index=False)

    from_csv = get_features_and_target(pd.read_csv(path))
    from_clean = get_features_and_target(cleaned)

    pd.testing.assert_frame_equal(from_csv.X, from_clean.X, check_dtype=False)
    assert from_csv.y.tolist() == from_clean.y.tolist()
    assert from_csv.customer_ids.tolist() == from_clean.customer_ids.tolist()


@pytest.mark.parametrize("column", ["customerID", "Churn"])
def test_validate_features_rejects_identifier_or_target(cleaned, column):
    X = cleaned.to_frame().drop(columns=[c for c in ["customerID", "Churn"] if c != column])

    with pytest.raises(FeatureError, match=column):
        validate_features(X)


def test_unexpected_extra_column_rejected(cleaned):
    df = cleaned.to_frame().assign(account_number=range(len(cleaned.features)))

    with pytest.raises(FeatureError, match="unexpected column.*account_number"):
        get_features_and_target(df)


def test_missing_feature_column_rejected(cleaned):
    df = cleaned.to_frame().drop(columns=["tenure"])

    with pytest.raises(FeatureError, match="missing column.*tenure"):
        get_features_and_target(df)


def test_reordered_columns_rejected(cleaned):
    X = cleaned.features[list(reversed(feature_columns()))]

    with pytest.raises(FeatureError, match="different order"):
        validate_features(X)


def test_non_numeric_feature_rejected(cleaned):
    X = cleaned.features.astype({"tenure": str})

    with pytest.raises(FeatureError, match="non-numeric.*tenure"):
        validate_features(X)


def test_missing_values_in_features_rejected(cleaned):
    X = cleaned.features.copy()
    X.loc[0, "TotalCharges"] = float("nan")

    with pytest.raises(FeatureError, match="missing values.*TotalCharges"):
        validate_features(X)


@pytest.mark.parametrize("bad_value", [2, float("nan")])
def test_invalid_target_rejected(cleaned, bad_value):
    df = cleaned.to_frame().astype({"Churn": float})
    df.loc[0, "Churn"] = bad_value

    with pytest.raises(FeatureError, match="Churn"):
        get_features_and_target(df)


def test_missing_target_rejected_unless_not_required(cleaned):
    df = cleaned.to_frame().drop(columns=["Churn"])

    with pytest.raises(FeatureError, match="Churn is required"):
        get_features_and_target(df)

    data = get_features_and_target(df, require_target=False)
    assert data.y is None
    assert data.X.shape == (7043, 41)


def test_missing_customer_id_rejected(cleaned):
    with pytest.raises(FeatureError, match="customerID is required"):
        get_features_and_target(cleaned.to_frame().drop(columns=["customerID"]))
