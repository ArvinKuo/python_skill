import pandas as pd
import pytest

from src.preprocessing import (
    NUMERIC_COLUMNS,
    DataCleaningError,
    FeatureError,
    build_preprocessor,
    clean,
    feature_columns,
    get_features_and_target,
    split_data,
    validate_features,
)


# --- Cleaning and encoding ----------------------------------------------------

def test_input_dataframe_is_not_modified(make_row):
    df = pd.DataFrame([make_row("A"), make_row("B", tenure=0, TotalCharges=" ")])
    original = df.copy()

    clean(df)

    pd.testing.assert_frame_equal(df, original)


def test_customer_id_kept_separately_not_in_features(make_row):
    df = pd.DataFrame([make_row("A"), make_row("B")])

    result = clean(df)

    assert "customerID" not in result.features.columns
    assert result.customer_ids.tolist() == ["A", "B"]
    assert result.to_frame().columns[0] == "customerID"


def test_total_charges_converted_to_number(make_row):
    result = clean(pd.DataFrame([make_row("A", TotalCharges="360.5")]))

    assert result.features["TotalCharges"].dtype == float
    assert result.features.loc[0, "TotalCharges"] == 360.5


def test_blank_total_charges_for_new_customer_becomes_zero(make_row):
    df = pd.DataFrame([make_row("A", tenure=0, MonthlyCharges=52.55, TotalCharges=" ")])

    result = clean(df)

    assert result.features.loc[0, "TotalCharges"] == 0.0
    assert any("set to 0" in step for step in result.log)


def test_blank_total_charges_for_existing_customer_is_estimated(make_row):
    df = pd.DataFrame([make_row("A", tenure=10, MonthlyCharges=20.0, TotalCharges="")])

    result = clean(df)

    assert result.features.loc[0, "TotalCharges"] == pytest.approx(200.0)
    assert any("estimated" in step for step in result.log)


def test_non_numeric_total_charges_raises(make_row):
    with pytest.raises(DataCleaningError, match="non-numeric"):
        clean(pd.DataFrame([make_row("A", TotalCharges="abc")]))


def test_missing_monthly_charges_raises(make_row):
    with pytest.raises(DataCleaningError, match="MonthlyCharges"):
        clean(pd.DataFrame([make_row("A", MonthlyCharges=None)]))


def test_churn_mapped_to_binary(make_row):
    df = pd.DataFrame([make_row("A", Churn="Yes"), make_row("B", Churn="No")])

    result = clean(df)

    assert result.target.tolist() == [1, 0]


def test_unexpected_churn_label_raises(make_row):
    with pytest.raises(DataCleaningError, match="Churn"):
        clean(pd.DataFrame([make_row("A", Churn="Maybe")]))


def test_missing_churn_column_allowed_for_prediction_data(make_row):
    result = clean(pd.DataFrame([make_row("A")]).drop(columns=["Churn"]))

    assert result.target is None
    assert "Churn" not in result.to_frame().columns


def test_yes_no_and_one_hot_encoding(make_row):
    df = pd.DataFrame([make_row("A", Partner="Yes", Dependents="No", Contract="Two year")])

    features = clean(df).features

    assert features.loc[0, "Partner"] == 1
    assert features.loc[0, "Dependents"] == 0
    assert features.loc[0, "Contract_Two year"] == 1
    assert features.loc[0, "Contract_Month-to-month"] == 0
    assert features.loc[0, "Contract_One year"] == 0


def test_encoded_columns_are_fixed_even_if_category_absent(make_row):
    one_row = clean(pd.DataFrame([make_row("A")])).features
    other_row = clean(pd.DataFrame([make_row("B", Contract="One year", gender="Male")])).features

    assert list(one_row.columns) == feature_columns()
    assert list(other_row.columns) == feature_columns()


def test_unexpected_category_raises(make_row):
    with pytest.raises(DataCleaningError, match="Contract"):
        clean(pd.DataFrame([make_row("A", Contract="Three year")]))


def test_surrounding_whitespace_is_ignored(make_row):
    df = pd.DataFrame([make_row(" A ", Partner=" Yes ", Contract=" One year", Churn="Yes ")])

    result = clean(df)

    assert result.customer_ids.tolist() == ["A"]
    assert result.features.loc[0, "Partner"] == 1
    assert result.features.loc[0, "Contract_One year"] == 1
    assert result.target.tolist() == [1]


def test_raw_dataset_cleans_completely(raw_df):
    raw = raw_df

    result = clean(raw)

    assert result.features.shape == (7043, len(feature_columns()))
    assert result.features.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in result.features.dtypes)
    assert set(result.target.unique()) == {0, 1}
    assert result.customer_ids.is_unique
    new_customers = raw["tenure"] == 0
    assert (result.features.loc[new_customers, "TotalCharges"] == 0).all()


def test_total_charges_numeric_after_cleaning(raw_df, cleaned):
    assert not pd.api.types.is_numeric_dtype(raw_df["TotalCharges"])  # text before cleaning
    total = cleaned.features["TotalCharges"]
    assert pd.api.types.is_float_dtype(total)
    assert total.notna().all()


def test_cleaned_data_has_no_negative_charges(cleaned):
    for col in NUMERIC_COLUMNS:
        assert (cleaned.features[col] >= 0).all(), col


@pytest.mark.parametrize(
    "overrides, column",
    [
        ({"MonthlyCharges": -5.0}, "MonthlyCharges"),
        ({"TotalCharges": "-10.0"}, "TotalCharges"),
        ({"tenure": -1}, "tenure"),
    ],
)
def test_negative_values_rejected(make_row, overrides, column):
    df = pd.DataFrame([make_row("A"), make_row("B", **overrides)])

    with pytest.raises(DataCleaningError, match=f"Negative values.*{column} \\(1\\)"):
        clean(df)


# --- Features (X) and target (y) -------------------------------------------


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


# --- Train/test split ------------------------------------------------------


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


# --- Scaling transformer ------------------------------------------------------


def test_preprocessor_scales_only_numeric_columns(split):
    preprocessor = build_preprocessor().fit(split.X_train)

    transformed = pd.DataFrame(
        preprocessor.transform(split.X_train), columns=preprocessor.get_feature_names_out()
    )

    assert list(transformed.columns) == feature_columns()
    for col in NUMERIC_COLUMNS:
        assert transformed[col].mean() == pytest.approx(0, abs=1e-9)
        assert transformed[col].std(ddof=0) == pytest.approx(1)
    binary = [c for c in feature_columns() if c not in NUMERIC_COLUMNS]
    assert (transformed[binary].to_numpy() == split.X_train[binary].to_numpy()).all()
