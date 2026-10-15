import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS, DataLoadError, load_raw_data
from src.validator import EXPECTED_TYPES


def test_raw_dataset_has_all_required_columns(raw_df):
    assert [c for c in REQUIRED_COLUMNS if c not in raw_df.columns] == []
    assert raw_df.shape == (7043, 21)


def test_required_columns_match_validator_schema():
    assert set(REQUIRED_COLUMNS) == set(EXPECTED_TYPES)


def test_loading_does_not_convert_values(raw_df):
    # TotalCharges stays text so its blank values can be investigated and handled in cleaning.
    assert not pd.api.types.is_numeric_dtype(raw_df["TotalCharges"])
    assert (raw_df["TotalCharges"].str.strip() == "").sum() == 11


def test_missing_required_column_rejected(tmp_path, raw_df):
    path = tmp_path / "no_churn.csv"
    raw_df.drop(columns=["Churn"]).head(5).to_csv(path, index=False)

    with pytest.raises(DataLoadError, match="missing required column.*Churn"):
        load_raw_data(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(DataLoadError, match="not found"):
        load_raw_data(tmp_path / "does_not_exist.csv")


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(DataLoadError, match="empty"):
        load_raw_data(path)


def test_header_only_file_rejected(tmp_path):
    path = tmp_path / "header_only.csv"
    path.write_text(",".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")

    with pytest.raises(DataLoadError, match="no data rows"):
        load_raw_data(path)


def test_loading_leaves_raw_file_unchanged(raw_data_path):
    before = raw_data_path.read_bytes()

    load_raw_data(raw_data_path)

    assert raw_data_path.read_bytes() == before


def test_valid_small_file_loads(tmp_path, make_row):
    path = tmp_path / "small.csv"
    pd.DataFrame([make_row("A"), make_row("B", Churn="Yes")]).to_csv(path, index=False)

    df = load_raw_data(path)

    assert df["customerID"].tolist() == ["A", "B"]
