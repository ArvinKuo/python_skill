import pandas as pd
import pytest

from src.eda import analyze, churn_rate_by, format_eda_report, plot_churn_by_contract, tenure_bands


def _df(rows: list[dict]) -> pd.DataFrame:
    base = {
        "Contract": "Month-to-month",
        "InternetService": "DSL",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "tenure": 5,
        "MonthlyCharges": 50.0,
        "Churn": "No",
    }
    return pd.DataFrame([{**base, **row} for row in rows])


@pytest.fixture
def small_df() -> pd.DataFrame:
    return _df(
        [
            {"Contract": "Month-to-month", "Churn": "Yes", "MonthlyCharges": 80.0, "tenure": 0},
            {"Contract": "Month-to-month", "Churn": "No", "MonthlyCharges": 40.0, "tenure": 12},
            {"Contract": "One year", "Churn": "No", "MonthlyCharges": 60.0, "tenure": 13},
            {"Contract": "Two year", "Churn": "No", "MonthlyCharges": 20.0, "tenure": 72},
        ]
    )


def test_overall_churn_rate(small_df):
    result = analyze(small_df)

    assert (result.customers, result.churned) == (4, 1)
    assert result.churn_rate == pytest.approx(0.25)


def test_churn_rate_by_group():
    table = churn_rate_by(pd.Series(["A", "A", "B"], name="g"), pd.Series([1, 0, 0]))

    assert table.loc["A"].tolist() == [2, 1, 0.5]
    assert table.loc["B"].tolist() == [1, 0, 0.0]


def test_contract_table_uses_natural_order(small_df):
    table = analyze(small_df).by_group["Contract"]

    assert table.index.tolist() == ["Month-to-month", "One year", "Two year"]
    assert table.loc["Month-to-month", "churn_rate"] == pytest.approx(0.5)


def test_tenure_band_boundaries():
    bands = tenure_bands(pd.Series([0, 12, 13, 24, 25, 48, 49, 72]))

    assert bands.astype(str).tolist() == [
        "0-12 months",
        "0-12 months",
        "13-24 months",
        "13-24 months",
        "25-48 months",
        "25-48 months",
        "49+ months",
        "49+ months",
    ]


def test_monthly_charges_by_churn(small_df):
    mc = analyze(small_df).monthly_charges

    assert mc.loc["Churned", "mean"] == pytest.approx(80.0)
    assert mc.loc["Stayed", "mean"] == pytest.approx(40.0)


def test_invalid_churn_label_raises(small_df):
    small_df.loc[0, "Churn"] = "Maybe"

    with pytest.raises(ValueError, match="Churn"):
        analyze(small_df)


def test_report_answers_all_questions(small_df):
    report = format_eda_report(analyze(small_df))

    for number in range(1, 8):
        assert f"\n{number}. " in report
    assert "1 of 4 customers churned = 25.0%" in report
    assert "not causes" in report
    assert "lowest: One year (0.0%); a gap of 50.0 percentage points." in report


def test_chart_is_saved(small_df, tmp_path):
    path = plot_churn_by_contract(analyze(small_df), tmp_path / "charts" / "contract.png")

    assert path.exists()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_raw_dataset_key_figures():
    result = analyze(pd.read_csv("data/raw/Telco-Customer-Churn.csv"))

    assert (result.customers, result.churned) == (7043, 1869)
    assert result.churn_rate == pytest.approx(0.2654, abs=1e-4)
    contract = result.by_group["Contract"]["churn_rate"]
    assert contract["Month-to-month"] > contract["One year"] > contract["Two year"]
    assert result.by_tenure["customers"].sum() == 7043
