import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.cleaning import clean
from src.evaluation import (
    PREDICTION_COLUMNS,
    build_predictions,
    evaluate,
    format_evaluation_report,
    save_predictions,
)
from src.features import get_features_and_target
from src.model import predict_churn, split_data, train_model


@pytest.fixture(scope="module")
def split():
    return split_data(
        get_features_and_target(clean(pd.read_csv("data/raw/Telco-Customer-Churn.csv")))
    )


@pytest.fixture(scope="module")
def test_predictions(split):
    model = train_model(split.X_train, split.y_train)
    return predict_churn(model, split.X_test)


@pytest.fixture(scope="module")
def predictions(split, test_predictions):
    return build_predictions(split.ids_test, split.y_test, test_predictions)


# --- Task 9: evaluation ------------------------------------------------------


def test_metrics_on_known_example():
    # 2 TP, 1 FN, 1 FP, 4 TN
    y_true = pd.Series([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = pd.Series([1, 1, 0, 1, 0, 0, 0, 0])

    result = evaluate(y_true, y_pred)

    assert (result.tp, result.fn, result.fp, result.tn) == (2, 1, 1, 4)
    assert result.accuracy == pytest.approx(6 / 8)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.baseline_accuracy == pytest.approx(5 / 8)


def test_no_predicted_churners_does_not_crash():
    result = evaluate(pd.Series([1, 0, 0]), pd.Series([0, 0, 0]))

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.fn == 1


def test_metrics_follow_from_confusion_matrix(split, test_predictions):
    result = evaluate(split.y_test, test_predictions["predicted_churn"])

    assert result.total == len(split.y_test) == 1409
    assert result.actual_churners == int(split.y_test.sum()) == 374
    assert result.accuracy == pytest.approx((result.tp + result.tn) / result.total)
    assert result.precision == pytest.approx(result.tp / (result.tp + result.fp))
    assert result.recall == pytest.approx(result.tp / (result.tp + result.fn))
    assert result.f1 == pytest.approx(f1_score(split.y_test, test_predictions["predicted_churn"]))


def test_model_beats_always_stay_baseline(split, test_predictions):
    result = evaluate(split.y_test, test_predictions["predicted_churn"])

    assert result.accuracy > result.baseline_accuracy
    assert result.recall > 0


def test_report_contains_table_and_confusion_matrix(split, test_predictions):
    report = format_evaluation_report(evaluate(split.y_test, test_predictions["predicted_churn"]))

    for label in ["Accuracy", "Precision", "Recall", "F1 Score", "TN =", "FP =", "FN =", "TP ="]:
        assert label in report
    assert "False negatives" in report


# --- Task 10: predictions.csv -------------------------------------------------


def test_predictions_have_required_columns(predictions):
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert len(predictions) == 1409
    assert predictions["customerID"].is_unique


def test_predictions_keep_test_set_order(predictions, split):
    assert predictions["customerID"].tolist() == split.ids_test.tolist()
    assert predictions["actual_churn"].tolist() == split.y_test.tolist()


def test_predictions_contain_only_test_customers(predictions, split):
    assert set(predictions["customerID"]).isdisjoint(split.ids_train)


def test_predicted_class_matches_probability(predictions):
    assert predictions["churn_probability"].between(0, 1).all()
    expected = (predictions["churn_probability"] >= 0.5).astype(int)
    assert (predictions["predicted_churn"] == expected).all()


def test_outcome_labels(predictions):
    expected = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}
    for (actual, predicted), label in expected.items():
        rows = predictions[
            (predictions["actual_churn"] == actual) & (predictions["predicted_churn"] == predicted)
        ]
        assert (rows["outcome"] == label).all()


def test_outcome_counts_match_confusion_matrix(predictions, split, test_predictions):
    result = evaluate(split.y_test, test_predictions["predicted_churn"])
    counts = predictions["outcome"].value_counts()

    assert counts.get("TP", 0) == result.tp
    assert counts.get("TN", 0) == result.tn
    assert counts.get("FP", 0) == result.fp
    assert counts.get("FN", 0) == result.fn


def test_misaligned_inputs_rejected(split, test_predictions):
    with pytest.raises(ValueError, match="same rows"):
        build_predictions(split.ids_test, split.y_test, test_predictions.iloc[::-1])


def test_saved_csv_round_trips(predictions, tmp_path):
    path = save_predictions(predictions, tmp_path / "out" / "predictions.csv")
    loaded = pd.read_csv(path)

    assert list(loaded.columns) == PREDICTION_COLUMNS
    assert loaded["customerID"].tolist() == predictions["customerID"].tolist()
    assert loaded["outcome"].tolist() == predictions["outcome"].tolist()
