"""Task 9 (model evaluation) and Task 10 (predictions.csv).

`evaluate()` compares predicted and actual churn on the test set.
`build_predictions()` creates one row per test-set customer with the
customer ID, actual and predicted churn, churn probability and the outcome
(TP / TN / FP / FN), in the original test-set order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.validation import ID_COLUMN

PREDICTION_COLUMNS = [ID_COLUMN, "actual_churn", "predicted_churn", "churn_probability", "outcome"]

OUTCOME_LABELS = {
    (1, 1): "TP",  # churned, predicted churn
    (0, 0): "TN",  # stayed, predicted stay
    (0, 1): "FP",  # stayed, predicted churn (false alarm)
    (1, 0): "FN",  # churned, predicted stay (missed churner)
}


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def total(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    @property
    def actual_churners(self) -> int:
        return self.tp + self.fn

    @property
    def predicted_churners(self) -> int:
        return self.tp + self.fp

    @property
    def baseline_accuracy(self) -> float:
        """Accuracy of always predicting the most common class (here: 'stays')."""
        return max(self.actual_churners, self.total - self.actual_churners) / self.total


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> EvaluationResult:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def build_predictions(
    customer_ids: pd.Series, actual: pd.Series, predictions: pd.DataFrame
) -> pd.DataFrame:
    """One row per customer, in the order of `customer_ids` (the test-set order)."""
    if not (customer_ids.index.equals(actual.index) and customer_ids.index.equals(predictions.index)):
        raise ValueError("customer IDs, actual churn and predictions must have the same rows in the same order.")

    table = pd.DataFrame(
        {
            ID_COLUMN: customer_ids.to_numpy(),
            "actual_churn": actual.astype(int).to_numpy(),
            "predicted_churn": predictions["predicted_churn"].astype(int).to_numpy(),
            "churn_probability": predictions["churn_probability"].to_numpy(),
        }
    )
    table["outcome"] = [
        OUTCOME_LABELS[(a, p)] for a, p in zip(table["actual_churn"], table["predicted_churn"])
    ]
    return table[PREDICTION_COLUMNS]


def save_predictions(table: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return path


def format_evaluation_report(result: EvaluationResult) -> str:
    def pct(value: float) -> str:
        return f"{value * 100:.1f}%"

    lines = [
        "MODEL EVALUATION (test set)",
        "---------------------------",
        f"Customers evaluated: {result.total:,} (not used for training)",
        "Decision threshold: churn probability >= 0.5 -> predicted churn",
        "",
        f"{'Metric':<11}  {'Value':>7}",
        f"{'-' * 11}  {'-' * 7}",
        f"{'Accuracy':<11}  {result.accuracy:>7.4f}",
        f"{'Precision':<11}  {result.precision:>7.4f}",
        f"{'Recall':<11}  {result.recall:>7.4f}",
        f"{'F1 Score':<11}  {result.f1:>7.4f}",
        "",
        "Confusion matrix (rows = actual, columns = predicted):",
        f"  {'':<16}  {'Predicted stay':>14}  {'Predicted churn':>15}",
        f"  {'Actually stayed':<16}  {f'TN = {result.tn:,}':>14}  {f'FP = {result.fp:,}':>15}",
        f"  {'Actually churned':<16}  {f'FN = {result.fn:,}':>14}  {f'TP = {result.tp:,}':>15}",
        "",
        "In plain words:",
        f"  Accuracy:  {pct(result.accuracy)} of all {result.total:,} test customers were classified correctly.",
        f"             Always predicting 'stays' would already be right {pct(result.baseline_accuracy)} of the time.",
        f"  Precision: of the {result.predicted_churners:,} customers flagged as churners, "
        f"{result.tp:,} ({pct(result.precision)}) really churned.",
        f"  Recall:    of the {result.actual_churners:,} customers who really churned, "
        f"the model caught {result.tp:,} ({pct(result.recall)}).",
        f"  F1 score:  {result.f1:.4f}, a single score balancing precision and recall.",
        f"  False negatives: {result.fn:,} customers churned but were predicted to stay "
        "- they would get no retention offer.",
        f"  False positives: {result.fp:,} customers were flagged but stayed "
        "- a retention offer would be spent on them unnecessarily.",
    ]
    return "\n".join(lines)
