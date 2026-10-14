"""Task 7 (train/test split) and Task 8 (Logistic Regression pipeline).

`split_data()` creates a reproducible, stratified split of X, y and the
customer IDs. `build_model()` returns an unfitted scikit-learn Pipeline:

    preprocessor  ColumnTransformer: StandardScaler on the 3 numeric columns,
                  the 38 already-encoded 0/1 columns pass through unchanged
    classifier    LogisticRegression

Fitting the pipeline learns the scaler's mean/standard deviation from the
training data only; predicting reuses those values, so the test data never
influences any preprocessing step.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.cleaning import NUMERIC_COLUMNS
from src.features import FeatureError, ModelData, validate_features

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ITER = 1000
SCALED_COLUMNS = list(NUMERIC_COLUMNS)


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series
    test_size: float
    random_state: int

    def summary_lines(self) -> list[str]:
        lines = [
            f"Settings: test_size={self.test_size}, random_state={self.random_state}, "
            "stratify=Churn (same churn share in both sets)"
        ]
        for name, y in (("Training set", self.y_train), ("Test set", self.y_test)):
            lines.append(
                f"{name}: {len(y):,} customers, {int(y.sum()):,} churned ({y.mean() * 100:.2f}%)"
            )
        return lines


def split_data(
    data: ModelData, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> DataSplit:
    """Stratified, reproducible split; customer IDs are split together with X and y."""
    if data.y is None:
        raise FeatureError("A target (Churn) is required to split data for training.")
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        data.X,
        data.y,
        data.customer_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=data.y,
    )
    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        ids_train=ids_train,
        ids_test=ids_test,
        test_size=test_size,
        random_state=random_state,
    )


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("scale_numeric", StandardScaler(), SCALED_COLUMNS)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=MAX_ITER)),
        ]
    )


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the pipeline on training data; fail loudly if the optimiser does not converge."""
    validate_features(X_train)
    model = build_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)
    return model


def predict_churn(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Predicted label (1 = churn, threshold 0.5) and churn probability for each row of X."""
    validate_features(X)
    probability = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return pd.DataFrame(
        {"predicted_churn": model.predict(X), "churn_probability": probability},
        index=X.index,
    )


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    """Model weights per feature, sorted from most 'towards churn' to most 'towards staying'."""
    names = model[:-1].get_feature_names_out()
    coef = model.named_steps["classifier"].coef_[0]
    table = pd.DataFrame({"coefficient": coef, "odds_ratio": np.exp(coef)}, index=names)
    table.index.name = "feature"
    return table.sort_values("coefficient", ascending=False)


def format_model_report(split: DataSplit, model: Pipeline, top_n: int = 8) -> str:
    clf = model.named_steps["classifier"]
    scaler = model.named_steps["preprocessor"].named_transformers_["scale_numeric"]
    lines = ["MODEL REPORT", "------------", "Train/test split:"]
    lines += [f"  {line}" for line in split.summary_lines()]
    lines += [
        "",
        "Pipeline:",
        f"  1. preprocessor: StandardScaler on {', '.join(SCALED_COLUMNS)} "
        f"(learned from the training set only); {len(split.X_train.columns) - len(SCALED_COLUMNS)} "
        "encoded 0/1 columns passed through unchanged",
        f"  2. classifier: LogisticRegression(solver={clf.solver}, C={clf.C}, max_iter={clf.max_iter}, "
        f"class_weight={clf.class_weight}) - converged after {int(clf.n_iter_[0])} iterations",
        "",
        "Scaling values learned from the training set:",
    ]
    for col, mean, std in zip(SCALED_COLUMNS, scaler.mean_, scaler.scale_):
        lines.append(f"  {col:<15} mean {mean:>9.2f}   standard deviation {std:>9.2f}")

    coef = coefficient_table(model)
    width = max(len(name) for name in coef.index)
    header = f"  {'Feature'.ljust(width)}  {'Coefficient':>11}  {'Odds ratio':>10}"

    def rows(table: pd.DataFrame) -> list[str]:
        return [
            f"  {name.ljust(width)}  {row.coefficient:>+11.3f}  {row.odds_ratio:>10.2f}"
            for name, row in table.iterrows()
        ]

    lines += [
        "",
        f"Intercept: {clf.intercept_[0]:+.3f}",
        "Coefficients are associations, not causes. Numeric features: change per 1 standard deviation;",
        "0/1 features: change from 0 to 1. Odds ratio > 1 = higher predicted churn odds, < 1 = lower.",
        "",
        f"Top {top_n} towards churn:",
        header,
        *rows(coef.head(top_n)),
        "",
        f"Top {top_n} towards staying:",
        header,
        *rows(coef.tail(top_n).iloc[::-1]),
    ]
    return "\n".join(lines)
