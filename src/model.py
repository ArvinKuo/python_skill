"""Build, train and use the churn classifier.

`build_model()` returns an unfitted scikit-learn Pipeline:

    preprocessor  preprocessing.build_preprocessor(): StandardScaler on the 3
                  numeric columns; the 38 encoded 0/1 columns pass through
    classifier    LogisticRegression

Fitting the pipeline learns the scaler's mean/standard deviation from the
training data only; predicting reuses those values, so the test data never
influences any preprocessing step.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import NUMERIC_COLUMNS, DataSplit, build_preprocessor, validate_features

MAX_ITER = 1000


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
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
        f"  1. preprocessor: StandardScaler on {', '.join(NUMERIC_COLUMNS)} "
        f"(learned from the training set only); {len(split.X_train.columns) - len(NUMERIC_COLUMNS)} "
        "encoded 0/1 columns passed through unchanged",
        f"  2. classifier: LogisticRegression(solver={clf.solver}, C={clf.C}, max_iter={clf.max_iter}, "
        f"class_weight={clf.class_weight}) - converged after {int(clf.n_iter_[0])} iterations",
        "",
        "Scaling values learned from the training set:",
    ]
    for col, mean, std in zip(NUMERIC_COLUMNS, scaler.mean_, scaler.scale_):
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
