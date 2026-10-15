"""Run the churn workflow end to end.

load -> validate -> clean -> explore -> X/y -> split -> train -> evaluate -> predictions

`run_pipeline()` is what `python main.py` calls. It only coordinates: each
step's logic lives in its own module (data_loader, validator, preprocessing,
eda, model, evaluation). All output paths come from `PipelineConfig`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_loader import load_raw_data
from src.eda import EdaResult, analyze, format_eda_report, plot_churn_by_contract
from src.evaluation import (
    EvaluationResult,
    build_predictions,
    evaluate,
    format_evaluation_report,
    save_predictions,
)
from src.model import format_model_report, predict_churn, train_model
from src.preprocessing import CleanedData, DataSplit, clean, get_features_and_target, split_data
from src.validator import ValidationResult, format_report, validate


class PipelineError(ValueError):
    """Raised when the pipeline configuration is unsafe, e.g. it would overwrite the raw CSV."""


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path = Path("data/raw/Telco-Customer-Churn.csv")
    processed_path: Path = Path("data/processed/telco_churn_clean.csv")
    validation_report_path: Path = Path("output/validation_report.txt")
    eda_report_path: Path = Path("output/eda_report.txt")
    chart_path: Path = Path("output/churn_by_contract.png")
    model_report_path: Path = Path("output/model_report.txt")
    evaluation_report_path: Path = Path("output/evaluation_report.txt")
    predictions_path: Path = Path("output/predictions.csv")

    def output_paths(self) -> dict[str, Path]:
        return {f.name: Path(getattr(self, f.name)) for f in fields(self) if f.name != "input_path"}

    def check(self) -> None:
        """Refuse to run if any output would overwrite the input or another output."""
        source = Path(self.input_path).resolve()
        seen: dict[Path, str] = {}
        for name, path in self.output_paths().items():
            resolved = path.resolve()
            if resolved == source:
                raise PipelineError(
                    f"{name} ({path}) is the input file; the raw CSV must stay unchanged."
                )
            if resolved in seen:
                raise PipelineError(f"{name} and {seen[resolved]} point to the same file ({path}).")
            seen[resolved] = name


@dataclass
class PipelineResult:
    validation: ValidationResult
    cleaned: CleanedData
    eda: EdaResult
    split: DataSplit
    model: Pipeline
    evaluation: EvaluationResult
    predictions: pd.DataFrame


def _write_text(path: Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _section(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def run_pipeline(
    config: PipelineConfig | None = None, log: Callable[[str], None] = print
) -> PipelineResult:
    config = config or PipelineConfig()
    config.check()
    source = Path(config.input_path).as_posix()

    df = load_raw_data(config.input_path)

    validation = validate(df)
    report = format_report(validation, source=source)
    log(report)
    path = _write_text(config.validation_report_path, report)
    log(f"\nValidation report saved to: {path}")

    cleaned = clean(df)
    log(_section("DATA CLEANING SUMMARY"))
    for step in cleaned.log:
        log(f"- {step}")
    processed_path = Path(config.processed_path)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_frame().to_csv(processed_path, index=False)
    log(f"\nCleaned data saved to: {processed_path}")

    eda = analyze(df)
    eda_report = format_eda_report(eda, source=source)
    log(f"\n{eda_report}")
    path = _write_text(config.eda_report_path, eda_report)
    chart_path = plot_churn_by_contract(eda, config.chart_path)
    log(f"\nExploratory analysis report saved to: {path}")
    log(f"Chart saved to: {chart_path}")

    model_data = get_features_and_target(cleaned)
    log(_section("MODEL INPUT (features and target)"))
    for line in model_data.summary_lines():
        log(f"- {line}")

    split = split_data(model_data)
    model = train_model(split.X_train, split.y_train)
    model_report = format_model_report(split, model)
    log(f"\n{model_report}")
    path = _write_text(config.model_report_path, model_report)
    log(f"\nModel report saved to: {path}")

    test_predictions = predict_churn(model, split.X_test)
    evaluation = evaluate(split.y_test, test_predictions["predicted_churn"])
    evaluation_report = format_evaluation_report(evaluation)
    log(f"\n{evaluation_report}")
    path = _write_text(config.evaluation_report_path, evaluation_report)
    log(f"\nEvaluation report saved to: {path}")

    predictions = build_predictions(split.ids_test, split.y_test, test_predictions)
    path = save_predictions(predictions, config.predictions_path)
    counts = predictions["outcome"].value_counts()
    log(
        f"Predictions saved to: {path} ({len(predictions):,} test-set customers; "
        + ", ".join(f"{label} {int(counts.get(label, 0)):,}" for label in ["TP", "TN", "FP", "FN"])
        + ")"
    )

    return PipelineResult(
        validation=validation,
        cleaned=cleaned,
        eda=eda,
        split=split,
        model=model,
        evaluation=evaluation,
        predictions=predictions,
    )
