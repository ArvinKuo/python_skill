"""Command-line entry point: python main.py [options]

Reads the options, then hands over to src.pipeline.run_pipeline().
"""

import argparse
import sys
from pathlib import Path

from src.data_loader import DataLoadError
from src.pipeline import PipelineConfig, PipelineError, run_pipeline
from src.preprocessing import DataCleaningError, FeatureError

EXPECTED_ERRORS = (DataLoadError, DataCleaningError, FeatureError, PipelineError)

# option name -> (PipelineConfig field, help text)
OPTIONS = {
    "--input": ("input_path", "Raw CSV (read only, never changed)"),
    "--output": ("predictions_path", "Predictions CSV for the test-set customers"),
    "--report": ("validation_report_path", "Data validation report"),
    "--processed": ("processed_path", "Cleaned and encoded CSV"),
    "--eda-report": ("eda_report_path", "Exploratory analysis report"),
    "--chart": ("chart_path", "Churn-by-contract chart (PNG)"),
    "--model-report": ("model_report_path", "Split settings, pipeline and coefficients"),
    "--evaluation-report": ("evaluation_report_path", "Test-set metrics and confusion matrix"),
}


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Telco Customer Churn Prediction")
    for option, (field, help_text) in OPTIONS.items():
        default = getattr(defaults, field)
        parser.add_argument(
            option,
            dest=field,
            type=Path,
            default=default,
            metavar="PATH",
            help=f"{help_text} (default: {default.as_posix()})",
        )
    return PipelineConfig(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        run_pipeline(config)
    except EXPECTED_ERRORS as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
