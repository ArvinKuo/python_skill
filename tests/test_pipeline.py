import hashlib

import pandas as pd
import pytest

import main
from src.evaluation import PREDICTION_COLUMNS
from src.pipeline import PipelineConfig, PipelineError, run_pipeline


def _config(tmp_path, raw_data_path, **overrides) -> PipelineConfig:
    paths = {
        "input_path": raw_data_path,
        "processed_path": tmp_path / "processed" / "clean.csv",
        "validation_report_path": tmp_path / "out" / "validation_report.txt",
        "eda_report_path": tmp_path / "out" / "eda_report.txt",
        "chart_path": tmp_path / "out" / "churn_by_contract.png",
        "model_report_path": tmp_path / "out" / "model_report.txt",
        "evaluation_report_path": tmp_path / "out" / "evaluation_report.txt",
        "predictions_path": tmp_path / "out" / "predictions.csv",
    }
    return PipelineConfig(**{**paths, **overrides})


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory, raw_data_path):
    tmp_path = tmp_path_factory.mktemp("pipeline")
    config = _config(tmp_path, raw_data_path)
    raw_hash = _sha256(raw_data_path)
    messages: list[str] = []
    result = run_pipeline(config, log=messages.append)
    return config, result, raw_hash, messages


def test_pipeline_creates_every_output(pipeline_run):
    config, _, _, _ = pipeline_run

    for name, path in config.output_paths().items():
        assert path.is_file(), name
        assert path.stat().st_size > 0, name


def test_pipeline_prediction_file_has_required_columns(pipeline_run):
    config, result, _, _ = pipeline_run

    saved = pd.read_csv(config.predictions_path)

    assert list(saved.columns) == PREDICTION_COLUMNS
    assert len(saved) == len(result.split.X_test) == 1409
    assert saved["customerID"].tolist() == result.split.ids_test.tolist()


def test_pipeline_leaves_raw_csv_unchanged(pipeline_run, raw_data_path):
    _, _, raw_hash, _ = pipeline_run

    assert _sha256(raw_data_path) == raw_hash


def test_pipeline_model_beats_baseline(pipeline_run):
    _, result, _, _ = pipeline_run

    assert result.evaluation.accuracy > result.evaluation.baseline_accuracy


def test_pipeline_logs_each_stage(pipeline_run):
    _, _, _, messages = pipeline_run
    text = "\n".join(messages)

    for heading in [
        "DATA VALIDATION REPORT",
        "DATA CLEANING SUMMARY",
        "EXPLORATORY ANALYSIS REPORT",
        "MODEL INPUT",
        "MODEL REPORT",
        "MODEL EVALUATION",
        "Predictions saved to",
    ]:
        assert heading in text


@pytest.mark.parametrize("field", ["predictions_path", "processed_path", "validation_report_path"])
def test_output_may_not_overwrite_raw_csv(tmp_path, raw_data_path, field):
    config = _config(tmp_path, raw_data_path, **{field: raw_data_path})
    before = _sha256(raw_data_path)

    with pytest.raises(PipelineError, match="raw CSV must stay unchanged"):
        run_pipeline(config, log=lambda _: None)

    assert _sha256(raw_data_path) == before


def test_two_outputs_may_not_share_a_path(tmp_path, raw_data_path):
    same = tmp_path / "out" / "report.txt"
    config = _config(tmp_path, raw_data_path, eda_report_path=same, model_report_path=same)

    with pytest.raises(PipelineError, match="same file"):
        config.check()


def test_main_reports_missing_input_as_error(tmp_path, capsys):
    exit_code = main.main(["--input", str(tmp_path / "missing.csv")])

    assert exit_code == 1
    assert "Input file not found" in capsys.readouterr().err


def test_main_options_map_to_config(tmp_path):
    config = main.parse_args(["--output", str(tmp_path / "p.csv"), "--chart", str(tmp_path / "c.png")])

    assert config.predictions_path == tmp_path / "p.csv"
    assert config.chart_path == tmp_path / "c.png"
    assert config.input_path == PipelineConfig().input_path
