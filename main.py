import argparse
from pathlib import Path

import pandas as pd

from src.cleaning import clean
from src.eda import analyze, format_eda_report, plot_churn_by_contract
from src.evaluation import build_predictions, evaluate, format_evaluation_report, save_predictions
from src.features import get_features_and_target
from src.model import format_model_report, predict_churn, split_data, train_model
from src.validation import format_report, validate


def main():
    parser = argparse.ArgumentParser(description="Telco Customer Churn Prediction")
    parser.add_argument("--input", default="data/raw/Telco-Customer-Churn.csv", help="Path to raw CSV")
    parser.add_argument("--output", default="output/predictions.csv", help="Path to predictions CSV")
    parser.add_argument(
        "--report", default="output/validation_report.txt", help="Path to data validation report"
    )
    parser.add_argument(
        "--processed",
        default="data/processed/telco_churn_clean.csv",
        help="Path to the cleaned and encoded CSV (the raw CSV is never overwritten)",
    )
    parser.add_argument(
        "--eda-report", default="output/eda_report.txt", help="Path to exploratory analysis report"
    )
    parser.add_argument(
        "--chart", default="output/churn_by_contract.png", help="Path to churn-by-contract chart"
    )
    parser.add_argument(
        "--model-report", default="output/model_report.txt", help="Path to model report"
    )
    parser.add_argument(
        "--evaluation-report",
        default="output/evaluation_report.txt",
        help="Path to test-set evaluation report",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    processed_path = Path(args.processed)
    output_path = Path(args.output)
    for name, path in (("--processed", processed_path), ("--output", output_path)):
        if path.resolve() == input_path.resolve():
            parser.error(f"{name} must be different from --input; the raw CSV must stay unchanged.")

    df = pd.read_csv(input_path)

    report = format_report(validate(df), source=args.input)
    print(report)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nValidation report saved to: {report_path}")

    cleaned = clean(df)
    print("\nDATA CLEANING SUMMARY")
    print("---------------------")
    for step in cleaned.log:
        print(f"- {step}")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_frame().to_csv(processed_path, index=False)
    print(f"\nCleaned data saved to: {processed_path}")

    eda = analyze(df)
    eda_report = format_eda_report(eda, source=args.input)
    print(f"\n{eda_report}")
    eda_report_path = Path(args.eda_report)
    eda_report_path.parent.mkdir(parents=True, exist_ok=True)
    eda_report_path.write_text(eda_report + "\n", encoding="utf-8")
    chart_path = plot_churn_by_contract(eda, args.chart)
    print(f"\nExploratory analysis report saved to: {eda_report_path}")
    print(f"Chart saved to: {chart_path}")

    model_data = get_features_and_target(cleaned)
    print("\nMODEL INPUT (features and target)")
    print("---------------------------------")
    for line in model_data.summary_lines():
        print(f"- {line}")

    split = split_data(model_data)
    model = train_model(split.X_train, split.y_train)
    model_report = format_model_report(split, model)
    print(f"\n{model_report}")
    model_report_path = Path(args.model_report)
    model_report_path.parent.mkdir(parents=True, exist_ok=True)
    model_report_path.write_text(model_report + "\n", encoding="utf-8")
    print(f"\nModel report saved to: {model_report_path}")

    test_predictions = predict_churn(model, split.X_test)

    evaluation = evaluate(split.y_test, test_predictions["predicted_churn"])
    evaluation_report = format_evaluation_report(evaluation)
    print(f"\n{evaluation_report}")
    evaluation_report_path = Path(args.evaluation_report)
    evaluation_report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_report_path.write_text(evaluation_report + "\n", encoding="utf-8")
    print(f"\nEvaluation report saved to: {evaluation_report_path}")

    predictions = build_predictions(split.ids_test, split.y_test, test_predictions)
    predictions_path = save_predictions(predictions, output_path)
    counts = predictions["outcome"].value_counts()
    print(
        f"Predictions saved to: {predictions_path} ({len(predictions):,} test-set customers; "
        + ", ".join(f"{label} {int(counts.get(label, 0)):,}" for label in ["TP", "TN", "FP", "FN"])
        + ")"
    )


if __name__ == "__main__":
    main()
