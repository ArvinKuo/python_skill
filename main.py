import argparse
from pathlib import Path

import pandas as pd

from src.cleaning import clean
from src.eda import analyze, format_eda_report, plot_churn_by_contract
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
    args = parser.parse_args()

    input_path = Path(args.input)
    processed_path = Path(args.processed)
    if processed_path.resolve() == input_path.resolve():
        parser.error("--processed must be different from --input; the raw CSV must stay unchanged.")

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

    # TODO: train -> evaluate -> export predictions
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
