"""Exploratory analysis of churn for the Telco Customer Churn dataset.

`analyze()` computes the churn tables, `format_eda_report()` turns them into a
plain-language text report and `plot_churn_by_contract()` saves a bar chart.
All results describe associations in this dataset, not causes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.data_loader import TARGET_COLUMN
from src.preprocessing import encode_target

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
TENURE_BINS = [0, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
GROUP_COLUMNS = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]


@dataclass
class EdaResult:
    customers: int
    churned: int
    churn_rate: float
    by_group: dict[str, pd.DataFrame]
    by_tenure: pd.DataFrame
    monthly_charges: pd.DataFrame


def churn_rate_by(groups: pd.Series, churn: pd.Series) -> pd.DataFrame:
    """Customers, churned customers and churn rate (0-1) for each group."""
    table = churn.groupby(groups, observed=True).agg(customers="size", churned="sum", churn_rate="mean")
    table.index.name = groups.name
    return table


def tenure_bands(tenure: pd.Series) -> pd.Series:
    return pd.cut(tenure, bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True).rename("TenureBand")


def analyze(df: pd.DataFrame) -> EdaResult:
    churn = encode_target(df[TARGET_COLUMN])

    by_group = {}
    for col in GROUP_COLUMNS:
        table = churn_rate_by(df[col], churn)
        if col == "Contract":
            table = table.reindex([c for c in CONTRACT_ORDER if c in table.index])
        else:
            table = table.sort_values("churn_rate", ascending=False)
        by_group[col] = table

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    by_tenure = churn_rate_by(tenure_bands(tenure), churn)

    charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    status = churn.map({1: "Churned", 0: "Stayed"}).rename("Customer group")
    monthly = charges.groupby(status).agg(customers="size", mean="mean", median="median")
    monthly = monthly.reindex(["Churned", "Stayed"])

    return EdaResult(
        customers=len(churn),
        churned=int(churn.sum()),
        churn_rate=float(churn.mean()),
        by_group=by_group,
        by_tenure=by_tenure,
        monthly_charges=monthly,
    )


def _pct(rate: float) -> str:
    return f"{rate * 100:.1f}%"


def _format_rate_table(table: pd.DataFrame) -> list[str]:
    width = max(len(str(label)) for label in table.index)
    lines = [f"  {'Group'.ljust(width)}  {'Customers':>9}  {'Churned':>7}  {'Churn rate':>10}"]
    for label, row in table.iterrows():
        lines.append(
            f"  {str(label).ljust(width)}  {int(row.customers):>9,}  {int(row.churned):>7,}  "
            f"{_pct(row.churn_rate):>10}"
        )
    return lines


def _highest_lowest(table: pd.DataFrame) -> str:
    high = table["churn_rate"].idxmax()
    low = table["churn_rate"].idxmin()
    high_rate = table.loc[high, "churn_rate"]
    low_rate = table.loc[low, "churn_rate"]
    gap = f"a gap of {(high_rate - low_rate) * 100:.1f} percentage points"
    if low_rate > 0:
        gap += f" (about {high_rate / low_rate:.1f}x)"
    return f"  Highest: {high} ({_pct(high_rate)}); lowest: {low} ({_pct(low_rate)}); {gap}."


def format_eda_report(result: EdaResult, source: str | None = None) -> str:
    lines = ["EXPLORATORY ANALYSIS REPORT", "---------------------------"]
    if source:
        lines.append(f"Dataset: {source}")
    lines.append(
        "Churn rate = churned customers / customers in the group. "
        "These figures show associations, not causes."
    )

    lines += [
        "",
        "1. What percentage of customers churned?",
        f"  {result.churned:,} of {result.customers:,} customers churned = {_pct(result.churn_rate)}",
    ]

    questions = {
        "Contract": "2. How does churn differ by Contract type?",
        "InternetService": "3. How does churn differ by InternetService?",
        "PaymentMethod": "5. How does churn differ by PaymentMethod?",
        "TechSupport": "7. Is there a visible relationship between TechSupport and churn?",
    }
    sections = {
        col: [question, *_format_rate_table(result.by_group[col]), _highest_lowest(result.by_group[col])]
        for col, question in questions.items()
    }
    sections["tenure"] = [
        "4. How does churn differ for different tenure ranges?",
        *_format_rate_table(result.by_tenure),
        _highest_lowest(result.by_tenure),
    ]

    mc = result.monthly_charges
    diff = float(mc.loc["Churned", "mean"]) - float(mc.loc["Stayed", "mean"])
    direction = "more" if diff >= 0 else "less"
    sections["monthly"] = [
        "6. How do average MonthlyCharges compare between churned and non-churned customers?",
        f"  {'Group':<8}  {'Customers':>9}  {'Average':>8}  {'Median':>8}",
        *[
            f"  {label:<8}  {int(row.customers):>9,}  {row['mean']:>8.2f}  {row['median']:>8.2f}"
            for label, row in mc.iterrows()
        ],
        f"  On average, churned customers were charged {abs(diff):.2f} {direction} per month "
        "than customers who stayed.",
    ]

    for key in ["Contract", "InternetService", "tenure", "PaymentMethod", "monthly", "TechSupport"]:
        lines.append("")
        lines += sections[key]

    return "\n".join(lines)


def plot_churn_by_contract(result: EdaResult, path: str | Path) -> Path:
    table = result.by_group["Contract"]
    rates = table["churn_rate"] * 100

    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    bars = ax.bar(table.index.astype(str), rates, color=["#d62728", "#ff7f0e", "#2ca02c"][: len(table)])
    ax.bar_label(
        bars,
        labels=[f"{r:.1f}%\n(n={int(n):,})" for r, n in zip(rates, table["customers"])],
        padding=3,
    )
    ax.axhline(result.churn_rate * 100, color="gray", linestyle="--", linewidth=1)
    ax.text(
        len(table) - 0.5,
        result.churn_rate * 100,
        f"All customers: {_pct(result.churn_rate)}",
        ha="right",
        va="bottom",
        color="gray",
        fontsize=9,
    )
    ax.set_title("Churn rate by contract type")
    ax.set_xlabel("Contract type")
    ax.set_ylabel("Churn rate (%)")
    ax.set_ylim(0, max(rates.max() * 1.25, result.churn_rate * 100 * 1.25))
    ax.spines[["top", "right"]].set_visible(False)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
