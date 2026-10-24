"""Exploratory Data Analysis (EDA) and Visualization Generator.

This module analyzes customer churn patterns across key dimensions:
- Contract type (Month-to-month vs 1-year vs 2-year)
- Tenure cohorts (New customers vs long-term loyalists)
- Service subscriptions (Internet Service, Tech Support, Online Security)
- Billing & Payment methods (Electronic check vs Automatic payments)
- Financial metrics (MonthlyCharges and TotalCharges)

Generates publication-quality figures saved to reports/figures/ and produces
the summary statistics needed for docs/eda_findings.md.
"""

import logging
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def setup_plotting_theme():
    """Configure modern, aesthetic styling for all matplotlib and seaborn charts."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "figure.titlesize": 14,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 200,
        }
    )


def load_cleaned_data(filepath: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    """Load the preprocessed dataset."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {filepath}. Run clean_data.py first."
        )
    return pd.read_csv(filepath)


def plot_overall_churn(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot overall churn distribution (pie / bar)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    churn_counts = df["Churn"].value_counts()
    churn_pcts = df["Churn"].value_counts(normalize=True) * 100

    colors = ["#2b5c8f", "#d9534f"]
    bars = ax.bar(
        ["Retained (No)", "Churned (Yes)"],
        churn_counts,
        color=colors,
        width=0.5,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, pct in zip(bars, churn_pcts):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 80,
            f"{yval:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_title("Overall Customer Churn Distribution", pad=15, fontweight="bold")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, 6000)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    out_file = output_dir / "01_overall_churn_distribution.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file.name)
    return out_file


def plot_churn_by_contract(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot churn rate across contract types."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    contract_summary = (
        df.groupby("Contract")["churn_label"]
        .agg(["count", "mean"])
        .reset_index()
    )
    contract_summary["churn_pct"] = contract_summary["mean"] * 100
    contract_summary = contract_summary.sort_values("churn_pct", ascending=False)

    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
    bars = ax.bar(
        contract_summary["Contract"],
        contract_summary["churn_pct"],
        color=colors,
        width=0.5,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, count in zip(bars, contract_summary["count"]):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.0,
            f"{yval:.1f}%\n(N={count:,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title("Churn Rate by Contract Type", pad=15, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract Type")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    out_file = output_dir / "02_churn_by_contract.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file.name)
    return out_file


def plot_churn_by_tenure_cohorts(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot churn rate by tenure cohorts (years with company)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["0-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
    df_temp = df.copy()
    df_temp["tenure_cohort"] = pd.cut(
        df_temp["tenure"], bins=bins, labels=labels, include_lowest=True
    )

    cohort_summary = (
        df_temp.groupby("tenure_cohort", observed=True)["churn_label"]
        .agg(["count", "mean"])
        .reset_index()
    )
    cohort_summary["churn_pct"] = cohort_summary["mean"] * 100

    palette = sns.color_palette("Reds_r", len(cohort_summary))
    bars = ax.bar(
        cohort_summary["tenure_cohort"].astype(str),
        cohort_summary["churn_pct"],
        color=palette,
        width=0.6,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, count in zip(bars, cohort_summary["count"]):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.0,
            f"{yval:.1f}%\n(N={count:,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title("Customer Churn Hazard by Tenure Cohort", pad=15, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure Cohort (Months)")
    ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    out_file = output_dir / "03_churn_by_tenure_cohorts.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file.name)
    return out_file


def plot_churn_by_services(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot churn rate by Internet Service and Tech Support."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Internet Service
    net_summary = (
        df.groupby("InternetService")["churn_label"]
        .agg(["count", "mean"])
        .reset_index()
    )
    net_summary["churn_pct"] = net_summary["mean"] * 100
    net_summary = net_summary.sort_values("churn_pct", ascending=False)

    axes[0].bar(
        net_summary["InternetService"],
        net_summary["churn_pct"],
        color=["#d9534f", "#428bca", "#5cb85c"],
        width=0.5,
        edgecolor="black",
        linewidth=0.8,
    )
    for i, row in net_summary.iterrows():
        axes[0].text(
            row["InternetService"],
            row["churn_pct"] + 1.0,
            f"{row['churn_pct']:.1f}%\n(N={row['count']:,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    axes[0].set_title("Churn Rate by Internet Service", fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_ylim(0, 50)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # 2. Tech Support
    tech_summary = (
        df.groupby("TechSupport")["churn_label"]
        .agg(["count", "mean"])
        .reset_index()
    )
    tech_summary["churn_pct"] = tech_summary["mean"] * 100
    tech_summary = tech_summary.sort_values("churn_pct", ascending=False)

    axes[1].bar(
        tech_summary["TechSupport"],
        tech_summary["churn_pct"],
        color=["#d9534f", "#428bca", "#5cb85c"],
        width=0.5,
        edgecolor="black",
        linewidth=0.8,
    )
    for i, row in tech_summary.iterrows():
        axes[1].text(
            row["TechSupport"],
            row["churn_pct"] + 1.0,
            f"{row['churn_pct']:.1f}%\n(N={row['count']:,})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )
    axes[1].set_title("Churn Rate by Tech Support Subscription", fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)")
    axes[1].set_ylim(0, 50)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    out_file = output_dir / "04_churn_by_internet_and_tech_support.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file.name)
    return out_file


def plot_churn_by_payment_method(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot churn rate across payment methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pay_summary = (
        df.groupby("PaymentMethod")["churn_label"]
        .agg(["count", "mean"])
        .reset_index()
    )
    pay_summary["churn_pct"] = pay_summary["mean"] * 100
    pay_summary = pay_summary.sort_values("churn_pct", ascending=True)

    colors = [
        "#5cb85c" if "automatic" in m.lower() else "#d9534f" if "electronic" in m.lower() else "#f0ad4e"
        for m in pay_summary["PaymentMethod"]
    ]

    bars = ax.barh(
        pay_summary["PaymentMethod"],
        pay_summary["churn_pct"],
        color=colors,
        height=0.55,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, count in zip(bars, pay_summary["count"]):
        xval = bar.get_width()
        ax.text(
            xval + 1.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{xval:.1f}% (N={count:,})",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_title("Churn Rate by Payment Method\n(Green = Automatic, Red = Electronic Check)", pad=15, fontweight="bold")
    ax.set_xlabel("Churn Rate (%)")
    ax.set_xlim(0, 55)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    out_file = output_dir / "05_churn_by_payment_method.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file.name)
    return out_file


def plot_charges_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot MonthlyCharges and TotalCharges density curves by Churn."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Monthly Charges
    sns.kdeplot(
        data=df,
        x="MonthlyCharges",
        hue="Churn",
        palette={"No": "#2b5c8f", "Yes": "#d9534f"},
        fill=True,
        common_norm=False,
        alpha=0.35,
        linewidth=2,
        ax=axes[0],
    )
    axes[0].set_title("Monthly Charges Distribution by Churn", fontweight="bold")
    axes[0].set_xlabel("Monthly Charges ($)")
    axes[0].set_ylabel("Density")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Total Charges
    sns.kdeplot(
        data=df,
        x="TotalCharges",
        hue="Churn",
        palette={"No": "#2b5c8f", "Yes": "#d9534f"},
        fill=True,
        common_norm=False,
        alpha=0.35,
        linewidth=2,
        ax=axes[1],
    )
    axes[1].set_title("Total Charges Distribution by Churn", fontweight="bold")
    axes[1].set_xlabel("Total Charges ($)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    out_file = output_dir / "06_charges_distribution_by_churn.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file.name)
    return out_file


def compute_statistical_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key quantitative EDA benchmarks for documentation."""
    summary = {}
    summary["total_customers"] = len(df)
    summary["churned_count"] = int((df["churn_label"] == 1).sum())
    summary["retained_count"] = int((df["churn_label"] == 0).sum())
    summary["churn_rate"] = round(df["churn_label"].mean() * 100, 2)

    # Monthly charges stats
    summary["monthly_charges_churned_mean"] = round(
        df.loc[df["churn_label"] == 1, "MonthlyCharges"].mean(), 2
    )
    summary["monthly_charges_retained_mean"] = round(
        df.loc[df["churn_label"] == 0, "MonthlyCharges"].mean(), 2
    )

    # Median tenure stats
    summary["median_tenure_churned"] = float(
        df.loc[df["churn_label"] == 1, "tenure"].median()
    )
    summary["median_tenure_retained"] = float(
        df.loc[df["churn_label"] == 0, "tenure"].median()
    )

    # Contract churn rates
    for contract, rate in (
        df.groupby("Contract")["churn_label"].mean() * 100
    ).items():
        summary[f"churn_rate_contract_{contract}"] = round(rate, 2)

    # Internet service churn rates
    for net, rate in (
        df.groupby("InternetService")["churn_label"].mean() * 100
    ).items():
        summary[f"churn_rate_internet_{net}"] = round(rate, 2)

    # Payment method churn rates
    for pay, rate in (
        df.groupby("PaymentMethod")["churn_label"].mean() * 100
    ).items():
        summary[f"churn_rate_payment_{pay}"] = round(rate, 2)

    # Tech support churn rates
    for tech, rate in (
        df.groupby("TechSupport")["churn_label"].mean() * 100
    ).items():
        summary[f"churn_rate_tech_support_{tech}"] = round(rate, 2)

    return summary


def run_eda_visualizations():
    """Execute all plot generation and print statistical benchmarks."""
    setup_plotting_theme()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_cleaned_data()

    logger.info("Generating EDA figures...")
    plot_overall_churn(df, FIGURES_DIR)
    plot_churn_by_contract(df, FIGURES_DIR)
    plot_churn_by_tenure_cohorts(df, FIGURES_DIR)
    plot_churn_by_services(df, FIGURES_DIR)
    plot_churn_by_payment_method(df, FIGURES_DIR)
    plot_charges_distribution(df, FIGURES_DIR)

    stats = compute_statistical_summary(df)
    print("\n" + "=" * 60)
    print("KEY EDA STATISTICAL BENCHMARKS")
    print("=" * 60)
    for k, v in stats.items():
        print(f" - {k}: {v}")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    run_eda_visualizations()
