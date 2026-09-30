"""
Exploratory Data Analysis.

Approach
--------
Each figure answers one question a retention team would ask:
  1. Which contract types churn most?            -> bar chart of churn rate
  2. When in the customer lifetime does churn happen? -> churn rate by tenure
  3. Do churners pay more per month?              -> overlaid distributions
  4. Does bundling more services reduce churn?     -> churn rate by #services
The same numbers are also saved as CSV so they can be quoted exactly.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src import config
from src.plot_style import C, apply_style, save


def _churn_rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Customers, churners and churn rate (%) per level of `by`."""
    t = df.groupby(by, observed=True)[config.TARGET].agg(customers="size", churned="sum")
    t["churn_rate_pct"] = 100 * t["churned"] / t["customers"]
    return t.reset_index()


def plot_churn_by_contract(df: pd.DataFrame) -> pd.DataFrame:
    t = _churn_rate_table(df, "Contract").sort_values("churn_rate_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bars = ax.bar(t["Contract"], t["churn_rate_pct"], color=C["blue"], width=0.55)
    # Direct labels on 3 bars are readable; no legend needed for one series
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in t["churn_rate_pct"]],
                 padding=3, color=C["text"])
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_ylabel("Churn rate")
    ax.grid(axis="x", visible=False)
    ax.set_title("Churn rate by contract type")
    save(fig, "01_churn_by_contract")
    return t


def plot_churn_by_tenure(df: pd.DataFrame) -> pd.DataFrame:
    # 6-month buckets keep each point backed by hundreds of customers
    bins = np.arange(0, 78, 6)
    d = df.assign(tenure_bin=pd.cut(df["tenure"], bins=bins, right=False))
    t = _churn_rate_table(d, "tenure_bin")
    t["bin_start"] = [iv.left for iv in t["tenure_bin"]]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(t["bin_start"], t["churn_rate_pct"], color=C["blue"], marker="o", markersize=5)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Tenure (months, 6-month buckets)")
    ax.set_ylabel("Churn rate")
    ax.set_ylim(bottom=0)
    ax.set_title("Churn is concentrated in the first year")
    save(fig, "02_churn_by_tenure")
    return t.drop(columns="tenure_bin")


def plot_monthly_charges(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bins = np.linspace(df["MonthlyCharges"].min(), df["MonthlyCharges"].max(), 30)
    # Density (not counts) so the two groups are comparable despite imbalance;
    # step outlines instead of filled bars so overlaps stay readable
    for label, value, color in [("Retained", 0, C["blue"]), ("Churned", 1, C["orange"])]:
        ax.hist(df.loc[df[config.TARGET] == value, "MonthlyCharges"], bins=bins,
                density=True, histtype="step", linewidth=2, color=color, label=label)
    ax.set_xlabel("Monthly charges (USD)")
    ax.set_ylabel("Density")
    # Legend sits top-centre, clear of the tall low-charge spike at the left
    ax.legend(loc="upper center")
    ax.set_title("Churners skew towards higher monthly charges")
    save(fig, "03_monthly_charges_by_churn")


def plot_churn_by_num_services(df: pd.DataFrame) -> pd.DataFrame:
    services = ["PhoneService", "MultipleLines"] + config.INTERNET_ADDONS
    d = df.assign(num_services=(df[services] == "Yes").sum(axis=1))
    t = _churn_rate_table(d, "num_services")
    t = t[t["customers"] >= 50]  # hide tiny groups with unstable rates

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(t["num_services"].astype(str), t["churn_rate_pct"], color=C["blue"], width=0.6)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_xlabel("Number of subscribed services")
    ax.set_ylabel("Churn rate")
    ax.grid(axis="x", visible=False)
    ax.set_title("Churn rate by number of services")
    save(fig, "04_churn_by_num_services")
    return t


def run(df: pd.DataFrame) -> dict:
    """Draw every EDA figure and save the underlying tables."""
    apply_style()
    out_dir = config.REPORTS_DIR / "eda_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "churn_by_contract": plot_churn_by_contract(df),
        "churn_by_tenure": plot_churn_by_tenure(df),
        "churn_by_num_services": plot_churn_by_num_services(df),
    }
    plot_monthly_charges(df)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    return tables
