"""
Data loading, quality checks and cleaning.

Approach
--------
1. Load the raw CSV exactly as shipped.
2. Profile it (duplicates, missing values, hidden blanks, type problems)
   BEFORE changing anything, and save that profile as a report.
3. Clean with explicit, documented rules, and assert that each rule did
   what we expected so silent data bugs fail loudly.
"""
import json

import numpy as np
import pandas as pd

from src import config


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the raw Telco CSV without any type coercion."""
    return pd.read_csv(path)


def data_quality_report(df: pd.DataFrame) -> dict:
    """
    Profile the raw data and return a JSON-serialisable summary.

    The key finding here is that TotalCharges is stored as text and contains
    blank strings (" "), which pandas does NOT count as missing. Coercing to
    numeric exposes them.
    """
    total_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    blank_mask = total_numeric.isna()

    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_customer_ids": int(df[config.ID_COL].duplicated().sum()),
        "missing_values_as_loaded": int(df.isna().sum().sum()),
        "hidden_blank_total_charges": int(blank_mask.sum()),
        # Every blank TotalCharges row should be a brand-new customer (tenure 0)
        "tenure_of_blank_total_charges": sorted(
            df.loc[blank_mask, "tenure"].unique().tolist()
        ),
        "churn_rate": round(float((df[config.TARGET] == "Yes").mean()), 4),
    }

    # Consistency check: TotalCharges should be roughly tenure * MonthlyCharges.
    # A large relative gap would indicate data-entry problems.
    expected = df["tenure"] * df["MonthlyCharges"]
    rel_gap = (total_numeric - expected).abs() / expected.replace(0, np.nan)
    report["pct_rows_total_vs_expected_gap_over_20pct"] = round(
        float((rel_gap > 0.20).mean() * 100), 2
    )
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the cleaning rules and return a new, analysis-ready DataFrame.

    Rules
    -----
    - TotalCharges -> numeric. Blank values belong to tenure-0 customers who
      have not been billed yet, so the correct value is 0 (not the mean).
    - 'No internet service' / 'No phone service' -> 'No'. Whether the
      customer has internet is already captured by InternetService, so the
      duplicate category only adds redundant one-hot columns.
    - SeniorCitizen 0/1 -> 'No'/'Yes' so all binary fields share one format.
    - Churn -> integer target (1 = churned).
    """
    out = df.copy()

    # Strip stray whitespace in every text column
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        out[col] = out[col].str.strip()

    # TotalCharges: text -> float, blanks -> 0 for unbilled new customers
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    unbilled = out["TotalCharges"].isna()
    assert (out.loc[unbilled, "tenure"] == 0).all(), (
        "Found blank TotalCharges for a customer with tenure > 0; "
        "the 'unbilled new customer' rule does not hold."
    )
    out.loc[unbilled, "TotalCharges"] = 0.0

    # Collapse redundant service categories
    for col in config.INTERNET_ADDONS:
        out[col] = out[col].replace("No internet service", "No")
    out["MultipleLines"] = out["MultipleLines"].replace("No phone service", "No")

    # Harmonise binary encodings
    out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})
    out[config.TARGET] = (out[config.TARGET] == "Yes").astype(int)

    # Final guard: no missing values should survive cleaning
    assert out.isna().sum().sum() == 0, "Missing values remain after cleaning."
    return out


def run(save=True):
    """Load -> profile -> clean, and optionally persist both outputs."""
    raw = load_raw()
    report = data_quality_report(raw)
    df = clean(raw)

    if save:
        config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.PROCESSED_CSV, index=False)
        with open(config.REPORTS_DIR / "data_quality.json", "w") as f:
            json.dump(report, f, indent=2)
    return df, report
