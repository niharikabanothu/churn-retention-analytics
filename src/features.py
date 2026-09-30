"""
Feature engineering and the preprocessing pipeline.

Approach
--------
Engineered features encode business intuition that the raw columns only
express indirectly:
  - num_services      : more services = more switching cost ("stickiness")
  - has_protection    : security/tech-support subscribers churn much less
  - avg_monthly_spend : lifetime spend / tenure = what they historically paid
  - charge_increase   : current bill minus historical average; a rising bill
                        is a plausible churn trigger
  - auto_pay          : automatic payments remove a monthly "decision point"
  - early_tenure      : first 12 months, where most churn happens (see EDA)

All transforms that LEARN from data (scaling, one-hot categories) live inside
a scikit-learn ColumnTransformer, so they are fit on the training fold only.
That prevents test-set information leaking into training.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

BASE_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BASE_CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", *config.INTERNET_ADDONS,
    "Contract", "PaperlessBilling", "PaymentMethod",
]
ENGINEERED_NUMERIC = ["num_services", "avg_monthly_spend", "charge_increase"]
ENGINEERED_BINARY = ["has_protection", "auto_pay", "early_tenure"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered columns added.

    Every feature here is computed row by row from that customer's own
    columns, so it is safe to compute before the train/test split.
    """
    out = df.copy()
    services = ["PhoneService", "MultipleLines"] + config.INTERNET_ADDONS

    out["num_services"] = (out[services] == "Yes").sum(axis=1)
    out["has_protection"] = (
        (out["OnlineSecurity"] == "Yes") | (out["TechSupport"] == "Yes")
    ).astype(int)

    # tenure 0 would divide by zero; for them the current bill IS the history
    out["avg_monthly_spend"] = np.where(
        out["tenure"] > 0, out["TotalCharges"] / out["tenure"].clip(lower=1),
        out["MonthlyCharges"],
    )
    out["charge_increase"] = out["MonthlyCharges"] - out["avg_monthly_spend"]
    out["auto_pay"] = out["PaymentMethod"].str.contains("automatic").astype(int)
    out["early_tenure"] = (out["tenure"] <= 12).astype(int)
    return out


def get_feature_lists(engineered: bool = True):
    """Numeric and categorical column lists, with or without engineered features."""
    numeric = BASE_NUMERIC + (ENGINEERED_NUMERIC + ENGINEERED_BINARY if engineered else [])
    return numeric, list(BASE_CATEGORICAL)


def build_preprocessor(engineered: bool = True) -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categorical ones."""
    numeric, categorical = get_feature_lists(engineered)
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        # drop='if_binary' keeps one column for Yes/No fields (no redundancy);
        # handle_unknown='ignore' protects against unseen categories at inference
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                              sparse_output=False), categorical),
    ])


def split_xy(df: pd.DataFrame, engineered: bool = True):
    """Return the model input frame X and target vector y."""
    numeric, categorical = get_feature_lists(engineered)
    return df[numeric + categorical], df[config.TARGET].to_numpy()
