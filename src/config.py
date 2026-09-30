"""
Central configuration for the churn analytics project.

Every path, random seed, and business assumption lives here so that the
analysis is reproducible and the assumptions are easy to find and change.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, so the repo runs from any folder)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "telco_clean.csv"
DB_PATH = ROOT / "data" / "processed" / "telco.db"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODEL_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42      # one seed for splits, CV folds, models and bootstraps
TEST_SIZE = 0.20       # 80/20 stratified hold-out split
CV_FOLDS = 5           # stratified k-fold on the training set only

# ---------------------------------------------------------------------------
# Business assumptions used to pick the decision threshold.
# These are ASSUMED values, not facts from the dataset; change them to match
# a real retention budget. The dataset's charges are in USD.
# ---------------------------------------------------------------------------
OFFER_COST = 100.0             # cost of one retention offer (discount + agent call), USD
RETENTION_SUCCESS_RATE = 0.30  # share of targeted churners the offer actually keeps
VALUE_HORIZON_MONTHS = 12      # months of revenue credited when a churner is kept

# ---------------------------------------------------------------------------
# Column groups (raw dataset names)
# ---------------------------------------------------------------------------
TARGET = "Churn"
ID_COL = "customerID"

# Add-on services that only exist when the customer has internet
INTERNET_ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Plain-hex colours from a colourblind-validated categorical palette.
# Slot 1 (blue) = retained / baseline, slot 2 (orange) = churned / highlight.
COLORS = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
}
