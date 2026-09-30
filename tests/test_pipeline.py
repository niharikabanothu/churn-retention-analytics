"""
Unit tests for the cleaning, SQL, feature and business-logic layers.

Run with:  pytest -q
They use the real dataset (7,043 rows), which loads in well under a second.
"""
import numpy as np
import pytest

from src import config, data_prep, sql_analysis
from src.features import add_features, build_preprocessor, split_xy
from src.modeling import campaign_net_savings, choose_threshold, top_decile_lift


@pytest.fixture(scope="module")
def clean_df():
    # save=False keeps tests from overwriting the pipeline's outputs
    df, _ = data_prep.run(save=False)
    return df


def test_quality_report_finds_hidden_blanks():
    report = data_prep.data_quality_report(data_prep.load_raw())
    # 11 TotalCharges values are blank strings, all for tenure-0 customers
    assert report["hidden_blank_total_charges"] == 11
    assert report["tenure_of_blank_total_charges"] == [0]
    assert report["duplicate_customer_ids"] == 0


def test_clean_has_no_missing_and_binary_target(clean_df):
    assert clean_df.isna().sum().sum() == 0
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert clean_df["TotalCharges"].dtype.kind == "f"
    # Redundant categories were collapsed into plain "No"
    assert "No internet service" not in clean_df["OnlineSecurity"].unique()


def test_sql_database_and_queries(clean_df, tmp_path):
    db = tmp_path / "test.db"
    sql_analysis.build_database(clean_df, db_path=db)
    q1 = sql_analysis.run_query(config.SQL_DIR / "queries" / "q1_churn_by_contract.sql", db)
    # Three contract types, and customer counts add back to the full dataset
    assert len(q1) == 3
    assert q1["customers"].sum() == len(clean_df)
    # Revenue shares from the window function must sum to 100%
    assert q1["share_of_lost_revenue_pct"].sum() == pytest.approx(100, abs=0.05)


def test_engineered_features_are_valid(clean_df):
    feats = add_features(clean_df)
    assert feats[["num_services", "avg_monthly_spend", "charge_increase"]].isna().sum().sum() == 0
    assert feats["num_services"].between(0, 8).all()
    # tenure-0 customers: average spend falls back to the current bill
    new = feats[feats["tenure"] == 0]
    assert np.allclose(new["avg_monthly_spend"], new["MonthlyCharges"])


def test_preprocessor_output_is_numeric(clean_df):
    X, y = split_xy(add_features(clean_df))
    Xt = build_preprocessor().fit_transform(X)
    assert Xt.shape[0] == len(y)
    assert np.isfinite(Xt).all()


def test_campaign_savings_math():
    y = np.array([1, 0, 1, 0])
    targeted = np.array([True, True, False, False])
    charges = np.array([50.0, 80.0, 60.0, 40.0])
    # One churner reached: 50 * horizon * success rate, minus 2 offers
    expected = (50 * config.VALUE_HORIZON_MONTHS * config.RETENTION_SUCCESS_RATE
                - 2 * config.OFFER_COST)
    assert campaign_net_savings(y, targeted, charges) == pytest.approx(expected)
    # Targeting nobody costs and saves nothing
    assert campaign_net_savings(y, np.zeros(4, dtype=bool), charges) == 0


def test_threshold_and_lift_on_perfect_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    perfect = y.astype(float)  # a model that is always right
    best_t, _, _ = choose_threshold(y, perfect, np.full(500, 100.0))
    # With perfect scores, any threshold in (0, 1] targets exactly the churners
    assert 0 < best_t <= 1
    assert top_decile_lift(y, perfect) == pytest.approx(1 / y.mean())
