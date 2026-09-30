"""
Statistical tests that validate the EDA findings.

Approach
--------
A chart can suggest a difference; a test tells us whether it is larger than
chance. For every test we report an EFFECT SIZE as well as a p-value, because
with 7,000 rows almost any difference is "significant" - the effect size says
whether it matters.

- Categorical feature vs churn : chi-square test of independence + Cramér's V,
  with a Bonferroni correction because we run one test per feature.
- Numeric feature vs churn     : Mann-Whitney U (no normality assumption;
  tenure and charges are clearly skewed) + rank-biserial correlation.
- Headline comparison          : two-proportion z-test and a bootstrap 95% CI
  for the churn-rate gap between month-to-month and longer contracts.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src import config

CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", *config.INTERNET_ADDONS,
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]


def chi_square_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL:
        table = pd.crosstab(df[col], df[config.TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        # Cramér's V: 0 = no association, 1 = perfect association
        cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
        rows.append({"feature": col, "chi2": chi2, "dof": dof,
                     "p_value": p, "cramers_v": cramers_v})

    res = pd.DataFrame(rows)
    # Bonferroni: multiply each p-value by the number of tests (capped at 1)
    res["p_value_bonferroni"] = (res["p_value"] * len(res)).clip(upper=1.0)
    res["significant"] = res["p_value_bonferroni"] < alpha
    return res.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def mann_whitney_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    churned = df[config.TARGET] == 1
    for col in NUMERIC:
        a, b = df.loc[churned, col], df.loc[~churned, col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        # Rank-biserial correlation: >0 means churners tend to have higher values
        rank_biserial = 2 * u / (len(a) * len(b)) - 1
        rows.append({
            "feature": col,
            "median_churned": a.median(),
            "median_retained": b.median(),
            "u_statistic": u,
            "p_value": p,
            "rank_biserial": rank_biserial,
        })
    return pd.DataFrame(rows)


def contract_gap_test(df: pd.DataFrame, n_boot: int = 5000) -> dict:
    """Is month-to-month churn really higher than 1-2 year contract churn?"""
    m2m = df.loc[df["Contract"] == "Month-to-month", config.TARGET].to_numpy()
    longer = df.loc[df["Contract"] != "Month-to-month", config.TARGET].to_numpy()

    # Two-proportion z-test using the pooled proportion
    p1, p2 = m2m.mean(), longer.mean()
    pooled = (m2m.sum() + longer.sum()) / (len(m2m) + len(longer))
    se = np.sqrt(pooled * (1 - pooled) * (1 / len(m2m) + 1 / len(longer)))
    z = (p1 - p2) / se
    p_value = 2 * stats.norm.sf(abs(z))

    # Bootstrap CI: resample each group with replacement, recompute the gap
    rng = np.random.default_rng(config.RANDOM_STATE)
    gaps = [
        rng.choice(m2m, len(m2m)).mean() - rng.choice(longer, len(longer)).mean()
        for _ in range(n_boot)
    ]
    lo, hi = np.percentile(gaps, [2.5, 97.5])

    return {
        "churn_rate_month_to_month": round(float(p1), 4),
        "churn_rate_one_or_two_year": round(float(p2), 4),
        "gap_pct_points": round(float(100 * (p1 - p2)), 2),
        "z_statistic": round(float(z), 2),
        "p_value": float(p_value),
        "bootstrap_95ci_pct_points": [round(float(100 * lo), 2), round(float(100 * hi), 2)],
    }


def run(df: pd.DataFrame) -> dict:
    out_dir = config.REPORTS_DIR / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    chi = chi_square_tests(df)
    mw = mann_whitney_tests(df)
    gap = contract_gap_test(df)

    chi.to_csv(out_dir / "chi_square_tests.csv", index=False)
    mw.to_csv(out_dir / "mann_whitney_tests.csv", index=False)
    return {"chi_square": chi, "mann_whitney": mw, "contract_gap": gap}
