"""
Run the full churn analytics pipeline end to end.

    python run_pipeline.py

Steps: clean -> SQL analysis -> EDA -> statistical tests -> modeling.
Every output lands in reports/ (tables, figures, metrics) and models/.
"""
import json

import pandas as pd

from src import data_prep, eda, modeling, sql_analysis, stats_tests
from src.features import add_features

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    section("1. Data quality and cleaning")
    df, report = data_prep.run()
    print(json.dumps(report, indent=2))

    section("2. SQL analysis (SQLite, normalized schema)")
    sql_analysis.build_database(df)
    for name, result in sql_analysis.run_all_queries().items():
        print(f"\n-- {name}\n{result.to_string(index=False)}")

    section("3. Exploratory data analysis")
    eda.run(df)
    print("Figures saved to reports/figures/")

    section("4. Statistical tests")
    tests = stats_tests.run(df)
    print(tests["chi_square"][["feature", "cramers_v", "p_value_bonferroni"]]
          .head(8).to_string(index=False))
    print(tests["mann_whitney"].to_string(index=False))
    print(json.dumps(tests["contract_gap"], indent=2))

    section("5. Modeling")
    results = modeling.run(add_features(df))
    print("Feature engineering ablation (logistic regression, 5-fold CV):")
    print(results["ablation"].to_string(index=False))
    print("\nCross-validation on training set:")
    print(results["cv_results"].round(4).to_string(index=False))
    print("\nCampaign policies on the test set:")
    print(results["policies"].to_string(index=False))
    print("\nTop 8 features by permutation importance:")
    print(results["importance"].head(8).round(4).to_string(index=False))
    print("\n" + json.dumps(results["summary"], indent=2))


if __name__ == "__main__":
    main()
