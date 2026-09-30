"""
SQL layer: load the cleaned data into a normalized SQLite database and run
the analysis queries in sql/queries/.

Approach
--------
SQLite ships with Python, so the SQL part runs with zero setup, while the
queries themselves (CTEs, joins, CASE, window functions like RANK/NTILE and
running sums) are standard SQL that ports directly to PostgreSQL or MySQL.
"""
import sqlite3

import pandas as pd

from src import config

# Every column that becomes a row in the long-format `services` table
SERVICE_COLUMNS = ["PhoneService", "MultipleLines"] + config.INTERNET_ADDONS


def build_database(df: pd.DataFrame, db_path=config.DB_PATH) -> None:
    """Create the schema and insert the three normalized tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    customers = df[[config.ID_COL, "gender", "SeniorCitizen", "Partner", "Dependents"]].rename(
        columns={
            config.ID_COL: "customer_id",
            "SeniorCitizen": "senior_citizen",
            "Partner": "partner",
            "Dependents": "dependents",
        }
    )

    accounts = df[[
        config.ID_COL, "tenure", "Contract", "PaperlessBilling", "PaymentMethod",
        "InternetService", "MonthlyCharges", "TotalCharges", config.TARGET,
    ]].rename(columns={
        config.ID_COL: "customer_id",
        "tenure": "tenure_months",
        "Contract": "contract",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "InternetService": "internet_service",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        config.TARGET: "churned",
    })

    # Wide -> long: one row per customer per service, subscribed as 0/1
    services = df[[config.ID_COL] + SERVICE_COLUMNS].melt(
        id_vars=config.ID_COL, var_name="service_name", value_name="subscribed"
    ).rename(columns={config.ID_COL: "customer_id"})
    services["subscribed"] = (services["subscribed"] == "Yes").astype(int)

    schema_sql = (config.SQL_DIR / "schema.sql").read_text()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_sql)
        # if_exists="append" keeps the constraints defined in schema.sql
        customers.to_sql("customers", conn, if_exists="append", index=False)
        accounts.to_sql("accounts", conn, if_exists="append", index=False)
        services.to_sql("services", conn, if_exists="append", index=False)


def run_query(sql_file, db_path=config.DB_PATH) -> pd.DataFrame:
    """Execute one .sql file and return the result as a DataFrame."""
    query = sql_file.read_text()
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def run_all_queries(db_path=config.DB_PATH, save=True) -> dict:
    """Run every query in sql/queries/, in file order, and save each as CSV."""
    out_dir = config.REPORTS_DIR / "sql_results"
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for sql_file in sorted((config.SQL_DIR / "queries").glob("*.sql")):
        result = run_query(sql_file, db_path)
        results[sql_file.stem] = result
        if save:
            result.to_csv(out_dir / f"{sql_file.stem}.csv", index=False)
    return results
