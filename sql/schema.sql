-- Normalized schema for the Telco churn data.
-- The flat CSV is split into three tables so the analysis queries have to use
-- joins, aggregation and window functions, the way they would on a real
-- warehouse instead of a single spreadsheet.

DROP TABLE IF EXISTS services;
DROP TABLE IF EXISTS accounts;
DROP TABLE IF EXISTS customers;

-- One row per customer: demographic attributes only
CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,
    gender          TEXT NOT NULL,
    senior_citizen  TEXT NOT NULL,
    partner         TEXT NOT NULL,
    dependents      TEXT NOT NULL
);

-- One row per customer: contract, billing and the churn outcome
CREATE TABLE accounts (
    customer_id        TEXT PRIMARY KEY REFERENCES customers(customer_id),
    tenure_months      INTEGER NOT NULL,
    contract           TEXT NOT NULL,
    paperless_billing  TEXT NOT NULL,
    payment_method     TEXT NOT NULL,
    internet_service   TEXT NOT NULL,
    monthly_charges    REAL NOT NULL,
    total_charges      REAL NOT NULL,
    churned            INTEGER NOT NULL CHECK (churned IN (0, 1))
);

-- Long format: one row per (customer, service). Lets us ask questions like
-- "churn rate among subscribers of each service" with a single GROUP BY.
CREATE TABLE services (
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id),
    service_name  TEXT NOT NULL,
    subscribed    INTEGER NOT NULL CHECK (subscribed IN (0, 1)),
    PRIMARY KEY (customer_id, service_name)
);

CREATE INDEX idx_accounts_contract ON accounts(contract);
CREATE INDEX idx_services_name ON services(service_name);
