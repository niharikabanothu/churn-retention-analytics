-- Q2. Churn by tenure cohort, with a running share of all churners.
-- Shows how concentrated churn is in the first months of the relationship.
WITH cohorts AS (
    SELECT
        CASE
            WHEN tenure_months <= 6  THEN '01. 0-6 months'
            WHEN tenure_months <= 12 THEN '02. 7-12 months'
            WHEN tenure_months <= 24 THEN '03. 13-24 months'
            WHEN tenure_months <= 48 THEN '04. 25-48 months'
            ELSE                          '05. 49-72 months'
        END AS tenure_cohort,
        churned
    FROM accounts
),
agg AS (
    SELECT
        tenure_cohort,
        COUNT(*)       AS customers,
        SUM(churned)   AS churned
    FROM cohorts
    GROUP BY tenure_cohort
)
SELECT
    tenure_cohort,
    customers,
    churned,
    ROUND(100.0 * churned / customers, 2) AS churn_rate_pct,
    -- Cumulative share of all churners, ordered from newest to oldest cohort
    ROUND(100.0 * SUM(churned) OVER (ORDER BY tenure_cohort)
          / SUM(churned) OVER (), 2)      AS cumulative_share_of_churners_pct
FROM agg
ORDER BY tenure_cohort;
