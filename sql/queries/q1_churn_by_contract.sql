-- Q1. Churn rate and monthly revenue at risk by contract type.
-- Revenue at risk = monthly charges of customers who churned.
SELECT
    contract,
    COUNT(*)                                        AS customers,
    SUM(churned)                                    AS churned,
    ROUND(100.0 * AVG(churned), 2)                  AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                  AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END), 2)
                                                    AS monthly_revenue_lost,
    -- Share of all lost monthly revenue that comes from this contract type
    ROUND(100.0 * SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END)
          / SUM(SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END)) OVER (), 2)
                                                    AS share_of_lost_revenue_pct
FROM accounts
GROUP BY contract
ORDER BY churn_rate_pct DESC;
