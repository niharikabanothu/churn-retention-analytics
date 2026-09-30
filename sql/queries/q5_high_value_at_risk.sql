-- Q5. High-value customers at risk.
-- NTILE(4) splits customers into monthly-charge quartiles; we then compare
-- churn for the top quartile on month-to-month plans against everyone else.
-- This is the segment a retention team would call first.
WITH ranked AS (
    SELECT
        a.*,
        NTILE(4) OVER (ORDER BY a.monthly_charges) AS charge_quartile
    FROM accounts AS a
)
SELECT
    CASE
        WHEN charge_quartile = 4 AND contract = 'Month-to-month'
            THEN 'Top-quartile spend, month-to-month'
        ELSE 'All other customers'
    END                                             AS segment,
    COUNT(*)                                        AS customers,
    ROUND(100.0 * AVG(churned), 2)                  AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)                  AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END), 2)
                                                    AS monthly_revenue_lost
FROM ranked
GROUP BY segment
ORDER BY churn_rate_pct DESC;
