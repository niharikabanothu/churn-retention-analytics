-- Q4. Rank payment methods by churn risk within each contract type.
-- RANK() is computed per contract, so we can see whether a payment method is
-- risky on its own or only because it is common on month-to-month plans.
WITH by_group AS (
    SELECT
        contract,
        payment_method,
        COUNT(*)                        AS customers,
        ROUND(100.0 * AVG(churned), 2)  AS churn_rate_pct
    FROM accounts
    GROUP BY contract, payment_method
    HAVING COUNT(*) >= 50               -- drop tiny groups with unstable rates
)
SELECT
    contract,
    payment_method,
    customers,
    churn_rate_pct,
    RANK() OVER (PARTITION BY contract ORDER BY churn_rate_pct DESC) AS risk_rank_within_contract
FROM by_group
ORDER BY contract, risk_rank_within_contract;
