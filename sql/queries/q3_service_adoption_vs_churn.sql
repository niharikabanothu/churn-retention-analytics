-- Q3. For each service: churn rate among subscribers vs non-subscribers.
-- Restricted to internet customers so add-on services are compared fairly
-- (customers without internet cannot subscribe to them at all).
SELECT
    s.service_name,
    SUM(s.subscribed)                                               AS subscribers,
    ROUND(100.0 * AVG(CASE WHEN s.subscribed = 1 THEN a.churned END), 2)
                                                                    AS churn_rate_subscribed_pct,
    ROUND(100.0 * AVG(CASE WHEN s.subscribed = 0 THEN a.churned END), 2)
                                                                    AS churn_rate_not_subscribed_pct,
    -- Positive gap = subscribing to this service goes with LOWER churn
    ROUND(100.0 * (AVG(CASE WHEN s.subscribed = 0 THEN a.churned END)
                 - AVG(CASE WHEN s.subscribed = 1 THEN a.churned END)), 2)
                                                                    AS churn_gap_pct_points
FROM services AS s
JOIN accounts AS a ON a.customer_id = s.customer_id
WHERE a.internet_service <> 'No'
GROUP BY s.service_name
ORDER BY churn_gap_pct_points DESC;
