"""
Model training, evaluation and business-driven threshold selection.

Approach
--------
1. Stratified 80/20 split. The test set is touched ONCE, at the very end.
2. Compare models with 5-fold stratified cross-validation on the training set:
   a prior-only baseline, logistic regression, random forest, and a
   hyperparameter-tuned histogram gradient boosting model.
3. Choose the model by PR-AUC, not accuracy: only ~26.5% of customers churn,
   so a model that predicts "no churn" for everyone is already 73.5% accurate.
4. Choose the decision THRESHOLD by money, not by 0.5: we sweep thresholds on
   out-of-fold training predictions and keep the one that maximises expected
   net savings of a retention campaign (assumptions in config.py). Using
   out-of-fold predictions means the threshold is never tuned on the test set.
5. Report hold-out metrics, lift, and permutation importance, and compare the
   model-driven campaign against simple rule-based alternatives.
"""
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, make_scorer,
    precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor, split_xy
from src.plot_style import C, apply_style, save

SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "recall": "recall",
    # zero_division=0 so the prior baseline (which never predicts churn)
    # scores 0 precision quietly instead of raising warnings
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}


def make_cv():
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                           random_state=config.RANDOM_STATE)


def make_pipeline(model, engineered: bool = True) -> Pipeline:
    """Preprocessing + model in one object, so CV refits BOTH on each fold."""
    return Pipeline([("prep", build_preprocessor(engineered)), ("model", model)])


def candidate_models() -> dict:
    """Untuned candidates. class_weight counters the 3:1 class imbalance."""
    return {
        "baseline_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_STATE,
        ),
    }


def tune_gradient_boosting(X, y) -> Pipeline:
    """Randomized search over the most influential boosting hyperparameters."""
    pipe = make_pipeline(HistGradientBoostingClassifier(
        class_weight="balanced", random_state=config.RANDOM_STATE))
    search = RandomizedSearchCV(
        pipe,
        param_distributions={
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_leaf_nodes": randint(8, 48),
            "model__max_depth": [3, 4, 5, 6, None],
            "model__min_samples_leaf": randint(20, 120),
            "model__l2_regularization": loguniform(1e-3, 10),
            "model__max_iter": randint(100, 400),
        },
        n_iter=25,
        scoring="average_precision",
        # Different fold seed from the comparison CV, which reduces (but does
        # not fully remove) optimism from selecting on the same folds
        cv=StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                           random_state=config.RANDOM_STATE + 1),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def cross_validate_models(pipelines: dict, X, y) -> pd.DataFrame:
    """Mean and std of every metric across the CV folds, one row per model."""
    rows = []
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X, y, cv=make_cv(), scoring=SCORING, n_jobs=-1)
        row = {"model": name}
        for metric in SCORING:
            vals = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


def feature_engineering_ablation(X_train_df, y_train) -> pd.DataFrame:
    """Does feature engineering actually help? Same model, with vs without."""
    rows = []
    for engineered in (False, True):
        pipe = make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"),
                             engineered=engineered)
        X, _ = split_xy(X_train_df, engineered=engineered)
        s = cross_validate(pipe, X, y_train, cv=make_cv(),
                           scoring=["roc_auc", "average_precision"], n_jobs=-1)
        rows.append({
            "feature_set": "raw + engineered" if engineered else "raw only",
            "roc_auc_mean": s["test_roc_auc"].mean(),
            "pr_auc_mean": s["test_average_precision"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Business evaluation
# ---------------------------------------------------------------------------
def campaign_net_savings(y_true, targeted, monthly_charges) -> float:
    """
    Expected net savings of contacting the `targeted` customers.

    For every targeted customer we pay OFFER_COST. For every targeted customer
    who would really have churned, the offer succeeds with probability
    RETENTION_SUCCESS_RATE and saves VALUE_HORIZON_MONTHS of their bill.
    """
    targeted = np.asarray(targeted, dtype=bool)
    saved_value = (
        np.asarray(y_true)[targeted] * np.asarray(monthly_charges)[targeted]
        * config.VALUE_HORIZON_MONTHS * config.RETENTION_SUCCESS_RATE
    ).sum()
    return float(saved_value - config.OFFER_COST * targeted.sum())


def choose_threshold(y_true, proba, monthly_charges):
    """Sweep thresholds and return the one with the highest net savings."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    savings = np.array([
        campaign_net_savings(y_true, proba >= t, monthly_charges) for t in thresholds
    ])
    best = int(np.argmax(savings))
    return float(thresholds[best]), thresholds, savings


def top_decile_lift(y_true, proba) -> float:
    """Churn rate in the 10% highest-risk customers divided by the base rate."""
    n_top = max(1, int(round(0.10 * len(proba))))
    top_idx = np.argsort(proba)[::-1][:n_top]
    return float(np.asarray(y_true)[top_idx].mean() / np.mean(y_true))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_curves(fitted: dict, X_test, y_test) -> None:
    """ROC and precision-recall curves for the three real models (no baseline)."""
    colors = [C["blue"], C["orange"], C["aqua"]]
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))
    for (name, pipe), color in zip(fitted.items(), colors):
        p = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        label = name.replace("_", " ")
        ax_roc.plot(fpr, tpr, color=color, label=f"{label} ({roc_auc_score(y_test, p):.3f})")
        ax_pr.plot(rec, prec, color=color,
                   label=f"{label} ({average_precision_score(y_test, p):.3f})")

    ax_roc.plot([0, 1], [0, 1], color=C["muted"], linestyle="--", linewidth=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate")
    ax_roc.set_title("ROC curve (test set, AUC in legend)")
    ax_pr.axhline(np.mean(y_test), color=C["muted"], linestyle="--", linewidth=1)
    ax_pr.set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    ax_pr.set_title("Precision-recall (test set, AP in legend)")
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="upper right")
    save(fig, "05_roc_pr_curves")


def plot_threshold_savings(thresholds, savings, best_t) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(thresholds, savings / 1000, color=C["blue"])
    ax.axvline(best_t, color=C["muted"], linestyle="--", linewidth=1)
    ax.annotate(f"best threshold = {best_t:.2f}", xy=(best_t, savings.max() / 1000),
                xytext=(8, -14), textcoords="offset points", color=C["text"])
    ax.axhline(0, color=C["muted"], linewidth=0.8)
    ax.set_xlabel("Decision threshold (predicted churn probability)")
    ax.set_ylabel("Net savings (USD, thousands)")
    ax.set_title("Threshold chosen by expected campaign savings (train, out-of-fold)")
    save(fig, "06_threshold_vs_savings")


def plot_importance(imp: pd.DataFrame) -> None:
    top = imp.head(10).iloc[::-1]  # largest at the top of a horizontal bar chart
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"],
            color=C["blue"], height=0.6, error_kw={"ecolor": C["muted"], "linewidth": 1})
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    ax.set_title("Permutation importance (test set, top 10)")
    save(fig, "07_permutation_importance")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(df: pd.DataFrame) -> dict:
    apply_style()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = config.REPORTS_DIR / "modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Hold-out split, stratified so both sets keep the churn rate -----
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET],
        random_state=config.RANDOM_STATE,
    )
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    # --- 2. Does feature engineering help? -----------------------------------
    ablation = feature_engineering_ablation(train_df, y_train)
    ablation.to_csv(out_dir / "feature_engineering_ablation.csv", index=False)

    # --- 3. Candidate models, including a tuned gradient boosting model -----
    pipelines = {name: make_pipeline(m) for name, m in candidate_models().items()}
    tuned_gb, best_params = tune_gradient_boosting(X_train, y_train)
    pipelines["gradient_boosting_tuned"] = tuned_gb

    cv_results = cross_validate_models(pipelines, X_train, y_train)
    cv_results.to_csv(out_dir / "cv_results.csv", index=False)

    # Best real model by mean CV PR-AUC (the prior baseline is never chosen)
    ranked = cv_results[cv_results["model"] != "baseline_prior"]
    best_name = ranked.iloc[0]["model"]
    best_pipe = clone(pipelines[best_name])

    # --- 4. Threshold from out-of-fold training predictions ------------------
    oof_proba = cross_val_predict(best_pipe, X_train, y_train, cv=make_cv(),
                                  method="predict_proba", n_jobs=-1)[:, 1]
    best_t, thresholds, savings = choose_threshold(
        y_train, oof_proba, train_df["MonthlyCharges"].to_numpy())

    # --- 5. Fit on the full training set, evaluate ONCE on the test set -----
    fitted = {}
    for name in ["logistic_regression", "random_forest", "gradient_boosting_tuned"]:
        fitted[name] = clone(pipelines[name]).fit(X_train, y_train)
    final = fitted[best_name]
    proba = final.predict_proba(X_test)[:, 1]
    pred = (proba >= best_t).astype(int)
    test_charges = test_df["MonthlyCharges"].to_numpy()

    # Compare the model-driven campaign with simple rules a business might use
    policies = {
        "model_at_chosen_threshold": pred.astype(bool),
        "model_at_default_0.5": proba >= 0.5,
        "rule_all_month_to_month": (test_df["Contract"] == "Month-to-month").to_numpy(),
        "rule_target_everyone": np.ones(len(y_test), dtype=bool),
        "rule_target_nobody": np.zeros(len(y_test), dtype=bool),
    }
    policy_table = pd.DataFrame([
        {
            "policy": name,
            "customers_targeted": int(t.sum()),
            "churners_reached": int(y_test[t].sum()),
            "recall": float(y_test[t].sum() / y_test.sum()),
            "net_savings_usd": round(campaign_net_savings(y_test, t, test_charges), 2),
        }
        for name, t in policies.items()
    ])
    policy_table.to_csv(out_dir / "campaign_policy_comparison.csv", index=False)

    test_metrics = {
        "model": best_name,
        "threshold": best_t,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier_score": brier_score_loss(y_test, proba),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "top_decile_lift": top_decile_lift(y_test, proba),
        "test_base_churn_rate": float(np.mean(y_test)),
    }

    # --- 6. Which inputs does the model rely on? ----------------------------
    # Permuting RAW input columns (before one-hot) keeps importances readable
    perm = permutation_importance(final, X_test, y_test, scoring="average_precision",
                                  n_repeats=10, random_state=config.RANDOM_STATE, n_jobs=-1)
    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(out_dir / "permutation_importance.csv", index=False)

    # --- 7. Ranked risk list: what a retention team would actually receive --
    risk = test_df[[config.ID_COL, "Contract", "tenure", "MonthlyCharges"]].copy()
    risk["churn_probability"] = proba.round(4)
    risk["target_for_offer"] = pred
    risk["actual_churn"] = y_test
    risk.sort_values("churn_probability", ascending=False).to_csv(
        out_dir / "test_risk_ranking.csv", index=False)

    # --- 8. Figures, model artefact and a single metrics file ---------------
    plot_curves(fitted, X_test, y_test)
    plot_threshold_savings(thresholds, savings, best_t)
    plot_importance(importance)
    joblib.dump(final, config.MODEL_DIR / "churn_model.joblib")

    summary = {
        "selected_model": best_name,
        "gradient_boosting_best_params": {k: (v if isinstance(v, (int, float, type(None)))
                                              else float(v)) for k, v in best_params.items()},
        "decision_threshold": best_t,
        "business_assumptions": {
            "offer_cost_usd": config.OFFER_COST,
            "retention_success_rate": config.RETENTION_SUCCESS_RATE,
            "value_horizon_months": config.VALUE_HORIZON_MONTHS,
        },
        "test_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in test_metrics.items()},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    return {
        "cv_results": cv_results,
        "ablation": ablation,
        "policies": policy_table,
        "importance": importance,
        "summary": summary,
    }
