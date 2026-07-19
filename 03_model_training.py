"""
Phase 4 - XGBoost + SHAP + AUROC
Reads training data from BigQuery, trains XGBoost, computes SHAP,
evaluates AUROC, writes all scores back to BigQuery.

PMLE relevance: this entire workflow mirrors Vertex AI custom training jobs.
"""

import os
import uuid
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from google.cloud import bigquery

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
import shap
import pickle

warnings.filterwarnings("ignore")

PROJECT_ID = "labs601-edis"
DATASET_ID = "edis"
client     = bigquery.Client(project=PROJECT_ID)
RUN_ID     = str(uuid.uuid4())[:8]

FEATURES = [
    "deal_size_log", "days_in_stage", "num_meetings", "email_open_rate",
    "champion_identified", "competitor_present", "product_fit_score",
    "account_tenure_yrs", "prev_purchases", "exec_sponsor", "nps_score",
    "days_since_contact", "meetings_per_day", "engagement_score",
    "stage_weight", "industry_score", "contact_lapsed",
    "high_value_deal", "champion_and_sponsor",
]

FEATURE_LABELS = [
    "Deal size (log)", "Days in stage", "Meetings", "Email open rate",
    "Champion identified", "Competitor present", "Product fit score",
    "Account tenure (yrs)", "Previous purchases", "Exec sponsor", "NPS score",
    "Days since contact", "Meetings per day", "Engagement score",
    "Stage weight", "Industry score", "Contact lapsed",
    "High value deal", "Champion and sponsor",
]


# ── Phase 4a: Load training data from BigQuery ────────────────────────────────
def load_training_data():
    print("Loading training data from BigQuery view...")
    query = f"""
        SELECT
            account_id,
            deal_size_log, days_in_stage, num_meetings, email_open_rate,
            champion_identified, competitor_present, product_fit_score,
            account_tenure_yrs, prev_purchases, exec_sponsor, nps_score,
            days_since_contact, meetings_per_day, engagement_score,
            stage_weight, industry_score, contact_lapsed,
            high_value_deal, champion_and_sponsor,
            converted
        FROM `{PROJECT_ID}.{DATASET_ID}.vw_features`
    """
    df = client.query(query).to_dataframe()
    print(f"  Rows loaded   : {len(df):,}")
    print(f"  Features      : {len(FEATURES)}")
    print(f"  Conversion    : {df['converted'].mean():.1%}")
    return df


# ── Phase 4b: Train XGBoost ───────────────────────────────────────────────────
def train_xgboost(df):
    print("\nTraining XGBoost model...")
    X = df[FEATURES]
    y = df["converted"]
    account_ids = df["account_id"]

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, account_ids, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train rows : {len(X_train):,}")
    print(f"  Test rows  : {len(X_test):,}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return model, X_train, X_test, y_test, y_prob, y_pred, ids_test


# ── Phase 4c: AUROC evaluation ────────────────────────────────────────────────
def evaluate_model(y_test, y_prob, y_pred):
    print("\nEvaluating model...")
    auroc = roc_auc_score(y_test, y_prob)
    ap    = average_precision_score(y_test, y_prob)
    cm    = confusion_matrix(y_test, y_pred)

    print(f"\n  AUROC             : {auroc:.4f}")
    print(f"  Avg Precision     : {ap:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives  : {cm[0][0]:,}  (correctly predicted lost)")
    print(f"    False Positives : {cm[0][1]:,}  (predicted won, was lost)")
    print(f"    False Negatives : {cm[1][0]:,}  (predicted lost, was won)")
    print(f"    True Positives  : {cm[1][1]:,}  (correctly predicted won)")

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  Precision (won)   : {precision:.3f}")
    print(f"  Recall    (won)   : {recall:.3f}")
    print(f"  F1-score  (won)   : {f1:.3f}")

    return auroc, ap, cm


# ── Phase 4d: SHAP explainability ─────────────────────────────────────────────
def compute_shap(model, X_train, X_test):
    print("\nComputing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    base_value  = float(explainer.expected_value)

    mean_shap  = np.abs(shap_values).mean(axis=0)
    global_imp = sorted(
        zip(FEATURE_LABELS, mean_shap),
        key=lambda x: x[1], reverse=True
    )

    print(f"\n  Base value (avg prediction): {base_value:.4f}")
    print(f"\n  Global feature importance (mean |SHAP|):")
    for name, val in global_imp[:10]:
        bar = ">" * int(val * 35)
        print(f"    {name:<25} {val:.4f}  {bar}")

    return explainer, shap_values, base_value, global_imp


# ── Phase 4e: Persist to BigQuery ─────────────────────────────────────────────
def persist_model_run(auroc, ap, global_imp):
    print("\nLogging model run to BigQuery...")
    row = [{
        "run_id":        RUN_ID,
        "run_at":        datetime.now(timezone.utc).isoformat(),
        "n_accounts":    5000,
        "auroc":         round(auroc, 4),
        "avg_precision": round(ap, 4),
        "top_feature":   global_imp[0][0],
        "notes":         "XGBoost v1 - 19 features - BigQuery feature view",
    }]
    errors = client.insert_rows_json(f"{PROJECT_ID}.{DATASET_ID}.model_runs", row)
    if errors:
        print(f"  Warning: {errors}")
    else:
        print(f"  Run ID: {RUN_ID} logged successfully.")


def persist_scores(ids_test, y_prob, y_test):
    print("Writing propensity scores to BigQuery...")
    now  = datetime.now(timezone.utc).isoformat()
    rows = []
    for acct_id, score, actual in zip(ids_test, y_prob, y_test):
        conf = "High" if score > 0.80 else ("Medium" if score > 0.55 else "Low")
        rows.append({
            "account_id":       acct_id,
            "run_id":           RUN_ID,
            "propensity_score": round(float(score), 4),
            "confidence":       conf,
            "scored_at":        now,
        })
    for i in range(0, len(rows), 500):
        errors = client.insert_rows_json(
            f"{PROJECT_ID}.{DATASET_ID}.scored_accounts",
            rows[i:i+500]
        )
        if errors:
            print(f"  Warning batch {i}: {errors}")
    print(f"  {len(rows):,} scores written.")


def persist_shap(ids_test, shap_values):
    print("Writing SHAP values to BigQuery...")
    rows = []
    for i, acct_id in enumerate(ids_test):
        for j, feat in enumerate(FEATURE_LABELS):
            rows.append({
                "account_id":   acct_id,
                "run_id":       RUN_ID,
                "feature_name": feat,
                "shap_value":   round(float(shap_values[i][j]), 5),
            })
    for i in range(0, len(rows), 500):
        errors = client.insert_rows_json(
            f"{PROJECT_ID}.{DATASET_ID}.shap_values",
            rows[i:i+500]
        )
        if errors:
            print(f"  Warning batch {i}: {errors}")
    print(f"  {len(rows):,} SHAP values written.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 4: XGBoost + SHAP + AUROC ===\n")

    df = load_training_data()

    model, X_train, X_test, y_test, y_prob, y_pred, ids_test = train_xgboost(df)

    auroc, ap, cm = evaluate_model(y_test, y_prob, y_pred)

    explainer, shap_values, base_value, global_imp = compute_shap(model, X_train, X_test)

    print("\nPersisting everything to BigQuery...")
    persist_model_run(auroc, ap, global_imp)
    persist_scores(ids_test, y_prob, y_test)
    persist_shap(ids_test, shap_values)

    # Save locally for Phase 5
    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(explainer, open("explainer.pkl", "wb"))

    scored_df = df.copy()
    scored_df["propensity_score"] = np.nan
    scored_df["run_id"] = RUN_ID
    test_idx = list(ids_test.index)

    for i, idx in enumerate(test_idx):
        scored_df.at[idx, "propensity_score"] = y_prob[i]

    for j, feat in enumerate(FEATURE_LABELS):
        col = "shap_" + feat.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        scored_df[col] = np.nan
        for i, idx in enumerate(test_idx):
            scored_df.at[idx, col] = shap_values[i][j]

    scored_df = scored_df.dropna(subset=["propensity_score"])
    scored_df.to_csv("scored_accounts.csv", index=False)

    print("\n  model.pkl        saved")
    print("  explainer.pkl    saved")
    print("  scored_accounts.csv saved")
    print(f"\n=== Phase 4 complete. Run ID: {RUN_ID} ===")
    print("\nVerify in BigQuery console:")
    print("  Left menu -> BigQuery -> edis dataset -> scored_accounts -> Preview tab")