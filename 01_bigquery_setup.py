"""
Phase 2 — BigQuery Setup + Data Load
Creates dataset, tables, generates synthetic data, loads to BigQuery
"""

import os
from google.cloud import bigquery
import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ID = "labs601-edis"
DATASET_ID = "edis"
client     = bigquery.Client(project=PROJECT_ID)

# ── Step 1: Create dataset ───────────────────────────────────────────────────
def create_dataset():
    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset_ref.location = "asia-southeast1"
    dataset_ref.description = "EDIS Enterprise Decision Intelligence System"
    try:
        dataset = client.create_dataset(dataset_ref, exists_ok=True)
        print(f"Dataset ready: {PROJECT_ID}.{DATASET_ID}")
    except Exception as e:
        print(f"Error: {e}")

# ── Step 2: Create tables ────────────────────────────────────────────────────
SCHEMAS = {
    "accounts": [
        bigquery.SchemaField("account_id",          "STRING",  mode="REQUIRED"),
        bigquery.SchemaField("industry",             "STRING"),
        bigquery.SchemaField("region",               "STRING"),
        bigquery.SchemaField("pipeline_stage",       "STRING"),
        bigquery.SchemaField("deal_size",            "FLOAT64"),
        bigquery.SchemaField("days_in_stage",        "INT64"),
        bigquery.SchemaField("num_meetings",         "INT64"),
        bigquery.SchemaField("email_open_rate",      "FLOAT64"),
        bigquery.SchemaField("champion_identified",  "INT64"),
        bigquery.SchemaField("competitor_present",   "INT64"),
        bigquery.SchemaField("product_fit_score",    "FLOAT64"),
        bigquery.SchemaField("account_tenure_yrs",   "FLOAT64"),
        bigquery.SchemaField("prev_purchases",       "INT64"),
        bigquery.SchemaField("exec_sponsor",         "INT64"),
        bigquery.SchemaField("nps_score",            "INT64"),
        bigquery.SchemaField("days_since_contact",   "INT64"),
        bigquery.SchemaField("converted",            "INT64"),
        bigquery.SchemaField("created_at",           "TIMESTAMP"),
    ],
    "model_runs": [
        bigquery.SchemaField("run_id",        "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("run_at",        "TIMESTAMP"),
        bigquery.SchemaField("n_accounts",    "INT64"),
        bigquery.SchemaField("auroc",         "FLOAT64"),
        bigquery.SchemaField("avg_precision", "FLOAT64"),
        bigquery.SchemaField("top_feature",   "STRING"),
        bigquery.SchemaField("notes",         "STRING"),
    ],
    "scored_accounts": [
        bigquery.SchemaField("account_id",        "STRING"),
        bigquery.SchemaField("run_id",            "STRING"),
        bigquery.SchemaField("propensity_score",  "FLOAT64"),
        bigquery.SchemaField("confidence",        "STRING"),
        bigquery.SchemaField("scored_at",         "TIMESTAMP"),
    ],
    "shap_values": [
        bigquery.SchemaField("account_id",    "STRING"),
        bigquery.SchemaField("run_id",        "STRING"),
        bigquery.SchemaField("feature_name",  "STRING"),
        bigquery.SchemaField("shap_value",    "FLOAT64"),
    ],
    "nba_outputs": [
        bigquery.SchemaField("account_id",  "STRING"),
        bigquery.SchemaField("run_id",      "STRING"),
        bigquery.SchemaField("action_1",    "STRING"),
        bigquery.SchemaField("action_2",    "STRING"),
        bigquery.SchemaField("action_3",    "STRING"),
        bigquery.SchemaField("risk",        "STRING"),
        bigquery.SchemaField("confidence",  "STRING"),
        bigquery.SchemaField("rag_sources", "STRING"),
        bigquery.SchemaField("created_at",  "TIMESTAMP"),
    ],
}

def create_tables():
    for table_name, schema in SCHEMAS.items():
        table_ref = client.dataset(DATASET_ID).table(table_name)
        table     = bigquery.Table(table_ref, schema=schema)
        table     = client.create_table(table, exists_ok=True)
        print(f"  Table ready: {table_name}")

# ── Step 3: Generate + load data ─────────────────────────────────────────────
def generate_data(n=5000, seed=42):
    np.random.seed(seed)
    industries = ["Financial Services","Healthcare","Retail",
                  "Manufacturing","Technology","Logistics"]
    regions    = ["North America","APAC","EMEA","India","SEA"]
    stages     = ["Prospecting","Qualification","Proposal",
                  "Negotiation","Closed Won","Closed Lost"]

    industry = np.random.choice(industries, n, p=[0.20,0.15,0.18,0.17,0.20,0.10])
    region   = np.random.choice(regions,    n, p=[0.30,0.25,0.20,0.15,0.10])
    stage    = np.random.choice(stages,     n, p=[0.30,0.25,0.20,0.12,0.07,0.06])

    deal_size           = np.random.lognormal(11.5,1.2,n).clip(10_000,2_000_000).round(-2)
    days_in_stage       = np.random.gamma(3,15,n).clip(1,180).astype(int)
    num_meetings        = np.random.poisson(4,n).clip(0,20)
    email_open_rate     = np.random.beta(2,5,n).round(3)
    champion_identified = np.random.binomial(1,0.45,n)
    competitor_present  = np.random.binomial(1,0.40,n)
    product_fit_score   = np.random.uniform(0.3,1.0,n).round(3)
    account_tenure_yrs  = np.random.exponential(3,n).clip(0,15).round(1)
    prev_purchases      = np.random.poisson(1.5,n).clip(0,10)
    exec_sponsor        = np.random.binomial(1,0.35,n)
    nps_score           = np.random.randint(0,11,n)
    days_since_contact  = np.random.gamma(2,10,n).clip(0,90).astype(int)

    log_odds = (
        -3.5
        + 0.0000008 * deal_size
        + 0.08      * num_meetings
        + 1.2       * champion_identified
        - 0.8       * competitor_present
        + 2.0       * product_fit_score
        + 0.15      * account_tenure_yrs
        + 0.25      * prev_purchases
        + 1.5       * exec_sponsor
        + 0.05      * nps_score
        - 0.02      * days_since_contact
        + 0.3       * email_open_rate
        + np.where(industry=="Technology",        0.4, 0)
        + np.where(industry=="Financial Services",0.3, 0)
        + np.where(region=="North America",       0.2, 0)
        + np.where(stage=="Negotiation",          1.8, 0)
        + np.where(stage=="Proposal",             0.9, 0)
        + np.random.normal(0,0.3,n)
    )
    prob      = 1 / (1 + np.exp(-log_odds))
    converted = (np.random.uniform(size=n) < prob).astype(int)

    return pd.DataFrame({
        "account_id":          [f"ACC-{i:05d}" for i in range(1,n+1)],
        "industry":            industry,
        "region":              region,
        "pipeline_stage":      stage,
        "deal_size":           deal_size,
        "days_in_stage":       days_in_stage.tolist(),
        "num_meetings":        num_meetings.tolist(),
        "email_open_rate":     email_open_rate,
        "champion_identified": champion_identified.tolist(),
        "competitor_present":  competitor_present.tolist(),
        "product_fit_score":   product_fit_score,
        "account_tenure_yrs":  account_tenure_yrs,
        "prev_purchases":      prev_purchases.tolist(),
        "exec_sponsor":        exec_sponsor.tolist(),
        "nps_score":           nps_score.tolist(),
        "days_since_contact":  days_since_contact.tolist(),
        "converted":           converted.tolist(),
"created_at":          pd.Timestamp.now("UTC"),    })

def load_to_bigquery(df):
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.accounts"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"  Loaded {len(df):,} accounts to BigQuery")
    print(f"  Conversion rate: {df['converted'].mean():.1%}")

# ── Run all steps ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Step 1: Creating BigQuery dataset ===")
    create_dataset()

    print("\n=== Step 2: Creating tables ===")
    create_tables()

    print("\n=== Step 3: Generating and loading 5,000 accounts ===")
    df = generate_data(5000)
    load_to_bigquery(df)
    df.to_csv("accounts.csv", index=False)
    print("  CSV saved locally as accounts.csv")

    print("\n=== All done. Verify in BigQuery console: ===")
    print(f"  https://console.cloud.google.com/bigquery?project={PROJECT_ID}")