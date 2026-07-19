"""
Phase 3 — BigQuery SQL Feature Engineering
Creates a view in BigQuery that engineers all model features using pure SQL.
This is the enterprise pattern: features live in the warehouse, not in Python.

PMLE relevance: BigQuery SQL feature engineering is tested directly on the exam.
"""

from google.cloud import bigquery

PROJECT_ID = "labs601-edis"
DATASET_ID = "edis"
client     = bigquery.Client(project=PROJECT_ID)

# ── The feature engineering view ─────────────────────────────────────────────
# This SQL does everything we previously did in pandas:
#   LOG(deal_size)              → compress right tail of deal sizes
#   num_meetings / days+1       → normalise meetings by time elapsed
#   weighted engagement score   → composite signal
#   CASE WHEN for stage weight  → encode pipeline stage as business priority
#
# Saving this as a VIEW means:
#   - Features are always fresh when queried
#   - No duplication of data
#   - Any downstream tool (Python, Looker, Data Studio) reads the same features
#   - The feature logic is versioned and auditable in BigQuery

FEATURE_VIEW_SQL = f"""
CREATE OR REPLACE VIEW `{PROJECT_ID}.{DATASET_ID}.vw_features` AS

WITH base AS (
    SELECT
        account_id,
        industry,
        region,
        pipeline_stage,
        deal_size,
        days_in_stage,
        num_meetings,
        email_open_rate,
        champion_identified,
        competitor_present,
        product_fit_score,
        account_tenure_yrs,
        prev_purchases,
        exec_sponsor,
        nps_score,
        days_since_contact,
        converted,

        -- Feature 1: Log-transform deal size
        -- Why: Deal sizes range from $10K to $2M (right-skewed).
        -- Log compression prevents large deals from dominating the model.
        LOG(deal_size + 1) AS deal_size_log,

        -- Feature 2: Meetings per day in stage
        -- Why: 10 meetings in 5 days signals urgency.
        -- 10 meetings in 90 days signals stagnation. Raw count misses this.
        SAFE_DIVIDE(num_meetings, days_in_stage + 1) AS meetings_per_day,

        -- Feature 3: Composite engagement score
        -- Why: Combines email, meetings, and NPS into one signal.
        -- Weights reflect relative importance from domain knowledge.
        (email_open_rate * 0.3) + 
        (num_meetings * 0.04) + 
        (nps_score * 0.03) AS engagement_score,

        -- Feature 4: Pipeline stage as ordered numeric weight
        -- Why: Negotiation is fundamentally different from Prospecting.
        -- Label encoding preserves this ordinal relationship.
        CASE pipeline_stage
            WHEN 'Prospecting'   THEN 1
            WHEN 'Qualification' THEN 2
            WHEN 'Proposal'      THEN 3
            WHEN 'Negotiation'   THEN 4
            WHEN 'Closed Won'    THEN 5
            WHEN 'Closed Lost'   THEN 0
            ELSE 1
        END AS stage_weight,

        -- Feature 5: Industry risk score
        -- Why: Some industries convert at higher rates historically.
        -- Encodes domain knowledge directly into the feature.
        CASE industry
            WHEN 'Technology'         THEN 0.8
            WHEN 'Financial Services' THEN 0.75
            WHEN 'Healthcare'         THEN 0.65
            WHEN 'Manufacturing'      THEN 0.60
            WHEN 'Retail'             THEN 0.55
            WHEN 'Logistics'          THEN 0.50
            ELSE 0.60
        END AS industry_score,

        -- Feature 6: Contact recency flag
        -- Why: Accounts not contacted in 21+ days are at risk.
        -- Binary flag makes this threshold explicit to the model.
        CASE WHEN days_since_contact > 21 THEN 1 ELSE 0 END AS contact_lapsed,

        -- Feature 7: High value deal flag
        -- Why: Deals above $500K have different sales motion.
        CASE WHEN deal_size > 500000 THEN 1 ELSE 0 END AS high_value_deal,

        -- Feature 8: Champion + exec sponsor combined signal
        -- Why: Having both is multiplicatively stronger than either alone.
        champion_identified * exec_sponsor AS champion_and_sponsor

    FROM `{PROJECT_ID}.{DATASET_ID}.accounts`
)

SELECT * FROM base
"""

# ── Training dataset export query ─────────────────────────────────────────────
# This query reads from the view and formats it for model training.
# In production this would be scheduled as a BigQuery job.

TRAINING_EXPORT_SQL = f"""
SELECT
    account_id,
    deal_size_log,
    days_in_stage,
    num_meetings,
    email_open_rate,
    champion_identified,
    competitor_present,
    product_fit_score,
    account_tenure_yrs,
    prev_purchases,
    exec_sponsor,
    nps_score,
    days_since_contact,
    meetings_per_day,
    engagement_score,
    stage_weight,
    industry_score,
    contact_lapsed,
    high_value_deal,
    champion_and_sponsor,
    converted
FROM `{PROJECT_ID}.{DATASET_ID}.vw_features`
"""

def create_feature_view():
    print("Creating feature engineering view in BigQuery...")
    job = client.query(FEATURE_VIEW_SQL)
    job.result()
    print("View created: labs601-edis.edis.vw_features")
    print("\nFeatures engineered in SQL:")
    features = [
        ("deal_size_log",        "LOG(deal_size+1) — compresses right tail"),
        ("meetings_per_day",     "meetings / (days_in_stage+1) — normalised activity"),
        ("engagement_score",     "weighted email + meetings + NPS composite"),
        ("stage_weight",         "pipeline stage as ordered numeric 0-5"),
        ("industry_score",       "historical conversion rate by industry"),
        ("contact_lapsed",       "binary flag: no contact in 21+ days"),
        ("high_value_deal",      "binary flag: deal size > $500K"),
        ("champion_and_sponsor", "champion * exec_sponsor interaction term"),
    ]
    for name, desc in features:
        print(f"  {name:<25} {desc}")

def export_training_data():
    print("\nExporting training dataset from BigQuery view...")
    df = client.query(TRAINING_EXPORT_SQL).to_dataframe()
    df.to_csv("training_data.csv", index=False)
    print(f"  Rows exported : {len(df):,}")
    print(f"  Features      : {len(df.columns)-2} (excluding account_id and converted)")
    print(f"  Conversion    : {df['converted'].mean():.1%}")
    print(f"  Saved to      : training_data.csv")
    return df

def verify_view():
    print("\nVerifying view with sample query...")
    sample_sql = f"""
    SELECT
        pipeline_stage,
        COUNT(*) as accounts,
        ROUND(AVG(propensity_proxy), 3) as avg_proxy_score
    FROM (
        SELECT
            pipeline_stage,
            (champion_identified * 0.3 + 
             exec_sponsor * 0.25 + 
             product_fit_score * 0.25 + 
             engagement_score * 0.2) as propensity_proxy
        FROM `{PROJECT_ID}.{DATASET_ID}.vw_features`
    )
    GROUP BY pipeline_stage
    ORDER BY avg_proxy_score DESC
    """
    result = client.query(sample_sql).to_dataframe()
    print(result.to_string(index=False))

if __name__ == "__main__":
    print("=== Phase 3: BigQuery SQL Feature Engineering ===\n")
    create_feature_view()
    df = export_training_data()
    verify_view()
    print("\n=== Phase 3 complete ===")
    print("View available at:")
    print(f"https://console.cloud.google.com/bigquery?project={PROJECT_ID}")