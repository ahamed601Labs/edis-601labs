"""
Phase 5 - RAG + Next Best Action Engine
Retrieves relevant playbooks using TF-IDF, generates NBA recommendations,
writes all outputs to BigQuery nba_outputs table.

RAG from first principles:
    Instead of asking an LLM to answer from general knowledge,
    we first RETRIEVE the most relevant documents from our own
    knowledge base, then pass them as context to the reasoning layer.
    Result: grounded, cited, trustworthy recommendations.

NBA from first principles:
    Combines three signals:
        1. Propensity score (XGBoost) - WHO is likely to convert
        2. SHAP values - WHY they scored that way
        3. RAG context - WHAT has worked in similar situations
    Output: ranked, specific, explainable actions for the AE
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ID = "labs601-edis"
DATASET_ID = "edis"
client     = bigquery.Client(project=PROJECT_ID)

FEATURE_LABELS = [
    "Deal size (log)", "Days in stage", "Meetings", "Email open rate",
    "Champion identified", "Competitor present", "Product fit score",
    "Account tenure (yrs)", "Previous purchases", "Exec sponsor", "NPS score",
    "Days since contact", "Meetings per day", "Engagement score",
    "Stage weight", "Industry score", "Contact lapsed",
    "High value deal", "Champion and sponsor",
]

# ── Knowledge base ────────────────────────────────────────────────────────────
# These are the documents RAG retrieves from.
# In production: loaded from Cloud Storage or a vector database.
# For this demo: hardcoded playbooks and win/loss stories.

KNOWLEDGE_BASE = [
    {
        "id": "exec_sponsor_playbook",
        "title": "Executive Sponsor Engagement Playbook",
        "content": (
            "When an executive sponsor is identified, arrange a C-level conversation "
            "within 7 days. Prepare a value briefing tied to strategic priorities and "
            "a 3-year ROI model. Exec-sponsored deals close 2.3x faster and at 18% "
            "higher ACV. Follow up with an executive summary within 24 hours of the call."
        ),
    },
    {
        "id": "champion_playbook",
        "title": "Champion Activation Playbook",
        "content": (
            "Equip the champion to sell internally. Provide a business case template, "
            "competitive comparison sheet, and internal presentation deck. Schedule "
            "weekly champion sync. Champions receiving structured enablement increase "
            "win rates by 34%. Escalate immediately if champion goes quiet for 10+ days."
        ),
    },
    {
        "id": "competitor_playbook",
        "title": "Competitive Displacement Playbook",
        "content": (
            "Deploy the competitive battlecard immediately when a competitor is present. "
            "Request a technical proof-of-concept on the customer's actual use case. "
            "Involve a Solutions Engineer within 48 hours. Never attack the competitor "
            "directly — position around customer outcomes instead."
        ),
    },
    {
        "id": "product_fit_playbook",
        "title": "High Product Fit Acceleration Playbook",
        "content": (
            "When product fit score exceeds 0.75, present a mutual success plan with "
            "milestones. Connect customer with a reference account in same industry. "
            "High fit accounts receiving a reference call close at 58% vs 31% without. "
            "Push for verbal commitment within 14 days of fit confirmation."
        ),
    },
    {
        "id": "win_techcorp",
        "title": "Win story: TechCorp Financial Services 2.1M deal",
        "content": (
            "Won with exec sponsor CFO and champion Head of Data. CFO briefing week 1, "
            "competitive PoC week 3, reference call week 4. Closed in 38 days from "
            "Negotiation stage. Product fit 0.91. NPS 9. "
            "Lesson: high fit plus exec sponsor means compress timeline aggressively."
        ),
    },
    {
        "id": "win_retailco",
        "title": "Win story: RetailCo APAC 890K deal",
        "content": (
            "Zero meetings for 3 weeks after proposal. Re-engaged via champion with "
            "quarterly payment model. Exec sponsor introduced week 5. Closed in 60 days. "
            "Lesson: when days since contact exceeds 21, re-engage through champion "
            "not direct outreach. Flexible commercials unlock stalled deals."
        ),
    },
    {
        "id": "loss_manufacturingco",
        "title": "Loss analysis: ManufacturingCo EMEA 450K lost to competitor",
        "content": (
            "Lost because no exec sponsor, champion left week 6, competitor offered "
            "20% discount. Champion went quiet 12 days before loss. "
            "Lesson: always identify backup champion. "
            "Price-sensitive deals need commercial flexibility ready in advance."
        ),
    },
    {
        "id": "win_healthcareco",
        "title": "Win story: HealthcareCo North America 1.4M deal",
        "content": (
            "High NPS account score 9 with 4 previous purchases. Upsell into new "
            "product line. No competitor present. Led with expansion ROI not acquisition "
            "pitch. Skipped discovery and went straight to mutual success plan. "
            "Closed in 22 days. Lesson: high tenure plus high NPS plus prior purchases "
            "means fast-track commercial discussion."
        ),
    },
]


# ── Build TF-IDF vector store ─────────────────────────────────────────────────
# TF-IDF from first principles:
#   TF  = term frequency = how often a word appears in THIS document
#   IDF = inverse document frequency = how rare the word is ACROSS all documents
#   TF-IDF = TF x IDF = words that are frequent in one doc but rare overall
#            get high scores. Common words like "the", "is" get low scores.
#
# Cosine similarity measures the angle between two vectors.
# Vectors pointing in the same direction = similar content = high score.
# Perpendicular vectors = completely different content = score of 0.

def build_rag_store(kb):
    corpus     = [doc["content"] for doc in kb]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix     = vectorizer.fit_transform(corpus)
    print(f"  RAG store built: {len(kb)} documents, {matrix.shape[1]:,} TF-IDF terms")
    return vectorizer, matrix


def retrieve(query, vectorizer, matrix, kb, n=3):
    q_vec   = vectorizer.transform([query])
    scores  = cosine_similarity(q_vec, matrix)[0]
    top_n   = np.argsort(scores)[::-1][:n]
    return [
        {
            "title":      kb[i]["title"],
            "content":    kb[i]["content"],
            "similarity": round(float(scores[i]), 3),
        }
        for i in top_n
    ]


# ── NBA engine ────────────────────────────────────────────────────────────────
def generate_nba(acct, shap_row, vectorizer, matrix, kb):
    pairs    = list(zip(FEATURE_LABELS, shap_row))
    top5     = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:5]
    pos_keys = " ".join([k for k, v in top5 if v > 0][:3])
    rag_q    = f"{acct['industry']} {acct['pipeline_stage']} {pos_keys}"
    ctx_docs = retrieve(rag_q, vectorizer, matrix, kb)

    actions = []
    risks   = []

    if int(acct.get("exec_sponsor", 0)) == 1:
        actions.append(
            "Schedule C-level briefing within 7 days. "
            "Prepare 3-year ROI model. [Exec Sponsor Playbook]"
        )
    if int(acct.get("champion_identified", 0)) == 1:
        actions.append(
            "Equip champion with business case deck and co-present "
            "at next internal review. [Champion Activation Playbook]"
        )
    if int(acct.get("competitor_present", 0)) == 1:
        actions.append(
            "Deploy competitive battlecard and request technical PoC "
            "within 48 hrs. [Competitive Displacement Playbook]"
        )
        risks.append(
            "Competitor active - monitor for end-of-quarter price discount. "
            "[ManufacturingCo Loss Analysis]"
        )
    if float(acct.get("product_fit_score", 0)) > 0.75 and len(actions) < 3:
        actions.append(
            f"Present mutual success plan and arrange reference call - "
            f"product fit {float(acct['product_fit_score']):.2f}. "
            f"[Product Fit Playbook]"
        )
    if float(acct.get("days_since_contact", 0)) > 20 and len(actions) < 3:
        actions.append(
            f"Re-engage immediately via champion - "
            f"{int(acct['days_since_contact'])} days since last contact. "
            f"[RetailCo Win Story]"
        )
    if not actions:
        actions.append(
            "Review account profile and schedule discovery call. "
            "No strong positive signals detected."
        )
    if not risks:
        risks.append(
            f"Monitor champion engagement - escalate if silent more than 10 days. "
            f"Currently {int(acct.get('days_since_contact', 0))} days since contact."
        )

    score = float(acct["propensity_score"])
    conf  = "High" if score > 0.80 else ("Medium" if score > 0.55 else "Low")

    return {
        "action_1":    actions[0] if len(actions) > 0 else "",
        "action_2":    actions[1] if len(actions) > 1 else "",
        "action_3":    actions[2] if len(actions) > 2 else "",
        "risk":        risks[0],
        "confidence":  conf,
        "rag_sources": " | ".join([d["title"] for d in ctx_docs]),
    }


# ── Load scored accounts + SHAP from BigQuery ─────────────────────────────────
def load_scored_with_shap():
    print("Loading scored accounts from BigQuery...")
    scores_query = f"""
        SELECT
            s.account_id, s.run_id, s.propensity_score, s.confidence,
            a.industry, a.region, a.pipeline_stage, a.deal_size,
            a.champion_identified, a.competitor_present,
            a.product_fit_score, a.exec_sponsor,
            a.days_since_contact, a.nps_score,
            a.prev_purchases, a.account_tenure_yrs
        FROM `{PROJECT_ID}.{DATASET_ID}.scored_accounts` s
        JOIN `{PROJECT_ID}.{DATASET_ID}.accounts` a
        ON s.account_id = a.account_id
        ORDER BY s.propensity_score DESC
    """
    scored_df = client.query(scores_query).to_dataframe()
    print(f"  Scored accounts loaded: {len(scored_df):,}")

    print("Loading SHAP values from BigQuery...")
    shap_query = f"""
        SELECT account_id, feature_name, shap_value
        FROM `{PROJECT_ID}.{DATASET_ID}.shap_values`
    """
    shap_df = client.query(shap_query).to_dataframe()
    print(f"  SHAP rows loaded: {len(shap_df):,}")

    return scored_df, shap_df


def get_shap_row(account_id, shap_df):
    acct_shap = shap_df[shap_df["account_id"] == account_id]
    shap_dict = dict(zip(acct_shap["feature_name"], acct_shap["shap_value"]))
    return [shap_dict.get(feat, 0.0) for feat in FEATURE_LABELS]


# ── Persist NBA outputs to BigQuery ───────────────────────────────────────────
def persist_nba(nba_rows):
    print(f"\nWriting {len(nba_rows):,} NBA outputs to BigQuery...")
    now = datetime.now(timezone.utc).isoformat()
    rows = [{**r, "created_at": now} for r in nba_rows]
    for i in range(0, len(rows), 500):
        errors = client.insert_rows_json(
            f"{PROJECT_ID}.{DATASET_ID}.nba_outputs",
            rows[i:i+500]
        )
        if errors:
            print(f"  Warning batch {i}: {errors}")
    print(f"  Done. {len(rows):,} NBA outputs written.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 5: RAG + NBA Engine ===\n")

    # Build RAG store
    print("Building RAG knowledge base...")
    vectorizer, matrix = build_rag_store(KNOWLEDGE_BASE)

    # Load data from BigQuery
    scored_df, shap_df = load_scored_with_shap()
    run_id = scored_df["run_id"].iloc[0]

    # Generate NBA for all scored accounts
    print(f"\nGenerating NBA recommendations for {len(scored_df):,} accounts...")
    nba_rows = []
    for _, acct in scored_df.iterrows():
        shap_row = get_shap_row(acct["account_id"], shap_df)
        result   = generate_nba(acct, shap_row, vectorizer, matrix, KNOWLEDGE_BASE)
        nba_rows.append({
            "account_id": acct["account_id"],
            "run_id":     run_id,
            **result,
        })

    # Persist to BigQuery
    persist_nba(nba_rows)

    # Print 3 showcase accounts
    print("\n=== SHOWCASE: 3 ACCOUNTS ===")

    high = scored_df.nlargest(1, "propensity_score").iloc[0]
    mid  = scored_df.iloc[(scored_df["propensity_score"] - 0.65).abs().argsort().iloc[0]]
    risk = scored_df[
        (scored_df["competitor_present"] == 1) &
        (scored_df["champion_identified"] == 0)
    ].nsmallest(1, "propensity_score").iloc[0]

    for label, acct in [
        ("HIGH CONVICTION", high),
        ("MID RANGE",       mid),
        ("RISK ACCOUNT",    risk),
    ]:
        shap_row = get_shap_row(acct["account_id"], shap_df)
        nba      = generate_nba(acct, shap_row, vectorizer, matrix, KNOWLEDGE_BASE)
        pairs    = list(zip(FEATURE_LABELS, shap_row))
        top5     = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:5]

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {acct['account_id']} | {acct['industry']} | {acct['pipeline_stage']}")
        print(f"  Score: {acct['propensity_score']:.3f} | Deal: ${acct['deal_size']:,.0f}")
        print(f"\n  TOP SHAP DRIVERS")
        for name, val in top5:
            arrow = "UP  " if val > 0 else "DOWN"
            print(f"    {arrow} {name:<25} {val:+.3f}")
        print(f"\n  RAG SOURCES")
        for src in nba["rag_sources"].split(" | "):
            print(f"    - {src}")
        print(f"\n  NEXT BEST ACTIONS")
        for i, key in enumerate(["action_1", "action_2", "action_3"], 1):
            if nba[key]:
                print(f"    {i}. {nba[key]}")
        print(f"\n  RISK    : {nba['risk']}")
        print(f"  CONFIDENCE: {nba['confidence']}")

    print(f"\n=== Phase 5 complete ===")
    print("Verify: BigQuery -> edis -> nba_outputs -> Preview tab")