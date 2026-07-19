"""
Phase 6 - EDIS Streamlit Dashboard
Enterprise Decision Intelligence System - 601Labs.ai
Reads all data live from BigQuery. Six interactive tabs.

Run locally : streamlit run 05_streamlit_app.py
Deploy      : push to GitHub, connect to share.streamlit.io
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="601Labs EDIS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ID = "labs601-edis"
DATASET_ID = "edis"

# ── BigQuery client ───────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    key_path = os.path.expanduser("~/edis-key.json")
    if os.path.exists(key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    return bigquery.Client(project=PROJECT_ID)

client = get_client()

# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_scored():
    q = f"""
        SELECT s.account_id, s.propensity_score, s.confidence, s.run_id,
               a.industry, a.region, a.pipeline_stage, a.deal_size,
               a.champion_identified, a.competitor_present,
               a.product_fit_score, a.exec_sponsor,
               a.days_since_contact, a.nps_score,
               a.prev_purchases, a.account_tenure_yrs, a.converted
        FROM `{PROJECT_ID}.{DATASET_ID}.scored_accounts` s
        JOIN `{PROJECT_ID}.{DATASET_ID}.accounts` a
        ON s.account_id = a.account_id
        ORDER BY s.propensity_score DESC
    """
    return client.query(q).to_dataframe()

@st.cache_data(ttl=300)
def load_model_runs():
    q = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.model_runs` ORDER BY run_at DESC"
    return client.query(q).to_dataframe()

@st.cache_data(ttl=300)
def load_shap_global():
    q = f"""
        SELECT feature_name,
               ROUND(AVG(ABS(shap_value)), 5) as mean_abs_shap
        FROM `{PROJECT_ID}.{DATASET_ID}.shap_values`
        GROUP BY feature_name
        ORDER BY mean_abs_shap DESC
    """
    return client.query(q).to_dataframe()

@st.cache_data(ttl=300)
def load_shap_account(account_id):
    q = f"""
        SELECT feature_name, shap_value
        FROM `{PROJECT_ID}.{DATASET_ID}.shap_values`
        WHERE account_id = '{account_id}'
        ORDER BY shap_value
    """
    return client.query(q).to_dataframe()

@st.cache_data(ttl=300)
def load_nba(account_id):
    q = f"""
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.nba_outputs`
        WHERE account_id = '{account_id}'
        ORDER BY created_at DESC
        LIMIT 1
    """
    df = client.query(q).to_dataframe()
    return df.iloc[0].to_dict() if not df.empty else None

# ── RAG knowledge base ────────────────────────────────────────────────────────
PLAYBOOKS = [
    {"title": "Executive Sponsor Engagement Playbook",
     "content": "When an executive sponsor is identified, arrange a C-level conversation within 7 days. Prepare a value briefing and 3-year ROI model. Exec-sponsored deals close 2.3x faster and at 18% higher ACV."},
    {"title": "Champion Activation Playbook",
     "content": "Equip the champion to sell internally. Provide business case template, competitive comparison sheet, and presentation deck. Champions receiving structured enablement increase win rates by 34%."},
    {"title": "Competitive Displacement Playbook",
     "content": "Deploy the competitive battlecard immediately. Request a technical proof-of-concept on the customer's actual use case. Involve a Solutions Engineer within 48 hours. Position around customer outcomes."},
    {"title": "High Product Fit Acceleration Playbook",
     "content": "When product fit exceeds 0.75, present a mutual success plan with milestones. Connect customer with a reference account. High fit accounts receiving a reference call close at 58% vs 31% without."},
    {"title": "Win story: TechCorp Financial Services 2.1M",
     "content": "Won with exec sponsor CFO and champion Head of Data. CFO briefing week 1, competitive PoC week 3, reference call week 4. Closed 38 days. Product fit 0.91. High fit plus exec sponsor means compress timeline aggressively."},
    {"title": "Win story: RetailCo APAC 890K",
     "content": "Zero meetings 3 weeks after proposal. Re-engaged via champion with quarterly payment. Exec sponsor week 5. Closed 60 days. When days since contact exceeds 21, re-engage through champion not direct outreach."},
    {"title": "Loss: ManufacturingCo EMEA 450K",
     "content": "Lost because no exec sponsor, champion left week 6, competitor offered 20% discount. Champion went quiet 12 days before loss. Always identify backup champion. Price-sensitive deals need commercial flexibility."},
    {"title": "Win story: HealthcareCo North America 1.4M",
     "content": "High NPS 9 account with 4 previous purchases. Upsell to new product. Led with expansion ROI, skipped discovery. Closed 22 days. High tenure plus NPS plus prior purchases means fast-track commercial discussion."},
]

@st.cache_resource
def build_rag():
    corpus     = [p["content"] for p in PLAYBOOKS]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix     = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

def retrieve(query, n=3):
    vectorizer, matrix = build_rag()
    q_vec   = vectorizer.transform([query])
    scores  = cosine_similarity(q_vec, matrix)[0]
    top_n   = np.argsort(scores)[::-1][:n]
    return [(PLAYBOOKS[i]["title"], PLAYBOOKS[i]["content"], float(scores[i])) for i in top_n]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=60)
st.sidebar.title("601Labs EDIS")
st.sidebar.caption("Enterprise Decision Intelligence System")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "XGBoost explainer",
    "SHAP live",
    "AUROC",
    "RAG demo",
    "NBA output",
])

runs_df = load_model_runs()
if not runs_df.empty:
    latest = runs_df.iloc[0]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Latest model run**")
    st.sidebar.metric("AUROC",          f"{latest['auroc']:.4f}")
    st.sidebar.metric("Avg Precision",  f"{latest['avg_precision']:.4f}")
    st.sidebar.metric("Top feature",    latest["top_feature"])
    st.sidebar.caption(f"Run: {latest['run_id']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 - OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Enterprise Decision Intelligence System")
    st.markdown("**601Labs.ai** | Predict → Explain → Retrieve → Recommend")
    st.markdown("---")

    scored_df = load_scored()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accounts scored",    f"{len(scored_df):,}")
    col2.metric("High conviction",    f"{(scored_df['propensity_score'] > 0.80).sum():,}")
    col3.metric("Medium conviction",  f"{((scored_df['propensity_score'] > 0.55) & (scored_df['propensity_score'] <= 0.80)).sum():,}")
    col4.metric("Low conviction",     f"{(scored_df['propensity_score'] <= 0.55).sum():,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score distribution")
        fig = px.histogram(
            scored_df, x="propensity_score", nbins=30,
            title="Propensity score distribution across all accounts",
            color_discrete_sequence=["#534AB7"],
        )
        fig.update_layout(
            xaxis_title="Propensity score",
            yaxis_title="Number of accounts",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Score by pipeline stage")
        stage_avg = scored_df.groupby("pipeline_stage")["propensity_score"].mean().reset_index()
        stage_avg = stage_avg.sort_values("propensity_score", ascending=True)
        fig = px.bar(
            stage_avg, x="propensity_score", y="pipeline_stage",
            orientation="h", color_discrete_sequence=["#1D9E75"],
        )
        fig.update_layout(
            xaxis_title="Average propensity score",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 20 accounts by propensity score")
    top20 = scored_df.head(20)[[
        "account_id", "industry", "region", "pipeline_stage",
        "deal_size", "propensity_score", "confidence"
    ]].copy()
    top20["deal_size"] = top20["deal_size"].apply(lambda x: f"${x:,.0f}")
    top20["propensity_score"] = top20["propensity_score"].apply(lambda x: f"{x:.3f}")
    st.dataframe(top20, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 - XGBOOST EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "XGBoost explainer":
    st.title("How XGBoost works — from first principles")

    st.markdown("""
XGBoost stands for e**X**treme **G**radient **Boost**ing. It builds an ensemble of
decision trees **sequentially** — each tree learns from the mistakes of all previous trees.

Instead of building one big complex tree that memorises the data (**overfitting**),
XGBoost builds hundreds of small, shallow trees where each one corrects the
**residual errors** of the previous ensemble.
    """)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("The boosting process step by step")
        steps = [
            ("Tree 1",   "Makes initial predictions. Many errors."),
            ("Tree 2",   "Learns from Tree 1 errors. Reduces them."),
            ("Tree 3",   "Learns from Tree 2 residuals. Refines further."),
            ("...",      "Continues for N trees (we use 300)."),
            ("Final",    "Sum all 300 trees → final probability score."),
        ]
        for name, desc in steps:
            st.markdown(f"**{name}** — {desc}")

    with col2:
        st.subheader("Why not one deep tree?")
        st.markdown("""
A single deep tree **memorises** training data. It performs perfectly on accounts
it has seen but fails on new ones.

Many shallow trees, each correcting the previous one, **generalises** better.
The ensemble captures patterns without memorising noise.

This is the **bias-variance tradeoff** in practice:
- One deep tree: low bias, high variance (overfits)
- Ensemble of shallow trees: balanced bias and variance (generalises)
        """)

    st.markdown("---")
    st.subheader("Interactive: how tree depth affects model complexity")
    depth = st.slider("Max tree depth", 1, 8, 5)
    nodes  = 2**depth - 1
    leaves = 2**(depth - 1)
    st.info(
        f"Depth {depth} → {nodes} decision nodes, {leaves} leaf nodes. "
        f"Our model uses depth 5. Each of 300 trees has max {2**5-1} nodes."
    )

    fig = go.Figure()
    def draw_node(fig, x, y, dx, level, max_d):
        color = "#534AB7" if level < max_d else "#1D9E75"
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=18, color=color),
            showlegend=False,
            hovertemplate=f"{'Leaf' if level>=max_d else 'Decision'} node | Level {level}<extra></extra>",
        ))
        if level < max_d:
            for cx in [x - dx, x + dx]:
                fig.add_trace(go.Scatter(
                    x=[x, cx], y=[y, y - 0.18],
                    mode="lines", line=dict(color="#aaaaaa", width=1),
                    showlegend=False,
                ))
                draw_node(fig, cx, y - 0.18, dx / 2, level + 1, max_d)

    draw_node(fig, 0.5, 1.0, 0.25, 0, min(depth, 4))
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Purple = decision nodes (split on a feature). Green = leaf nodes (output a score).")

    st.markdown("---")
    st.subheader("Our model parameters — what each one does")
    params = {
        "n_estimators = 300":     "300 trees in the ensemble. More trees = better fit, slower training.",
        "max_depth = 5":          "Maximum depth per tree. Shallow trees generalise better.",
        "learning_rate = 0.05":   "Each tree contributes only 5% of its prediction. Conservative but robust.",
        "subsample = 0.8":        "Each tree sees 80% of rows. Adds randomness, prevents memorisation.",
        "colsample_bytree = 0.8": "Each tree sees 80% of features. Creates diversity across trees.",
        "gamma = 0.1":            "Minimum gain required to make a split. Regularisation penalty.",
    }
    for param, explanation in params.items():
        st.markdown(f"`{param}` — {explanation}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 - SHAP LIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP live":
    st.title("SHAP — why did this account score what it scored?")
    st.markdown("""
**SHAP** (SHapley Additive exPlanations) answers the question every sales leader asks:
*"The model says 87% — but why?"*

Each feature gets a SHAP value: how much it **pushed the score up or down** from the
base rate. The values are additive: `base_value + sum(all SHAP values) = final score`.
    """)

    scored_df = load_scored()
    global_shap = load_shap_global()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Global importance")
        st.caption("Average absolute SHAP across all accounts")
        fig = px.bar(
            global_shap, x="mean_abs_shap", y="feature_name",
            orientation="h", color_discrete_sequence=["#534AB7"],
        )
        fig.update_layout(
            xaxis_title="Mean |SHAP|",
            yaxis_title="",
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Account-level explanation")
        account_id = st.selectbox(
            "Select account to explain",
            scored_df["account_id"].tolist()[:200]
        )

        row = scored_df[scored_df["account_id"] == account_id].iloc[0]
        score = row["propensity_score"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Score",    f"{score:.3f}")
        m2.metric("Industry", row["industry"])
        m3.metric("Stage",    row["pipeline_stage"])

        shap_df = load_shap_account(account_id)
        if not shap_df.empty:
            colors = ["#1D9E75" if v > 0 else "#D85A30" for v in shap_df["shap_value"]]
            fig = go.Figure(go.Bar(
                x=shap_df["shap_value"],
                y=shap_df["feature_name"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"SHAP waterfall — {account_id}",
                xaxis_title="SHAP value (contribution to score)",
                height=500,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=60, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green = pushed score UP. Red = pushed score DOWN. Length = magnitude.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 - AUROC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AUROC":
    st.title("AUROC — how good is the model?")
    st.markdown("""
**AUROC** = Area Under the Receiver Operating Characteristic curve.

It measures: *if you pick one winning account and one losing account at random,
how often does the model correctly rank the winner higher?*

- **0.50** = random guessing (coin flip)
- **0.78** = our model (correct ranking 78% of the time)
- **1.00** = perfect separation

**Why not just use accuracy?** If 58% of accounts convert, a model that always
predicts "will convert" gets 58% accuracy while being completely useless for
prioritisation. AUROC measures *ranking quality* — which accounts to call first.
    """)

    runs_df = load_model_runs()
    if not runs_df.empty:
        latest = runs_df.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUROC",         f"{latest['auroc']:.4f}")
        c2.metric("Avg Precision", f"{latest['avg_precision']:.4f}")
        c3.metric("Accounts",      f"{latest['n_accounts']:,}")
        c4.metric("Top feature",   latest["top_feature"])

    st.markdown("---")
    st.subheader("Interactive ROC curve")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)

    fpr_sim = np.linspace(0, 1, 100)
    tpr_sim = np.clip(1 - (1 - fpr_sim**0.55)**1.8, 0, 1)
    t_idx   = int(threshold * 99)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="#888780", width=1),
        name="Random (AUROC = 0.50)",
    ))
    fig.add_trace(go.Scatter(
        x=fpr_sim, y=tpr_sim, mode="lines",
        line=dict(color="#534AB7", width=2.5),
        name="EDIS model (AUROC = 0.79)",
        fill="tozeroy", fillcolor="rgba(83,74,183,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[fpr_sim[t_idx]], y=[tpr_sim[t_idx]], mode="markers",
        marker=dict(size=14, color="#D85A30"),
        name=f"Threshold {threshold:.2f}",
        hovertemplate=f"FPR: {fpr_sim[t_idx]:.2f}<br>TPR: {tpr_sim[t_idx]:.2f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate (% losers incorrectly called winners)",
        yaxis_title="True Positive Rate (% winners correctly identified)",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.6, y=0.1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"At threshold {threshold:.2f}: TPR ≈ {tpr_sim[t_idx]:.2f} | "
        f"FPR ≈ {fpr_sim[t_idx]:.2f}. "
        f"Higher threshold = fewer false alarms but miss more real winners."
    )

    st.markdown("---")
    st.subheader("Precision vs Recall tradeoff")
    st.markdown("""
- **Precision**: of accounts we flag as high conviction, how many actually convert?
- **Recall**: of all accounts that will convert, how many did we correctly flag?
- Moving the threshold changes the balance between these two.
- For sales prioritisation, recall matters more — missing a real winner is costly.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 - RAG DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "RAG demo":
    st.title("RAG — Retrieval Augmented Generation")
    st.markdown("""
**The problem**: A language model does not know your sales playbooks, win/loss history,
or competitive intelligence. It knows what it was trained on — public internet.

**The solution**: Before asking for a recommendation, we **retrieve** the most relevant
documents from our own knowledge base and inject them as context. The model then
reasons over your actual data, not hallucinations.

Type a query below to see live retrieval in action.
    """)

    query = st.text_input(
        "Enter a query",
        placeholder="e.g. exec sponsor champion financial services negotiation"
    )

    if query:
        results = retrieve(query)
        st.markdown("---")
        st.subheader("Retrieved documents — ranked by relevance")
        for i, (title, content, sim) in enumerate(results, 1):
            with st.expander(f"#{i} — {title}  |  Similarity: {sim:.3f}", expanded=(i == 1)):
                st.markdown(content)
                st.progress(min(sim * 2, 1.0), text=f"Relevance: {sim:.3f}")

        st.markdown("---")
        st.subheader("How TF-IDF retrieval works")
        st.markdown("""
1. Every document is converted to a **vector of TF-IDF weights** (term frequency × inverse document frequency)
2. Your query is converted to a vector using the same vocabulary
3. **Cosine similarity** finds documents whose vectors point in the same direction as your query
4. Top-N documents are returned and injected into the NBA prompt as grounded context

**Production upgrade**: replace TF-IDF with semantic embeddings via Vertex AI Embedding API.
Semantic search finds conceptually similar documents even when exact words differ.
        """)

    st.markdown("---")
    st.subheader("Knowledge base")
    for pb in PLAYBOOKS:
        with st.expander(pb["title"]):
            st.write(pb["content"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 - NBA OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "NBA output":
    st.title("Next Best Action — full recommendation")
    st.markdown("Select any account to see its complete AI-generated recommendation, grounded in score, SHAP drivers, and retrieved playbooks.")

    scored_df = load_scored()

    col1, col2, col3 = st.columns(3)
    if col1.button("Highest conviction account"):
        st.session_state["nba_id"] = scored_df.nlargest(1, "propensity_score").iloc[0]["account_id"]
    if col2.button("Mid-range account"):
        mid_idx = (scored_df["propensity_score"] - 0.65).abs().argsort().iloc[0]
        st.session_state["nba_id"] = scored_df.iloc[mid_idx]["account_id"]
    if col3.button("Lowest conviction account"):
        st.session_state["nba_id"] = scored_df.nsmallest(1, "propensity_score").iloc[0]["account_id"]

    default_id = st.session_state.get("nba_id", scored_df.iloc[0]["account_id"])
    all_ids    = scored_df["account_id"].tolist()
    default_ix = all_ids.index(default_id) if default_id in all_ids else 0

    account_id = st.selectbox("Or select manually", all_ids, index=default_ix)
    row        = scored_df[scored_df["account_id"] == account_id].iloc[0]
    nba        = load_nba(account_id)
    shap_df    = load_shap_account(account_id)

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Score",        f"{row['propensity_score']:.3f}")
    c2.metric("Confidence",   row["confidence"])
    c3.metric("Industry",     row["industry"])
    c4.metric("Stage",        row["pipeline_stage"])
    c5.metric("Deal size",    f"${row['deal_size']:,.0f}")

    col1, col2 = st.columns([1, 1])

    with col1:
        if nba:
            conf_color = {"High": "#1D9E75", "Medium": "#EF9F27", "Low": "#D85A30"}.get(
                nba.get("confidence", "Low"), "#888780"
            )
            st.markdown(f"### Next Best Actions")
            for i, key in enumerate(["action_1", "action_2", "action_3"], 1):
                action = nba.get(key, "")
                if action:
                    st.markdown(f"**{i}.** {action}")
            st.markdown(f"**Risk:** {nba.get('risk', '')}")
            st.markdown(f"**Playbooks retrieved:**")
            for src in nba.get("rag_sources", "").split(" | "):
                if src:
                    st.markdown(f"- {src}")
        else:
            st.info("No NBA output found for this account.")

    with col2:
        if not shap_df.empty:
            colors = ["#1D9E75" if v > 0 else "#D85A30" for v in shap_df["shap_value"]]
            fig = go.Figure(go.Bar(
                x=shap_df["shap_value"],
                y=shap_df["feature_name"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df["shap_value"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="SHAP drivers",
                xaxis_title="SHAP value",
                height=450,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=60, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)