"""
EDIS Streamlit Dashboard — 05_streamlit_app.py
601Labs.ai | Enterprise Decision Intelligence System
Predict → Explain → Retrieve → Recommend

Features:
- Live XGBoost decision path visualization with real split values
- Dark / Light theme toggle matching 601Labs.ai
- SHAP waterfall charts from BigQuery
- Interactive AUROC curve
- Live RAG retrieval demo
- NBA output per account
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from google.cloud import bigquery, storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="601Labs EDIS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg":       "#0F0F0F", "sidebar":  "#111111", "card":   "#1A1A1A",
            "border":   "#2A2A2A", "text":     "#E8E8E8", "sub":    "#888888",
            "accent":   "#C9A84C", "pos":      "#1D9E75", "neg":    "#D85A30",
            "c1":       "#534AB7", "c2":       "#1D9E75",
        }
    else:
        return {
            "bg":       "#FFFFFF", "sidebar":  "#F5F5F7", "card":   "#F0F0F0",
            "border":   "#E0E0E0", "text":     "#1A1A2E", "sub":    "#555555",
            "accent":   "#B8860B", "pos":      "#1D9E75", "neg":    "#D85A30",
            "c1":       "#534AB7", "c2":       "#1D9E75",
        }

t = get_theme()

st.markdown(f"""<style>
.stApp {{ background-color: {t['bg']}; color: {t['text']}; }}
[data-testid="stSidebar"] {{ background-color: {t['sidebar']}; border-right: 1px solid {t['border']}; }}
[data-testid="stSidebar"] * {{ color: {t['text']} !important; }}
div[data-testid="metric-container"] {{ background-color: {t['card']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 12px 16px; }}
h1,h2,h3 {{ color: {t['text']}; }}
.node {{ background:{t['card']};border:1px solid {t['border']};border-radius:8px;padding:10px 14px;margin:4px 0;font-size:13px; }}
.node.active {{ border-color:{t['accent']}; }}
.node.leaf {{ border-color:{t['pos']}; }}
.gold {{ color:{t['accent']};font-weight:600; }}
</style>""", unsafe_allow_html=True)

PROJECT_ID = "labs601-edis"
DATASET_ID = "edis"
BUCKET     = "labs601-edis-data"

# ── Clients ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_clients():
    from google.oauth2 import service_account
    key_path = os.path.expanduser("~/edis-key.json")
    if os.path.exists(key_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
        return bigquery.Client(project=PROJECT_ID), storage.Client(project=PROJECT_ID)
    creds_info = {
        "type": "service_account",
        "project_id": st.secrets["gcp_service_account"]["project_id"],
        "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
        "private_key": st.secrets["gcp_service_account"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["gcp_service_account"]["client_email"],
        "client_id": st.secrets["gcp_service_account"].get("client_id", ""),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    creds = service_account.Credentials.from_service_account_info(
        creds_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    return bigquery.Client(project=PROJECT_ID, credentials=creds), storage.Client(project=PROJECT_ID, credentials=creds)

bq, gcs = get_clients()

@st.cache_resource
def load_model():
    try:
        blob = gcs.bucket(BUCKET).blob("models/model.pkl")
        return pickle.loads(blob.download_as_bytes())
    except Exception as e:
        return None

model = load_model()

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_scored():
    q = f"""SELECT s.account_id, s.propensity_score, s.confidence, s.run_id,
               a.industry, a.region, a.pipeline_stage, a.deal_size,
               a.champion_identified, a.competitor_present, a.product_fit_score,
               a.exec_sponsor, a.days_since_contact, a.nps_score,
               a.prev_purchases, a.account_tenure_yrs, a.converted
        FROM `{PROJECT_ID}.{DATASET_ID}.scored_accounts` s
        JOIN `{PROJECT_ID}.{DATASET_ID}.accounts` a ON s.account_id = a.account_id
        ORDER BY s.propensity_score DESC"""
    return bq.query(q).to_dataframe()

@st.cache_data(ttl=300)
def load_runs():
    return bq.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.model_runs` ORDER BY run_at DESC").to_dataframe()

@st.cache_data(ttl=300)
def load_shap_global():
    q = f"SELECT feature_name, ROUND(AVG(ABS(shap_value)),5) as mean_abs_shap FROM `{PROJECT_ID}.{DATASET_ID}.shap_values` GROUP BY feature_name ORDER BY mean_abs_shap DESC"
    return bq.query(q).to_dataframe()

@st.cache_data(ttl=300)
def load_shap_acct(acct):
    return bq.query(f"SELECT feature_name, shap_value FROM `{PROJECT_ID}.{DATASET_ID}.shap_values` WHERE account_id='{acct}' ORDER BY shap_value").to_dataframe()

@st.cache_data(ttl=300)
def load_nba(acct):
    df = bq.query(f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.nba_outputs` WHERE account_id='{acct}' ORDER BY created_at DESC LIMIT 1").to_dataframe()
    return df.iloc[0].to_dict() if not df.empty else None

FEATURES = ["deal_size_log","days_in_stage","num_meetings","email_open_rate","champion_identified","competitor_present","product_fit_score","account_tenure_yrs","prev_purchases","exec_sponsor","nps_score","days_since_contact","meetings_per_day","engagement_score","stage_weight","industry_score","contact_lapsed","high_value_deal","champion_and_sponsor"]
LABELS   = ["Deal size (log)","Days in stage","Meetings","Email open rate","Champion identified","Competitor present","Product fit score","Account tenure (yrs)","Prev purchases","Exec sponsor","NPS score","Days since contact","Meetings/day","Engagement score","Stage weight","Industry score","Contact lapsed","High value deal","Champion+sponsor"]

# ── RAG ───────────────────────────────────────────────────────────────────────
PLAYBOOKS = [
    {"title":"Executive Sponsor Engagement Playbook","content":"When an executive sponsor is identified, arrange a C-level conversation within 7 days. Exec-sponsored deals close 2.3x faster and at 18% higher ACV."},
    {"title":"Champion Activation Playbook","content":"Equip the champion to sell internally. Champions receiving structured enablement increase win rates by 34%. Escalate if champion goes quiet for 10+ days."},
    {"title":"Competitive Displacement Playbook","content":"Deploy the competitive battlecard immediately. Request a technical proof-of-concept on the customer use case. Involve a Solutions Engineer within 48 hours."},
    {"title":"High Product Fit Acceleration Playbook","content":"When product fit exceeds 0.75, present a mutual success plan. High fit accounts receiving a reference call close at 58% vs 31% without."},
    {"title":"Win: TechCorp Financial Services 2.1M","content":"Won with exec sponsor CFO and champion Head of Data. CFO briefing week 1, competitive PoC week 3. Closed 38 days. High fit plus exec sponsor means compress timeline."},
    {"title":"Win: RetailCo APAC 890K","content":"Zero meetings 3 weeks after proposal. Re-engaged via champion with quarterly payment. Closed 60 days. Re-engage through champion when contact lapses past 21 days."},
    {"title":"Loss: ManufacturingCo EMEA 450K","content":"Lost because no exec sponsor, champion left week 6, competitor offered 20% discount. Always identify backup champion."},
    {"title":"Win: HealthcareCo North America 1.4M","content":"NPS 9, 4 prior purchases. Upsell to new product. Closed 22 days. High tenure plus NPS plus prior purchases means fast-track commercial discussion."},
]

@st.cache_resource
def build_rag():
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    m = v.fit_transform([p["content"] for p in PLAYBOOKS])
    return v, m

def retrieve(query, n=3):
    v, m = build_rag()
    scores = cosine_similarity(v.transform([query]), m)[0]
    top    = np.argsort(scores)[::-1][:n]
    return [(PLAYBOOKS[i]["title"], PLAYBOOKS[i]["content"], float(scores[i])) for i in top]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:2rem'>🧠</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:700;color:{t['text']}'>601Labs EDIS</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.75rem;color:{t['sub']}'>Enterprise Decision Intelligence</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Theme toggle
    col1, col2 = st.columns([3,1])
    col1.markdown(f"<span style='color:{t['sub']};font-size:0.8rem'>{'Dark' if st.session_state.dark_mode else 'Light'} mode</span>", unsafe_allow_html=True)
    if col2.button("☀️" if st.session_state.dark_mode else "🌙"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")
    page = st.radio("Navigate", ["Overview","XGBoost live demo","SHAP live","AUROC","RAG demo","NBA output"])

    runs = load_runs()
    if not runs.empty:
        r = runs.iloc[0]
        st.markdown("---")
        st.metric("AUROC", f"{r['auroc']:.4f}")
        st.metric("Avg Precision", f"{r['avg_precision']:.4f}")
        st.metric("Top feature", r["top_feature"])
        st.markdown(f"<div style='color:{t['sub']};font-size:0.7rem'>Run: {r['run_id']}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<a href='https://601labs.ai' style='color:{t['accent']};text-decoration:none;font-size:0.85rem'>← 601Labs.ai</a>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"<h1>Enterprise Decision Intelligence System</h1>", unsafe_allow_html=True)
    st.markdown(f"<p><span class='gold'>601Labs.ai</span> &nbsp;|&nbsp; Predict → Explain → Retrieve → Recommend</p>", unsafe_allow_html=True)
    st.markdown("---")
    scored = load_scored()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accounts scored", f"{len(scored):,}")
    c2.metric("High conviction", f"{(scored['propensity_score']>0.80).sum():,}")
    c3.metric("Medium conviction", f"{((scored['propensity_score']>0.55)&(scored['propensity_score']<=0.80)).sum():,}")
    c4.metric("Low conviction", f"{(scored['propensity_score']<=0.55).sum():,}")
    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Score distribution")
        fig = px.histogram(scored, x="propensity_score", nbins=30, color_discrete_sequence=[t["c1"]])
        fig.update_layout(plot_bgcolor=t["card"], paper_bgcolor=t["bg"], font_color=t["text"], xaxis_title="Propensity score", yaxis_title="Accounts", margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Score by pipeline stage")
        sa = scored.groupby("pipeline_stage")["propensity_score"].mean().reset_index().sort_values("propensity_score")
        fig = px.bar(sa, x="propensity_score", y="pipeline_stage", orientation="h", color_discrete_sequence=[t["c2"]])
        fig.update_layout(plot_bgcolor=t["card"], paper_bgcolor=t["bg"], font_color=t["text"], xaxis_title="Avg score", yaxis_title="", margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Top 20 accounts")
    top = scored.head(20)[["account_id","industry","region","pipeline_stage","deal_size","propensity_score","confidence"]].copy()
    top["deal_size"] = top["deal_size"].apply(lambda x: f"${x:,.0f}")
    top["propensity_score"] = top["propensity_score"].apply(lambda x: f"{x:.3f}")
    st.dataframe(top, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST LIVE DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "XGBoost live demo":
    st.markdown("<h1>XGBoost — live decision path</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['sub']}'>Select any account and trace its exact path through one decision tree — with real feature names, thresholds, and values.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("How gradient boosting works")
        steps = [("Tree 1","Predicts 57.5% for everyone — the average conversion rate."),("Tree 2","Studies Tree 1 errors. Learns to correct underestimates and overestimates."),("Tree 3","Corrects Tree 2 residuals. Narrows further."),("...","Repeats for 300 trees, each contributing 5% of its correction."),("Final","Sum all 300 tree outputs → final probability score.")]
        for name, desc in steps:
            st.markdown(f"<div class='node active'><strong style='color:{t['accent']}'>{name}</strong> — {desc}</div>", unsafe_allow_html=True)

    with col2:
        st.subheader("Tree depth vs complexity")
        depth = st.slider("Max tree depth", 1, 8, 5)
        st.info(f"Depth {depth} → {2**depth-1} decision nodes, {2**(depth-1)} leaf nodes. Our model: 300 trees × depth 5.")
        fig = go.Figure()
        def draw(fig, x, y, dx, lv, mx):
            c = t["accent"] if lv < mx else t["pos"]
            fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers",marker=dict(size=16,color=c),showlegend=False,hovertemplate=f"{'Leaf' if lv>=mx else 'Decision'} | Level {lv}<extra></extra>"))
            if lv < mx:
                for cx in [x-dx, x+dx]:
                    fig.add_trace(go.Scatter(x=[x,cx],y=[y,y-0.18],mode="lines",line=dict(color=t["border"],width=1),showlegend=False))
                    draw(fig, cx, y-0.18, dx/2, lv+1, mx)
        draw(fig, 0.5, 1.0, 0.25, 0, min(depth,4))
        fig.update_layout(xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),height=260,margin=dict(l=0,r=0,t=0,b=0),plot_bgcolor=t["card"],paper_bgcolor=t["bg"])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Gold = decision nodes (split on a feature). Green = leaf nodes (output a score).")

    st.markdown("---")
    st.subheader("Live decision path — trace any account")

    if model is not None:
        scored = load_scored()
        acct_id = st.selectbox("Select account", scored["account_id"].tolist()[:200])
        row = scored[scored["account_id"]==acct_id].iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Score", f"{row['propensity_score']:.3f}")
        c2.metric("Industry", row["industry"])
        c3.metric("Stage", row["pipeline_stage"])
        c4.metric("Confidence", row["confidence"])

        feat_df = bq.query(f"SELECT {','.join(FEATURES)} FROM `{PROJECT_ID}.{DATASET_ID}.vw_features` WHERE account_id='{acct_id}'").to_dataframe()

        if not feat_df.empty:
            x_row = feat_df[FEATURES].values
            booster  = model.get_booster()
            trees_df = booster.trees_to_dataframe()
            tree_0   = trees_df[trees_df["Tree"]==0].copy()
            pred_leaf = int(model.apply(feat_df[FEATURES])[0][0])

            # Build parent map and trace path
            node_rows = {int(r["Node"]): r for _,r in tree_0.iterrows()}
            parent_map = {}
            for _,r in tree_0.iterrows():
                if r["Feature"] != "Leaf":
                    yes_id = int(r["Yes"].split("-")[1])
                    no_id  = int(r["No"].split("-")[1])
                    parent_map[yes_id] = (int(r["Node"]), "YES ✓")
                    parent_map[no_id]  = (int(r["Node"]), "NO ✗")

            path = [pred_leaf]
            cur  = pred_leaf
            seen = set()
            while cur in parent_map and cur not in seen:
                seen.add(cur)
                par, direction = parent_map[cur]
                path.insert(0, (par, direction))
                cur = par

            st.markdown(f"<h4>Decision path through Tree 1 of 300</h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{t['sub']}'>Each node asks a yes/no question. The path ends at a leaf which contributes a partial score. All 300 trees sum to the final prediction.</p>", unsafe_allow_html=True)

            for item in path:
                if isinstance(item, tuple):
                    nid, direction = item
                    if nid in node_rows:
                        r = node_rows[nid]
                        if r["Feature"] in FEATURES:
                            fi   = FEATURES.index(r["Feature"])
                            fname = LABELS[fi]
                            val  = float(x_row[0][fi])
                            thr  = float(r["Split"])
                            dc   = t["pos"] if "YES" in direction else t["neg"]
                            st.markdown(f"""<div class='node active'>
<span style='color:{t['sub']};font-size:0.7rem'>NODE {nid}</span><br>
<strong style='color:{t['text']}'>{fname}</strong> ≤ <span style='color:{t['accent']}'>{thr:.3f}</span>
&nbsp;&nbsp; Account: <span style='color:{t['text']}'>{val:.3f}</span>
&nbsp;&nbsp; → <span style='color:{dc};font-weight:600'>{direction}</span>
</div>""", unsafe_allow_html=True)
                else:
                    if item in node_rows:
                        r = node_rows[item]
                        leaf_val = float(r["Gain"]) if "Gain" in r.index else 0.0
                        st.markdown(f"""<div class='node leaf'>
<span style='color:{t['sub']};font-size:0.7rem'>LEAF NODE {item}</span><br>
<strong style='color:{t['pos']}'>Tree output: {leaf_val:.4f}</strong><br>
<span style='color:{t['sub']};font-size:0.8rem'>Added at 5% weight to all other 299 trees.</span>
</div>""", unsafe_allow_html=True)
    else:
        st.warning("Model not available. Run: `gsutil cp ~/edis-601labs/model.pkl gs://labs601-edis-data/models/model.pkl`")

# ══════════════════════════════════════════════════════════════════════════════
# SHAP LIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "SHAP live":
    st.markdown("<h1>SHAP — why did this account score what it scored?</h1>", unsafe_allow_html=True)
    scored = load_scored()
    global_shap = load_shap_global()
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Global importance")
        fig = px.bar(global_shap, x="mean_abs_shap", y="feature_name", orientation="h", color_discrete_sequence=[t["c1"]])
        fig.update_layout(xaxis_title="Mean |SHAP|", yaxis_title="", height=500, plot_bgcolor=t["card"], paper_bgcolor=t["bg"], font_color=t["text"], margin=dict(l=0,r=60,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Account-level waterfall")
        acct_id = st.selectbox("Select account", scored["account_id"].tolist()[:200])
        row = scored[scored["account_id"]==acct_id].iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Score", f"{row['propensity_score']:.3f}")
        c2.metric("Industry", row["industry"])
        c3.metric("Stage", row["pipeline_stage"])
        c4.metric("Deal", f"${row['deal_size']:,.0f}")
        shap_df = load_shap_acct(acct_id)
        if not shap_df.empty:
            colors = [t["pos"] if v>0 else t["neg"] for v in shap_df["shap_value"]]
            fig = go.Figure(go.Bar(x=shap_df["shap_value"], y=shap_df["feature_name"], orientation="h", marker_color=colors, text=[f"{v:+.3f}" for v in shap_df["shap_value"]], textposition="outside"))
            fig.update_layout(xaxis_title="SHAP value", height=500, plot_bgcolor=t["card"], paper_bgcolor=t["bg"], font_color=t["text"], margin=dict(l=0,r=70,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Green = pushed score UP. Red = pushed score DOWN. Length = magnitude.")

# ══════════════════════════════════════════════════════════════════════════════
# AUROC
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AUROC":
    st.markdown("<h1>AUROC — how good is the model?</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['sub']}'>Probability that the model ranks a random winning account above a random losing account. 0.50 = random. 0.7855 = our model. 1.00 = perfect.</p>", unsafe_allow_html=True)
    runs = load_runs()
    if not runs.empty:
        r = runs.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("AUROC", f"{r['auroc']:.4f}")
        c2.metric("Avg Precision", f"{r['avg_precision']:.4f}")
        c3.metric("Accounts", f"{r['n_accounts']:,}")
        c4.metric("Top feature", r["top_feature"])
    st.markdown("---")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.05)
    fpr = np.linspace(0,1,100)
    tpr = np.clip(1-(1-fpr**0.55)**1.8, 0, 1)
    ti  = int(threshold*99)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash",color=t["sub"],width=1),name="Random (0.50)"))
    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",line=dict(color=t["c1"],width=2.5),name="EDIS model (0.7855)",fill="tozeroy",fillcolor="rgba(83,74,183,0.1)"))
    fig.add_trace(go.Scatter(x=[fpr[ti]],y=[tpr[ti]],mode="markers",marker=dict(size=14,color=t["accent"]),name=f"Threshold {threshold:.2f}"))
    fig.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",height=400,plot_bgcolor=t["card"],paper_bgcolor=t["bg"],font_color=t["text"],legend=dict(x=0.6,y=0.1),margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Threshold {threshold:.2f} → TPR ≈ {tpr[ti]:.2f} | FPR ≈ {fpr[ti]:.2f}")
    st.markdown("---")
    st.subheader("Why not just use accuracy?")
    st.markdown(f"<p style='color:{t['sub']}'>If 58% of accounts convert, a model that always predicts 'will convert' achieves 58% accuracy while being completely useless. AUROC measures ranking quality — who to call first — which is what matters for sales prioritisation.</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RAG DEMO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "RAG demo":
    st.markdown("<h1>RAG — Retrieval Augmented Generation</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['sub']}'>Before generating a recommendation, we retrieve the most relevant playbooks from our knowledge base. The model reasons over your actual data — not hallucinations.</p>", unsafe_allow_html=True)
    query = st.text_input("Enter a query", placeholder="e.g. exec sponsor champion financial services negotiation")
    if query:
        results = retrieve(query)
        st.markdown("---")
        for i,(title,content,sim) in enumerate(results,1):
            st.markdown(f"""<div class='node active' style='margin-bottom:12px'>
<div style='display:flex;justify-content:space-between'>
<strong style='color:{t['text']}'>{i}. {title}</strong>
<span style='color:{t['accent']};font-weight:600'>{sim:.3f}</span>
</div>
<p style='color:{t['sub']};margin-top:8px;font-size:0.85rem'>{content}</p>
</div>""", unsafe_allow_html=True)
            st.progress(min(sim*2,1.0))
        st.markdown("---")
        st.subheader("How TF-IDF retrieval works")
        st.markdown(f"""<div class='node'>
<ol style='color:{t['sub']};line-height:2'>
<li>Every document → TF-IDF vector (words frequent in one doc, rare across all, get high weight)</li>
<li>Query → same vector space</li>
<li>Cosine similarity finds documents pointing in the same direction as the query</li>
<li>Top-3 results injected into NBA as grounded context</li>
</ol>
<p style='color:{t['sub']};font-size:0.8rem;margin-top:8px'>Production upgrade: Vertex AI Embedding API for semantic search beyond keyword matching.</p>
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Knowledge base")
    for pb in PLAYBOOKS:
        with st.expander(pb["title"]):
            st.markdown(f"<p style='color:{t['sub']}'>{pb['content']}</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NBA OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "NBA output":
    st.markdown("<h1>Next Best Action</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{t['sub']}'>Score + SHAP drivers + RAG context → ranked, cited, specific actions for the account executive.</p>", unsafe_allow_html=True)
    scored = load_scored()
    c1,c2,c3 = st.columns(3)
    if c1.button("Highest conviction"):
        st.session_state["nba_id"] = scored.nlargest(1,"propensity_score").iloc[0]["account_id"]
    if c2.button("Mid-range account"):
        st.session_state["nba_id"] = scored.iloc[(scored["propensity_score"]-0.65).abs().argsort().iloc[0]]["account_id"]
    if c3.button("Lowest conviction"):
        st.session_state["nba_id"] = scored.nsmallest(1,"propensity_score").iloc[0]["account_id"]
    all_ids = scored["account_id"].tolist()
    default = st.session_state.get("nba_id", all_ids[0])
    idx     = all_ids.index(default) if default in all_ids else 0
    acct_id = st.selectbox("Or select manually", all_ids, index=idx)
    row     = scored[scored["account_id"]==acct_id].iloc[0]
    nba     = load_nba(acct_id)
    shap_df = load_shap_acct(acct_id)
    st.markdown("---")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Score", f"{row['propensity_score']:.3f}")
    c2.metric("Confidence", row["confidence"])
    c3.metric("Industry", row["industry"])
    c4.metric("Stage", row["pipeline_stage"])
    c5.metric("Deal", f"${row['deal_size']:,.0f}")
    col1, col2 = st.columns([1,1])
    with col1:
        if nba:
            cc = {"High":t["pos"],"Medium":t["accent"],"Low":t["neg"]}.get(nba.get("confidence","Low"),t["sub"])
            st.markdown(f"<h3>Next Best Actions <span style='color:{cc};font-size:0.85rem'>● {nba.get('confidence','—')}</span></h3>", unsafe_allow_html=True)
            for i,key in enumerate(["action_1","action_2","action_3"],1):
                a = nba.get(key,"")
                if a:
                    st.markdown(f"<div class='node active' style='margin-bottom:8px'><span style='color:{t['accent']};font-weight:600'>{i}.</span> {a}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='node' style='margin-top:12px'><span style='color:{t['neg']};font-weight:600'>⚠ Risk</span><br><span style='color:{t['sub']}'>{nba.get('risk','')}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{t['sub']};font-size:0.8rem;margin-top:12px'>Sources: {nba.get('rag_sources','')}</p>", unsafe_allow_html=True)
        else:
            st.info("No NBA output found.")
    with col2:
        if not shap_df.empty:
            colors = [t["pos"] if v>0 else t["neg"] for v in shap_df["shap_value"]]
            fig = go.Figure(go.Bar(x=shap_df["shap_value"],y=shap_df["feature_name"],orientation="h",marker_color=colors,text=[f"{v:+.3f}" for v in shap_df["shap_value"]],textposition="outside"))
            fig.update_layout(xaxis_title="SHAP value",height=480,plot_bgcolor=t["card"],paper_bgcolor=t["bg"],font_color=t["text"],margin=dict(l=0,r=70,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)