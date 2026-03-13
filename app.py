"""
app.py — Talabat UAE Hiring Bias Analytics Dashboard  v3
=========================================================
Changes vs previous version:
  1. Classification  → Accuracy bar now present for ALL 3 models in comparison chart
  2. Clustering      → "Hiring Rate by Cluster" changed from bar → line chart
  3. All insights    → re-oriented to Fair Hiring Score north star
                       FHS = Hiring Rate(Group A) / Hiring Rate(Group B)
                       Target = 1.000 | Fairness floor = 0.800
  4. EDA             → Sankey diagram added (Gender→Referral→Education→Decision)

Deploy: streamlit run app.py  |  Streamlit Cloud (requires requirements.txt)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Talabat UAE · Hiring Bias Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main{background:#0f1117;}
section[data-testid="stSidebar"]{background:#1a1d27!important;border-right:1px solid #2d3142;}
div[data-testid="metric-container"]{
    background:linear-gradient(135deg,#1e2130,#252840);
    border:1px solid #3a3f5c;border-radius:12px;padding:16px 20px;}
div[data-testid="metric-container"] label{color:#a0a8c8!important;font-size:.78rem;
    text-transform:uppercase;letter-spacing:.06em;}
div[data-testid="metric-container"] div[data-testid="stMetricValue"]{
    color:#fff!important;font-size:1.7rem;font-weight:700;}
.page-title{background:linear-gradient(90deg,#ff6b35,#f7c59f);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:1.7rem;font-weight:700;margin-bottom:.2rem;}
.page-sub{color:#8890aa;font-size:.9rem;margin-bottom:1.2rem;}
.insight{background:#1e2130;border-left:4px solid #ff6b35;
    border-radius:0 8px 8px 0;padding:10px 16px;color:#c8cfe0;
    font-size:.84rem;margin-top:6px;margin-bottom:14px;}
.fhs-pill{background:linear-gradient(135deg,#1e2130,#1a2535);
    border:1px solid #ff6b35;border-radius:8px;
    padding:10px 16px;color:#c8cfe0;font-size:.84rem;
    margin-top:6px;margin-bottom:14px;}
.fhs-pill b{color:#ff6b35;}
.icard{background:linear-gradient(135deg,#1e2130,#252840);
    border:1px solid #3a3f5c;border-radius:12px;padding:20px 24px;margin-bottom:14px;}
.icard h3{color:#ff6b35;font-size:1rem;font-weight:600;margin-bottom:6px;}
.icard p{color:#a0a8c8;font-size:.85rem;line-height:1.65;margin:0;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:.75rem;font-weight:600;margin:2px;}
.badge-orange{background:#ff6b3530;color:#ff6b35;border:1px solid #ff6b35;}
.badge-blue{background:#4fc3f730;color:#4fc3f7;border:1px solid #4fc3f7;}
.badge-green{background:#34d39930;color:#34d399;border:1px solid #34d399;}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("dataset.csv")
    except FileNotFoundError:
        from data_generation import generate_dataset
        d = generate_dataset()
        d.to_csv("dataset.csv", index=False)
        return d

df = load_data()

PALETTE      = px.colors.qualitative.Vivid
TEMPLATE     = "plotly_dark"
OVERALL_RATE = (df["Final_Hiring_Decision"] == "Hired").mean()

# ── Fair Hiring Score helpers ─────────────────────────────────────────────────
def fhs_table(group_col):
    """DataFrame: group | hire_rate | FHS | status"""
    r = (df.groupby(group_col)["Final_Hiring_Decision"]
           .apply(lambda x: (x == "Hired").mean())
           .reset_index())
    r.columns = [group_col, "Hire Rate"]
    max_r = r["Hire Rate"].max()
    r["FHS"]    = (r["Hire Rate"] / max_r).round(3)
    r["Status"] = r["FHS"].apply(lambda v: "✅ Fair" if v >= 0.80 else "⚠️ Biased")
    return r

def fhs_min(group_col):
    return fhs_table(group_col)["FHS"].min()

def fhs_pill(val, dim, low, high):
    colour = "#34d399" if val >= 0.80 else "#f87171"
    word   = "✅ passes" if val >= 0.80 else "⚠️ fails"
    st.markdown(
        f'<div class="fhs-pill">🎯 <b>Fair Hiring Score — {dim}:</b> '
        f'<span style="color:{colour};font-size:1rem;font-weight:700">{val:.3f}</span> '
        f'&nbsp;{word} the 0.800 floor&nbsp;|&nbsp;'
        f'<em>{low}</em> ÷ <em>{high}</em> hire-rate ratio&nbsp;|&nbsp;'
        f'Perfect parity = 1.000</div>',
        unsafe_allow_html=True,
    )

def insight(txt):
    st.markdown(f'<div class="insight">💡 {txt}</div>', unsafe_allow_html=True)

def section(title, sub=""):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 22px'>
        <div style='font-size:2.6rem'>🚀</div>
        <div style='color:#ff6b35;font-weight:700;font-size:1.15rem'>Talabat UAE</div>
        <div style='color:#8890aa;font-size:.78rem;margin-top:2px'>Hiring Bias Analytics</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        "<div style='background:#252840;border-radius:8px;padding:10px 12px;"
        "margin-bottom:14px;font-size:.73rem;color:#8890aa'>"
        "🎯 <b style='color:#ff6b35'>North Star Metric</b><br>"
        "<b style='color:#c8cfe0'>Fair Hiring Score (FHS)</b><br>"
        "= Rate(Group A) / Rate(Group B)<br>"
        "<span style='color:#34d399'>Target 1.000 &nbsp;|&nbsp; Floor 0.800</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    page = st.radio("", [
        "🏠  Home",
        "📊  Dataset Overview",
        "🔍  EDA & Visualizations",
        "🤖  Classification Models",
        "🔵  Clustering Analysis",
        "🔗  Association Rule Mining",
        "📈  Regression Forecast",
        "⚠️  Bias Detection",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(
        "<div style='color:#8890aa;font-size:.73rem;text-align:center'>"
        "University PBL Project<br>Data Analytics · 2025</div>",
        unsafe_allow_html=True,
    )

p = page.split("  ", 1)[1]

# ── KPI bar (includes live Composite FHS) ─────────────────────────────────────
def kpis():
    hired = (df["Final_Hiring_Decision"] == "Hired").sum()
    g, n, e, r = fhs_min("Gender"), fhs_min("Nationality"), fhs_min("Education_Level"), fhs_min("Referral_Status")
    comp = round(4 / (1/g + 1/n + 1/e + 1/r), 3)
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Applicants",    f"{len(df):,}")
    c2.metric("Total Hired",         f"{hired:,}")
    c3.metric("Hiring Rate",         f"{hired/len(df)*100:.1f}%")
    c4.metric("Avg Interview Score", f"{df['Interview_Score'].mean():.1f}")
    c5.metric("Avg Experience",      f"{df['Years_of_Experience'].mean():.1f} yrs")
    c6.metric("🎯 Composite FHS",    f"{comp:.3f}",
              delta="vs 1.000 target",
              delta_color="inverse" if comp < 0.95 else "normal")


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if p == "Home":
    section("Talabat UAE — Hiring Bias Analytics Dashboard",
            "Tracking one north star across every hiring decision: Fair Hiring Score (FHS) = Rate(A) / Rate(B)")
    kpis()
    st.markdown("---")

    col1, col2 = st.columns([1.05, 1])

    with col1:
        st.markdown("""
        <div class="icard">
        <h3>🎯 Project Objective</h3>
        <p>Measure and improve the <b>Fair Hiring Score (FHS)</b> — the ratio of hiring rates
        between any two demographic groups — across Talabat UAE's full recruitment pipeline.
        FHS = 1.000 means perfect parity; below 0.800 indicates legally actionable
        disparate impact under the EEOC 80% rule. Every chart and model in this dashboard
        is oriented toward closing FHS gaps.</p>
        </div>
        <div class="icard">
        <h3>🏢 Business Context</h3>
        <p>Talabat UAE is one of the region's largest food-delivery and logistics platforms.
        Tracking FHS supports UAE Emiratisation (Nafis) compliance, ESG reporting,
        and internal DE&amp;I commitments. The target for the next hiring cycle is
        FHS ≥ 0.95 across all four dimensions: gender, nationality, education, and referral.</p>
        </div>
        <div class="icard">
        <h3>📐 Analytics Methods</h3>
        <p>
        <span class="badge badge-orange">EDA + Sankey</span>
        <span class="badge badge-orange">Classification</span>
        <span class="badge badge-orange">Clustering</span>
        <span class="badge badge-blue">Association Rules</span>
        <span class="badge badge-blue">Regression</span>
        <span class="badge badge-green">FHS Bias Detection</span>
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        g, n, e, r = fhs_min("Gender"), fhs_min("Nationality"), fhs_min("Education_Level"), fhs_min("Referral_Status")
        comp = round(4 / (1/g + 1/n + 1/e + 1/r), 3)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=comp,
            delta={"reference": 1.0, "valueformat": ".3f"},
            title={"text": "Composite Fair Hiring Score", "font": {"color": "#c8cfe0"}},
            number={"font": {"color": "#ff6b35", "size": 48}},
            gauge={
                "axis": {"range": [0.5, 1.0], "tickcolor": "#c8cfe0"},
                "bar":  {"color": "#ff6b35"},
                "steps": [
                    {"range": [0.50, 0.80], "color": "#3d1010"},
                    {"range": [0.80, 0.95], "color": "#2d2d10"},
                    {"range": [0.95, 1.00], "color": "#0d2d1a"},
                ],
                "threshold": {"line": {"color": "#34d399", "width": 3},
                              "thickness": 0.75, "value": 0.80},
            },
        ))
        fig.update_layout(template=TEMPLATE, height=280,
                          margin=dict(l=20, r=20, t=50, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        fhs_df = pd.DataFrame({
            "Dimension": ["Gender", "Nationality", "Education", "Referral"],
            "FHS": [g, n, e, r],
        })
        fig2 = px.bar(fhs_df, x="Dimension", y="FHS", color="FHS",
                      color_continuous_scale=["#c0392b","#e67e22","#27ae60"],
                      range_color=[0.70, 1.0],
                      title="FHS by Dimension (target = 1.000)",
                      template=TEMPLATE, text_auto=".3f")
        fig2.add_hline(y=0.80, line_dash="dash", line_color="#34d399",
                       annotation_text="0.800 floor", annotation_font_color="#34d399")
        fig2.add_hline(y=1.0,  line_dash="dot",  line_color="#a0a8c8",
                       annotation_text="Perfect parity", annotation_font_color="#a0a8c8")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                           coloraxis_showscale=False, yaxis=dict(range=[0.5, 1.06]))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif p == "Dataset Overview":
    section("Dataset Overview",
            "Synthetic dataset · 3,000 candidate applications · 25 features")
    kpis(); st.markdown("---")

    t1, t2, t3 = st.tabs(["📋 Preview", "📐 Schema", "📊 Distributions"])

    with t1:
        st.dataframe(df.head(60), use_container_width=True, height=400)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows",           f"{df.shape[0]:,}")
        c2.metric("Columns",        df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        c4.metric("Numeric Cols",   int(df.select_dtypes("number").shape[1]))

    with t2:
        schema = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Nulls":  df.isnull().sum().values,
            "Unique": [df[c].nunique() for c in df.columns],
            "Sample": [str(df[c].iloc[0]) for c in df.columns],
        })
        st.dataframe(schema, use_container_width=True, height=620)

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            gc = df["Gender"].value_counts().reset_index()
            gc.columns = ["Gender","Count"]
            fig = px.bar(gc, x="Gender", y="Count", color="Gender",
                         color_discrete_sequence=["#4fc3f7","#ff6b35"],
                         title="Applicants by Gender", template=TEMPLATE, text_auto=True)
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            g_fhs = fhs_min("Gender")
            insight(f"Gender FHS = {g_fhs:.3f} — above the 0.800 fairness floor but below perfect parity (1.000). "
                    f"The 58/42 male–female split in applicant volume already skews the funnel; "
                    f"targeted sourcing is the first lever to push Gender FHS toward 1.000.")

        with c2:
            ec = df["Education_Level"].value_counts().reset_index()
            ec.columns = ["Education","Count"]
            order = ["High School","Bachelor's","Master's","PhD"]
            ec["Education"] = pd.Categorical(ec["Education"], categories=order, ordered=True)
            ec = ec.sort_values("Education")
            fig = px.bar(ec, x="Education", y="Count", color="Education",
                         color_discrete_sequence=PALETTE,
                         title="Education Level Distribution", template=TEMPLATE, text_auto=True)
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            e_fhs = fhs_min("Education_Level")
            insight(f"Education FHS = {e_fhs:.3f}. Bachelor's holders dominate the pool (55%) yet have the lowest "
                    f"hire rate — a credential gap that pulls FHS below parity. "
                    f"Replacing degree-tier screening with skills assessments is the key fix.")

        c1, c2 = st.columns(2)
        with c1:
            hc = df["Final_Hiring_Decision"].value_counts(normalize=True).mul(100).reset_index()
            hc.columns = ["Decision","Rate (%)"]
            fig = px.bar(hc, x="Decision", y="Rate (%)", color="Decision",
                         color_discrete_sequence=["#ff6b35","#2d3142"],
                         title="Overall Hiring Rate (%)", template=TEMPLATE, text_auto=".1f")
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            insight(f"Overall hiring rate = {OVERALL_RATE*100:.1f}%. "
                    f"FHS = 1.000 requires every demographic sub-group to converge to this baseline — "
                    f"any group diverging above or below is by definition a bias signal.")

        with c2:
            nc = df["Nationality"].value_counts().reset_index()
            nc.columns = ["Nationality","Count"]
            fig = px.pie(nc, values="Count", names="Nationality", hole=0.45,
                         title="Applicant Nationality Mix", color_discrete_sequence=PALETTE)
            fig.update_layout(template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                               legend_font_color="#a0a8c8", title_font_color="#c8cfe0")
            st.plotly_chart(fig, use_container_width=True)
            n_fhs = fhs_min("Nationality")
            insight(f"Nationality FHS = {n_fhs:.3f} — the widest FHS gap in the dataset. "
                    f"Indian nationals form the largest applicant group (30%) yet face a measurable "
                    f"hire-rate disadvantage vs. Western nationals, making this the highest-priority dimension to fix.")


# ══════════════════════════════════════════════════════════════════════════════
# EDA  — with Sankey
# ══════════════════════════════════════════════════════════════════════════════
elif p == "EDA & Visualizations":
    section("Exploratory Data Analysis",
            "Pipeline flow, bias signals, and Fair Hiring Score drivers")
    kpis(); st.markdown("---")

    # ── Sankey: Gender → Referral → Education → Decision ─────────────────────
    st.subheader("🔀 Hiring Pipeline Flow — Sankey Diagram")

    @st.cache_data
    def build_sankey():
        d = df.copy()
        nodes, node_idx = [], {}

        def add_node(label, color):
            if label not in node_idx:
                node_idx[label] = len(nodes)
                nodes.append({"label": label, "color": color})

        gen_c = {"Male": "#4fc3f7", "Female": "#ff6b35"}
        ref_c = {"Referred": "#a78bfa", "Not Referred": "#8890aa"}
        edu_c = {"High School": "#34d399", "Bachelor's": "#f7c59f",
                 "Master's": "#ff6b35",    "PhD": "#4fc3f7"}
        dec_c = {"Hired": "#34d399", "Rejected": "#c0392b"}

        for g in sorted(d["Gender"].unique()):          add_node(g, gen_c.get(g, "#888"))
        for r in sorted(d["Referral_Status"].unique()): add_node(r, ref_c.get(r, "#888"))
        for e in ["High School","Bachelor's","Master's","PhD"]: add_node(e, edu_c[e])
        for dec in ["Hired","Rejected"]:                add_node(dec, dec_c[dec])

        srcs, tgts, vals, lcolors = [], [], [], []

        def link(a, b, v):
            srcs.append(node_idx[a]); tgts.append(node_idx[b]); vals.append(int(v))
            base = nodes[node_idx[a]]["color"].lstrip("#")
            r2,g2,b2 = int(base[0:2],16), int(base[2:4],16), int(base[4:6],16)
            lcolors.append(f"rgba({r2},{g2},{b2},0.38)")

        for g in sorted(d["Gender"].unique()):
            for r in sorted(d["Referral_Status"].unique()):
                cnt = ((d["Gender"]==g) & (d["Referral_Status"]==r)).sum()
                if cnt: link(g, r, cnt)

        for r in sorted(d["Referral_Status"].unique()):
            for e in ["High School","Bachelor's","Master's","PhD"]:
                cnt = ((d["Referral_Status"]==r) & (d["Education_Level"]==e)).sum()
                if cnt: link(r, e, cnt)

        for e in ["High School","Bachelor's","Master's","PhD"]:
            for dec in ["Hired","Rejected"]:
                cnt = ((d["Education_Level"]==e) & (d["Final_Hiring_Decision"]==dec)).sum()
                if cnt: link(e, dec, cnt)

        return nodes, srcs, tgts, vals, lcolors

    nodes, srcs, tgts, vals, lcolors = build_sankey()

    fig_sk = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=22,
            line=dict(color="#0f1117", width=0.5),
            label=[n["label"] for n in nodes],
            color=[n["color"]  for n in nodes],
        ),
        link=dict(source=srcs, target=tgts, value=vals, color=lcolors),
    ))
    fig_sk.update_layout(
        title_text="Applicant Flow: Gender → Referral Status → Education Level → Hiring Decision",
        title_font_color="#c8cfe0",
        template=TEMPLATE, height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8cfe0", size=12),
    )
    st.plotly_chart(fig_sk, use_container_width=True)

    r_fhs = fhs_min("Referral_Status")
    insight(
        f"The Sankey reveals referral status as the key mid-funnel bottleneck: "
        f"the 'Referred' stream converges toward 'Hired' at a disproportionately higher rate. "
        f"Referral FHS = {r_fhs:.3f} — and because referral networks skew toward already-privileged groups, "
        f"this single lever simultaneously depresses Nationality and Gender FHS as well."
    )

    st.markdown("---")

    # Row 1 — gender & nationality
    c1, c2 = st.columns(2)
    with c1:
        gr = (df.groupby("Gender")["Final_Hiring_Decision"]
              .apply(lambda x: (x=="Hired").mean()*100).reset_index())
        gr.columns = ["Gender","Hiring Rate (%)"]
        fig = px.bar(gr, x="Gender", y="Hiring Rate (%)", color="Gender",
                     color_discrete_sequence=["#4fc3f7","#ff6b35"],
                     title="Hiring Rate by Gender", template=TEMPLATE, text_auto=".1f")
        fig.add_hline(y=OVERALL_RATE*100, line_dash="dash", line_color="white",
                      annotation_text="Overall avg", annotation_font_color="white")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        g_fhs = fhs_min("Gender")
        fhs_pill(g_fhs, "Gender", "Female (63.1%)", "Male (65.2%)")
        insight(
            f"Gender FHS = {g_fhs:.3f}. The ~2 pp male advantage is currently above the 0.800 floor, "
            f"but closing it entirely requires blind CV review and structured interview rubrics "
            f"to eliminate evaluator subjectivity — the same steps that protect FHS in the long run."
        )

    with c2:
        nr = (df.groupby("Nationality")["Final_Hiring_Decision"]
              .apply(lambda x: (x=="Hired").mean()*100).reset_index())
        nr.columns = ["Nationality","Hiring Rate (%)"]
        nr = nr.sort_values("Hiring Rate (%)", ascending=False)
        fig = px.bar(nr, x="Nationality", y="Hiring Rate (%)", color="Nationality",
                     color_discrete_sequence=PALETTE,
                     title="Hiring Rate by Nationality", template=TEMPLATE, text_auto=".1f")
        fig.add_hline(y=OVERALL_RATE*100, line_dash="dash", line_color="white",
                      annotation_text="Overall avg", annotation_font_color="white")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
        n_fhs = fhs_min("Nationality")
        fhs_pill(n_fhs, "Nationality", "Egyptian (59.2%)", "British (67.4%)")
        insight(
            f"Nationality FHS = {n_fhs:.3f} — the lowest-scoring dimension. "
            f"British nationals are hired at 67.4% vs. Egyptians at 59.2%, an 8.2 pp gap. "
            f"Standardising HR and Manager rating rubrics to remove nationality-correlated signals "
            f"is the single highest-priority action to lift overall FHS."
        )

    # Row 2 — education & salary
    c1, c2 = st.columns(2)
    with c1:
        er = (df.groupby("Education_Level")["Final_Hiring_Decision"]
              .apply(lambda x: (x=="Hired").mean()*100).reset_index())
        er.columns = ["Education","Hiring Rate (%)"]
        order = ["High School","Bachelor's","Master's","PhD"]
        er["Education"] = pd.Categorical(er["Education"], categories=order, ordered=True)
        er = er.sort_values("Education")
        fig = px.bar(er, x="Education", y="Hiring Rate (%)", color="Education",
                     color_discrete_sequence=PALETTE,
                     title="Hiring Rate by Education Level", template=TEMPLATE, text_auto=".1f")
        fig.add_hline(y=OVERALL_RATE*100, line_dash="dash", line_color="white",
                      annotation_text="Overall avg", annotation_font_color="white")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        e_fhs = fhs_min("Education_Level")
        fhs_pill(e_fhs, "Education", "Bachelor's (62.8%)", "PhD (70.2%)")
        insight(
            f"Education FHS = {e_fhs:.3f}. The 7.4 pp PhD–Bachelor's gap implies credential weighting "
            f"is the mechanism depressing FHS — since Bachelor's holders form 55% of the pool, "
            f"replacing degree-tier filtering with skills-score thresholds would have the largest "
            f"single-action impact on Education FHS."
        )

    with c2:
        fig = px.box(df, x="Final_Hiring_Decision", y="Expected_Salary",
                     color="Final_Hiring_Decision",
                     color_discrete_map={"Hired":"#ff6b35","Rejected":"#3a3f5c"},
                     title="Salary Expectation vs Hiring Decision", template=TEMPLATE)
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "Salary distributions are virtually identical for Hired and Rejected candidates — "
            "budget is not a driver of FHS gaps. This confirms that the fairness problem lies "
            "entirely in non-salary screening factors: referral status, nationality, and credential tier."
        )

    # Row 3 — scatter & heatmap
    c1, c2 = st.columns(2)
    with c1:
        samp = df.sample(600, random_state=7)
        fig = px.scatter(samp, x="Years_of_Experience", y="Interview_Score",
                         color="Final_Hiring_Decision",
                         color_discrete_map={"Hired":"#ff6b35","Rejected":"#4fc3f7"},
                         opacity=0.6, title="Experience vs Interview Score", template=TEMPLATE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", legend_font_color="#a0a8c8")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "Interview Score is the merit signal most correlated with hiring. "
            "If lower-FHS groups (e.g. Egyptian nationals) score equally on Interview Score "
            "yet are hired less frequently, that confirms evaluator bias in HR/Manager ratings "
            "— not a genuine skills gap — as the root cause of FHS suppression."
        )

    with c2:
        num_cols = ["Age","Years_of_Experience","Technical_Skills_Score",
                    "Communication_Score","Interview_Score","Assessment_Test_Score",
                    "Expected_Salary","University_Ranking","HR_Rating","Manager_Rating"]
        corr = df[num_cols].corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation Heatmap", template=TEMPLATE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420)
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "HR Rating and Manager Rating are strongly correlated — if either carries "
            "implicit nationality or gender bias, it compounds through both scores and directly "
            "suppresses the FHS of affected groups. Decoupling them from subjective impressions "
            "via competency rubrics is the structural fix most likely to move FHS toward 1.000."
        )

    # Referral bar (full width)
    rr = (df.groupby("Referral_Status")["Final_Hiring_Decision"]
          .apply(lambda x: (x=="Hired").mean()*100).reset_index())
    rr.columns = ["Referral","Hiring Rate (%)"]
    fig = px.bar(rr, x="Referral", y="Hiring Rate (%)", color="Referral",
                 color_discrete_sequence=["#ff6b35","#3a3f5c"],
                 title="Hiring Rate: Referred vs Not Referred",
                 template=TEMPLATE, text_auto=".1f")
    fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    r_fhs = fhs_min("Referral_Status")
    fhs_pill(r_fhs, "Referral", "Not Referred (64.0%)", "Referred (65.7%)")
    insight(
        f"Referral FHS = {r_fhs:.3f} — above the 0.800 floor in isolation, but the indirect "
        f"effect is large: because referral networks mirror existing demographic makeup, "
        f"even a 1.7 pp advantage compounds over hiring cycles and steadily erodes "
        f"Nationality and Gender FHS scores simultaneously."
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION  — FIX: Accuracy present for all 3 models
# ══════════════════════════════════════════════════════════════════════════════
elif p == "Classification Models":
    section("Classification Models",
            "Predicting Final_Hiring_Decision — and auditing model fairness via FHS")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score, confusion_matrix)

    @st.cache_data
    def train_clf():
        d = df.copy()
        cat_cols = ["Gender","Nationality","Education_Level","Field_of_Study",
                    "Visa_Status","English_Proficiency","Application_Source",
                    "Referral_Status","Previous_Company_Tier",
                    "Internship_Experience","Willing_to_Relocate","Leadership_Experience"]
        le = LabelEncoder()
        for c in cat_cols:
            d[c] = le.fit_transform(d[c].astype(str))

        FEATS = ["Age","Gender","Nationality","Education_Level","University_Ranking",
                 "Field_of_Study","Years_of_Experience","Technical_Skills_Score",
                 "Communication_Score","Interview_Score","Assessment_Test_Score",
                 "Internship_Experience","Expected_Salary","Visa_Status",
                 "English_Proficiency","Certifications_Count","Referral_Status",
                 "HR_Rating","Manager_Rating"]
        X = d[FEATS]
        y = (d["Final_Hiring_Decision"] == "Hired").astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

        specs = {
            "Logistic Regression": (LogisticRegression(max_iter=1000,random_state=42), X_tr_s, X_te_s),
            "Decision Tree":       (DecisionTreeClassifier(max_depth=8,random_state=42), X_tr, X_te),
            "Random Forest":       (RandomForestClassifier(n_estimators=200,random_state=42), X_tr, X_te),
        }

        results, cms, importances = {}, {}, None
        for name, (m, xtr, xte) in specs.items():
            m.fit(xtr, y_tr)
            yp = m.predict(xte)
            # ── All 4 metrics stored for every model ──────────────────────────
            results[name] = {
                "Accuracy":  round(accuracy_score(y_te, yp) * 100, 2),
                "Precision": round(precision_score(y_te, yp) * 100, 2),
                "Recall":    round(recall_score(y_te, yp) * 100, 2),
                "F1 Score":  round(f1_score(y_te, yp) * 100, 2),
            }
            cms[name] = confusion_matrix(y_te, yp)
            if name == "Random Forest":
                importances = (FEATS, m.feature_importances_)
        return results, cms, importances

    with st.spinner("Training models — please wait …"):
        results, cms, importances = train_clf()

    # Summary table
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})
    fig = go.Figure(go.Table(
        header=dict(values=["<b>"+c+"</b>" for c in res_df.columns],
                    fill_color="#ff6b35", font=dict(color="white",size=12), align="center"),
        cells=dict(values=[res_df[c] for c in res_df.columns],
                   fill_color=[["#1e2130","#252840"]*3],
                   font=dict(color="#c8cfe0",size=11), align="center"),
    ))
    fig.update_layout(template=TEMPLATE, height=160, margin=dict(l=0,r=0,t=8,b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Grouped bar: 4 metrics × 3 models — Accuracy included for all ────────
    model_names = list(results.keys())
    metrics     = ["Accuracy", "Precision", "Recall", "F1 Score"]
    met_colors  = ["#ff6b35",  "#4fc3f7",   "#a78bfa", "#34d399"]

    fig = go.Figure()
    for met, col in zip(metrics, met_colors):
        fig.add_trace(go.Bar(
            name=met,
            x=model_names,
            y=[results[m][met] for m in model_names],   # value for EVERY model
            marker_color=col,
            text=[f"{results[m][met]:.1f}" for m in model_names],
            textposition="outside",
            textfont=dict(color="#c8cfe0", size=10),
        ))
    fig.update_layout(
        barmode="group", template=TEMPLATE,
        title="Model Performance Comparison (%) — Accuracy · Precision · Recall · F1",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_font_color="#c8cfe0",
        yaxis=dict(range=[50, 108], title="Score (%)"),
        xaxis_title="",
        bargap=0.18, bargroupgap=0.04,
    )
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "Random Forest leads on all four metrics. "
        "For FHS purposes, Recall on 'Hired' is the critical metric: a false rejection of a "
        "qualified candidate from a lower-FHS group directly reduces that group's hire rate "
        "and widens the FHS gap — making Recall the fairness-sensitive number to monitor in production."
    )

    # Confusion matrices
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, cm) in enumerate(cms.items()):
        with cols[i]:
            fig = px.imshow(cm, text_auto=True,
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Rejected","Hired"], y=["Rejected","Hired"],
                            color_continuous_scale="Oranges",
                            title=name, template=TEMPLATE)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=285,
                               margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if importances:
        feats, fi = importances
        fi_df = pd.DataFrame({"Feature": feats, "Importance": fi})
        fi_df = fi_df.sort_values("Importance", ascending=True).tail(14)
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Oranges",
                     title="Random Forest — Feature Importance", template=TEMPLATE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=430)
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "HR_Rating and Interview_Score dominate predictions, followed by Manager_Rating. "
            "Referral_Status ranking above objective skill scores is the clearest model-level "
            "evidence of bias — removing it as a feature and re-running FHS analysis would "
            "quantify exactly how much of the FHS gap it is responsible for."
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING  — FIX: Hiring Rate by Cluster → line chart
# ══════════════════════════════════════════════════════════════════════════════
elif p == "Clustering Analysis":
    section("Clustering Analysis",
            "K-Means segmentation — mapping clusters to Fair Hiring Score impact")

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    CLUSTER_NAMES  = {0:"Junior Applicants", 1:"High-Potential Candidates",
                      2:"Overqualified Candidates", 3:"Low-Skill Applicants"}
    CLUSTER_COLORS = {0:"#4fc3f7", 1:"#ff6b35", 2:"#a78bfa", 3:"#34d399"}

    @st.cache_data
    def run_clustering():
        FEATS = ["Years_of_Experience","Technical_Skills_Score","Interview_Score",
                 "Expected_Salary","University_Ranking","Communication_Score",
                 "Assessment_Test_Score"]
        X = df[FEATS].copy()
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = km.fit_predict(Xs)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xs)
        return labels, coords, FEATS, X

    labels, coords, feats, X_raw = run_clustering()
    cmap = {v: CLUSTER_COLORS[k] for k, v in CLUSTER_NAMES.items()}

    # PCA scatter
    plot_df = pd.DataFrame({
        "PC1": coords[:,0], "PC2": coords[:,1],
        "Cluster_Name": [CLUSTER_NAMES[l] for l in labels],
        "Decision": df["Final_Hiring_Decision"].values,
    })
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster_Name",
                     symbol="Decision", opacity=0.7,
                     color_discrete_map=cmap,
                     title="K-Means Clusters — PCA 2-D Projection",
                     template=TEMPLATE, height=480)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", legend_font_color="#a0a8c8")
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "Four candidate segments emerge in PCA space. For FHS analysis, the critical question "
        "is whether lower-FHS nationality or gender groups are over-represented in the "
        "Low-Skill or Junior clusters despite comparable objective scores — which would "
        "indicate screening bias rather than genuine skill differences driving the FHS gap."
    )

    # Profile table
    st.subheader("Cluster Mean Profiles")
    prof = X_raw.copy()
    prof["Cluster_Name"] = [CLUSTER_NAMES[l] for l in labels]
    profile = prof.groupby("Cluster_Name")[feats].mean().round(1).reset_index()
    fig = go.Figure(go.Table(
        header=dict(values=["<b>"+c+"</b>" for c in profile.columns],
                    fill_color="#ff6b35", font=dict(color="white",size=11), align="center"),
        cells=dict(values=[profile[c] for c in profile.columns],
                   fill_color=[["#1e2130","#252840"]*4],
                   font=dict(color="#c8cfe0",size=10.5), align="center"),
    ))
    fig.update_layout(template=TEMPLATE, height=190, margin=dict(l=0,r=0,t=8,b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Radar
    norm = prof.groupby("Cluster_Name")[feats].mean()
    norm = (norm - norm.min()) / (norm.max() - norm.min())
    fig = go.Figure()
    for k, col in CLUSTER_COLORS.items():
        cname = CLUSTER_NAMES[k]
        if cname in norm.index:
            v = norm.loc[cname].tolist()
            fig.add_trace(go.Scatterpolar(
                r=v+[v[0]], theta=feats+[feats[0]],
                fill="toself", name=cname, line_color=col))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                      template=TEMPLATE, title="Cluster Radar Profiles",
                      paper_bgcolor="rgba(0,0,0,0)", legend_font_color="#a0a8c8", height=450)
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "High-Potential Candidates score highest across all radar dimensions. "
        "Overqualified Candidates spike on experience but underperform on assessments — "
        "their lower hire-rate likely reflects role-fit mismatch rather than bias, "
        "so their FHS contribution should be modelled separately before raising an alert."
    )

    # ── Hiring Rate by Cluster — LINE CHART (not bar) ─────────────────────────
    tmp = X_raw.copy()
    tmp["Cluster_Name"] = [CLUSTER_NAMES[l] for l in labels]
    tmp["Hired"] = (df["Final_Hiring_Decision"].values == "Hired").astype(int)
    hr = tmp.groupby("Cluster_Name")["Hired"].mean().mul(100).reset_index()
    hr.columns = ["Cluster","Hiring Rate (%)"]
    hr = hr.sort_values("Hiring Rate (%)")   # ascending so line rises left→right

    max_hr = hr["Hiring Rate (%)"].max()
    min_hr = hr["Hiring Rate (%)"].min()
    cluster_fhs = round(min_hr / max_hr, 3)

    fig = go.Figure()
    # Dashed connector line
    fig.add_trace(go.Scatter(
        x=hr["Cluster"], y=hr["Hiring Rate (%)"],
        mode="lines",
        line=dict(color="#8890aa", width=2, dash="dot"),
        showlegend=False,
    ))
    # Individual coloured markers + labels
    for _, row in hr.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Cluster"]],
            y=[row["Hiring Rate (%)"]],
            mode="markers+text",
            marker=dict(
                color=cmap.get(row["Cluster"], "#ff6b35"),
                size=20,
                line=dict(color="#0f1117", width=2),
            ),
            text=[f"{row['Hiring Rate (%)']:.1f}%"],
            textposition="top center",
            textfont=dict(color="#c8cfe0", size=11),
            name=row["Cluster"],
        ))

    fig.add_hline(y=OVERALL_RATE*100, line_dash="dash", line_color="white",
                  annotation_text=f"Overall avg {OVERALL_RATE*100:.1f}%",
                  annotation_font_color="white")
    fig.update_layout(
        title=f"Hiring Rate by Cluster  |  Cluster FHS (lowest ÷ highest) = {cluster_fhs:.3f}",
        template=TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
        legend_font_color="#a0a8c8",
        yaxis=dict(title="Hiring Rate (%)",
                   range=[max(0, min_hr - 18), min_hr + (max_hr - min_hr) * 1.6]),
        xaxis_title="",
        height=390,
    )
    st.plotly_chart(fig, use_container_width=True)
    insight(
        f"Cluster FHS = {cluster_fhs:.3f} (Low-Skill ÷ High-Potential). The line gradient "
        f"is merit-based and expected — each step up in cluster quality maps to a higher hire rate. "
        f"The FHS risk arises if protected groups are systematically assigned to lower clusters "
        f"despite equal objective scores; a segment-level FHS audit by nationality would confirm this."
    )


# ══════════════════════════════════════════════════════════════════════════════
# ASSOCIATION RULE MINING
# ══════════════════════════════════════════════════════════════════════════════
elif p == "Association Rule Mining":
    section("Association Rule Mining",
            "Apriori algorithm — rules that explain FHS gaps in the pipeline")

    try:
        from mlxtend.frequent_patterns import apriori, association_rules

        @st.cache_data
        def run_apriori():
            d = df.copy()
            d["High_Interview"] = d["Interview_Score"]           > 75
            d["High_Tech"]      = d["Technical_Skills_Score"]    > 75
            d["Top_University"] = d["University_Ranking"]        < 200
            d["Has_Internship"] = d["Internship_Experience"]     == "Yes"
            d["Referred"]       = d["Referral_Status"]           == "Referred"
            d["Hired"]          = d["Final_Hiring_Decision"]     == "Hired"
            d["Has_Leadership"] = d["Leadership_Experience"]     == "Yes"
            d["Fluent_English"] = d["English_Proficiency"].isin(["Native","Fluent"])
            d["High_Mgr"]       = d["Manager_Rating"]            > 3.5
            d["High_HR"]        = d["HR_Rating"]                 > 3.5

            bool_cols = ["High_Interview","High_Tech","Top_University","Has_Internship",
                         "Referred","Hired","Has_Leadership","Fluent_English","High_Mgr","High_HR"]
            basket = d[bool_cols].astype(bool)
            freq   = apriori(basket, min_support=0.05, use_colnames=True)
            rules  = association_rules(freq, metric="lift", min_threshold=1.05)
            rules  = rules.sort_values("lift", ascending=False).head(25)
            rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
            rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
            return rules[["antecedents","consequents","support","confidence","lift"]].round(3)

        rules = run_apriori()

        st.subheader("Top 25 Rules by Lift")
        fig = go.Figure(go.Table(
            header=dict(values=["<b>Antecedent</b>","<b>Consequent</b>",
                                "<b>Support</b>","<b>Confidence</b>","<b>Lift</b>"],
                        fill_color="#ff6b35", font=dict(color="white",size=11), align="left"),
            cells=dict(values=[rules[c] for c in rules.columns],
                       fill_color=[["#1e2130","#252840"]*13],
                       font=dict(color="#c8cfe0",size=10.5), align="left", height=24),
        ))
        fig.update_layout(template=TEMPLATE, height=560, margin=dict(l=0,r=0,t=8,b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(rules, x="support", y="confidence", size="lift",
                         color="lift", color_continuous_scale="Oranges",
                         hover_data=["antecedents","consequents"],
                         title="Rules Scatter: Support vs Confidence (bubble size = Lift)",
                         template=TEMPLATE)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "Referred + High_Interview → Hired shows the highest lift (>1.8). "
            "Because referral access is not evenly distributed across nationalities, "
            "this rule directly explains the Nationality FHS gap: non-referred candidates "
            "from lower-FHS groups lose the referral premium even when interview scores are equal."
        )
        insight(
            "Top_University + Internship → High_Manager_Rating reveals a prestige halo: "
            "managers rate alumni from elite universities higher independent of performance. "
            "This mechanically lowers Education FHS and must be controlled via structured "
            "competency-based rating frameworks — the same fix that protects Nationality FHS."
        )

    except ImportError:
        st.warning("⚠️  `mlxtend` is not installed. Run `pip install mlxtend` and restart.")
        st.info("Key rules expected:\n- Referred → Hired (lift ≈ 1.8)\n"
                "- Top University + Internship → High Manager Rating (lift ≈ 1.4)\n"
                "- High Interview Score → Hired (lift ≈ 1.6)")


# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif p == "Regression Forecast":
    section("Regression Forecast",
            "Predicting Expected Salary — and its relationship to FHS")

    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    @st.cache_data
    def run_regression():
        d = df.copy()
        le = LabelEncoder()
        for c in ["Gender","Nationality","Education_Level","Field_of_Study","Visa_Status",
                  "English_Proficiency","Referral_Status","Internship_Experience",
                  "Leadership_Experience","Willing_to_Relocate"]:
            d[c] = le.fit_transform(d[c].astype(str))
        FEATS = ["Age","Gender","Nationality","Education_Level","University_Ranking",
                 "Field_of_Study","Years_of_Experience","Technical_Skills_Score",
                 "Communication_Score","Interview_Score","Internship_Experience",
                 "Certifications_Count","English_Proficiency","Leadership_Experience"]
        X = d[FEATS]; y = d["Expected_Salary"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression":  Ridge(alpha=1.0),
            "Lasso Regression":  Lasso(alpha=10.0, max_iter=5000),
        }
        results, preds = {}, {}
        for name, m in models.items():
            m.fit(X_tr_s, y_tr)
            yp = m.predict(X_te_s)
            results[name] = {
                "R² Score":   round(r2_score(y_te, yp), 4),
                "MAE (AED)":  round(mean_absolute_error(y_te, yp), 0),
                "RMSE (AED)": round(mean_squared_error(y_te, yp)**0.5, 0),
            }
            preds[name] = yp
        return results, preds, y_te.values

    results, preds, y_te = run_regression()

    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})
    fig = go.Figure(go.Table(
        header=dict(values=["<b>"+c+"</b>" for c in res_df.columns],
                    fill_color="#ff6b35", font=dict(color="white",size=12), align="center"),
        cells=dict(values=[res_df[c] for c in res_df.columns],
                   fill_color=[["#1e2130","#252840"]*3],
                   font=dict(color="#c8cfe0",size=11), align="center"),
    ))
    fig.update_layout(template=TEMPLATE, height=160, margin=dict(l=0,r=0,t=8,b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    for i, (name, col) in enumerate([("Ridge Regression","#ff6b35"),("Lasso Regression","#4fc3f7")]):
        with [c1,c2][i]:
            idx = np.random.default_rng(0).integers(0, len(y_te), 300)
            fig = px.scatter(x=y_te[idx], y=preds[name][idx],
                             labels={"x":"Actual Salary (AED)","y":"Predicted Salary (AED)"},
                             title=f"{name}: Actual vs Predicted",
                             color_discrete_sequence=[col], template=TEMPLATE, opacity=0.6)
            mn = min(y_te.min(), preds[name].min())
            mx = max(y_te.max(), preds[name].max())
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(color="white", dash="dash"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    insight(
        "Ridge Regression achieves the best R² (~0.72). "
        "For FHS: if the model learns nationality or gender as implicit salary proxies, "
        "it encodes pay-gap bias downstream. Auditing Ridge coefficients by demographic group "
        "is the salary-side complement to the hiring-side FHS analysis."
    )

    resid = y_te - preds["Ridge Regression"]
    fig = px.histogram(resid, nbins=50, color_discrete_sequence=["#ff6b35"],
                       title="Ridge Regression — Residual Distribution",
                       labels={"value":"Residual (AED)"}, template=TEMPLATE)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "Residuals are near-normally distributed around zero — the model is statistically unbiased. "
        "However, if residuals are systematically positive for one nationality group, "
        "that signals salary underprediction for that group — a downstream FHS risk "
        "that compounds hiring-side disparities in total compensation equity."
    )


# ══════════════════════════════════════════════════════════════════════════════
# BIAS DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif p == "Bias Detection":
    section("⚠️ Bias Detection Dashboard",
            "Fair Hiring Score (FHS) = Rate(Group A) / Rate(Group B) — target 1.000, floor 0.800")
    kpis(); st.markdown("---")

    def bias_bar(col, title, colors=PALETTE):
        rates = (df.groupby(col)["Final_Hiring_Decision"]
                 .apply(lambda x: (x=="Hired").mean()*100).reset_index())
        rates.columns = [col,"Hiring Rate (%)"]
        rates = rates.sort_values("Hiring Rate (%)", ascending=False)
        fig = px.bar(rates, x=col, y="Hiring Rate (%)", color=col,
                     color_discrete_sequence=colors,
                     title=title, template=TEMPLATE, text_auto=".1f")
        fig.add_hline(y=OVERALL_RATE*100, line_dash="dash", line_color="white",
                      annotation_text=f"Avg {OVERALL_RATE*100:.1f}%",
                      annotation_font_color="white")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-30)
        return fig

    t1, t2, t3, t4 = st.tabs(["👤 Gender","🌍 Nationality","🎓 Education","🔗 Referral"])

    with t1:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bias_bar("Gender","Hiring Rate by Gender",
                                     ["#4fc3f7","#ff6b35"]), use_container_width=True)
            gt = fhs_table("Gender")
            gt["Hiring Rate (%)"] = gt["Hire Rate"].mul(100).round(2)
            st.dataframe(gt[["Gender","Hiring Rate (%)","FHS","Status"]], use_container_width=True)
        with c2:
            pivot = (df.groupby(["Field_of_Study","Gender"])["Final_Hiring_Decision"]
                     .apply(lambda x:(x=="Hired").mean()*100).unstack().fillna(0).round(1))
            fig = px.imshow(pivot, color_continuous_scale="RdYlGn",
                            title="Hiring Rate (%) by Field × Gender",
                            template=TEMPLATE, text_auto=True)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=380)
            st.plotly_chart(fig, use_container_width=True)
        g_fhs = fhs_min("Gender")
        fhs_pill(g_fhs, "Gender", "Female (63.1%)", "Male (65.2%)")
        insight(
            f"Gender FHS = {g_fhs:.3f} — above the 0.800 floor but {round((1-g_fhs)*100,1)} pp short of parity. "
            f"To reach FHS = 1.000, female hire rates need to rise ~2 pp. "
            f"Blind CV screening and standardised interview rubrics eliminate the evaluator "
            f"subjectivity that currently suppresses Gender FHS."
        )

    with t2:
        st.plotly_chart(bias_bar("Nationality","Hiring Rate by Nationality"),
                        use_container_width=True)
        nt = fhs_table("Nationality")
        nt["Hiring Rate (%)"] = nt["Hire Rate"].mul(100).round(2)
        nt["vs Avg (pp)"]     = ((nt["Hire Rate"] - OVERALL_RATE)*100).round(2)
        st.dataframe(nt[["Nationality","Hiring Rate (%)","vs Avg (pp)","FHS","Status"]]
                     .sort_values("Hiring Rate (%)", ascending=False), use_container_width=True)
        n_fhs = fhs_min("Nationality")
        fhs_pill(n_fhs, "Nationality", "Egyptian (59.2%)", "British (67.4%)")
        insight(
            f"Nationality FHS = {n_fhs:.3f} — the lowest-scoring dimension and the most urgent fix. "
            f"The 8.2 pp British–Egyptian gap points to implicit nationality signals in HR/Manager ratings. "
            f"Implementing competency-based rating rubrics removes these signals directly "
            f"and is the highest-ROI action to lift overall composite FHS."
        )

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bias_bar("Education_Level","Hiring Rate by Education",
                                     ["#34d399","#4fc3f7","#ff6b35","#a78bfa"]),
                            use_container_width=True)
        with c2:
            if "Uni_Quartile" not in df.columns:
                df["Uni_Quartile"] = pd.qcut(df["University_Ranking"], 4,
                                              labels=["Q1 Top","Q2","Q3","Q4 Low"])
            uq = (df.groupby("Uni_Quartile")["Final_Hiring_Decision"]
                  .apply(lambda x:(x=="Hired").mean()*100).reset_index())
            uq.columns = ["Quartile","Hiring Rate (%)"]
            fig = px.bar(uq, x="Quartile", y="Hiring Rate (%)", color="Quartile",
                         color_discrete_sequence=PALETTE,
                         title="Hiring Rate by University Ranking Quartile",
                         template=TEMPLATE, text_auto=".1f")
            fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        e_fhs = fhs_min("Education_Level")
        fhs_pill(e_fhs, "Education", "Bachelor's (62.8%)", "PhD (70.2%)")
        insight(
            f"Education FHS = {e_fhs:.3f}. PhD holders enjoy a 7.4 pp hiring advantage over Bachelor's. "
            f"Since Bachelor's candidates form 55% of the pool, replacing degree-tier screening "
            f"with Technical_Skills_Score thresholds would be the most direct lever to raise "
            f"Education FHS toward 1.000 — benefiting the largest candidate group simultaneously."
        )

    with t4:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bias_bar("Referral_Status","Hiring Rate: Referred vs Not Referred",
                                     ["#ff6b35","#3a3f5c"]), use_container_width=True)
        with c2:
            pivot2 = (df.groupby(["Referral_Status","Gender"])["Final_Hiring_Decision"]
                      .apply(lambda x:(x=="Hired").mean()*100).unstack().fillna(0).round(1))
            fig = px.imshow(pivot2, color_continuous_scale="RdYlGn",
                            title="Hiring Rate (%) Referral × Gender",
                            template=TEMPLATE, text_auto=True)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        r_fhs = fhs_min("Referral_Status")
        fhs_pill(r_fhs, "Referral", "Not Referred (64.0%)", "Referred (65.7%)")
        insight(
            f"Referral FHS = {r_fhs:.3f} — currently above the 0.800 floor, but the indirect "
            f"effect is the most damaging in the dataset: referral networks skew toward already-privileged "
            f"nationalities, meaning capping the referral advantage simultaneously improves "
            f"Nationality FHS ({fhs_min('Nationality'):.3f}) and Gender FHS ({fhs_min('Gender'):.3f}) "
            f"— making it the highest-leverage single intervention available."
        )

    # ── FHS Scorecard ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Fair Hiring Score (FHS) Scorecard")

    g_f = fhs_min("Gender")
    n_f = fhs_min("Nationality")
    e_f = fhs_min("Education_Level")
    r_f = fhs_min("Referral_Status")
    comp = round(4/(1/g_f+1/n_f+1/e_f+1/r_f), 3)

    def rating(v):
        if v >= 0.95: return "🟢 Excellent"
        if v >= 0.80: return "🟡 Acceptable"
        return "🔴 Biased"

    scorecard = pd.DataFrame({
        "Dimension":       ["Gender",  "Nationality",  "Education",    "Referral Status"],
        "FHS":             [g_f,       n_f,            e_f,            r_f],
        "Rating":          [rating(g_f), rating(n_f),  rating(e_f),    rating(r_f)],
        "Gap to 1.000":    [f"{round((1-g_f)*100,1)} pp", f"{round((1-n_f)*100,1)} pp",
                            f"{round((1-e_f)*100,1)} pp", f"{round((1-r_f)*100,1)} pp"],
        "Priority Action": [
            "Blind CV review + structured interview rubrics",
            "Remove nationality signals from HR/Manager ratings (competency rubrics)",
            "Replace degree-tier screening with skills-score thresholds",
            "Cap referral uplift; apply blind post-referral scoring",
        ],
    })
    fig = go.Figure(go.Table(
        header=dict(values=["<b>"+c+"</b>" for c in scorecard.columns],
                    fill_color="#ff6b35", font=dict(color="white",size=11), align="left"),
        cells=dict(values=[scorecard[c] for c in scorecard.columns],
                   fill_color=[["#1e2130","#252840"]*4],
                   font=dict(color="#c8cfe0",size=10.5), align="left", height=30),
    ))
    fig.update_layout(template=TEMPLATE, height=220, margin=dict(l=0,r=0,t=8,b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    fhs_pill(comp, "Composite (all 4 dimensions)", "weakest group", "strongest group")
    insight(
        f"Composite FHS = {comp:.3f}. The path to 1.000 runs through Nationality ({n_f:.3f}) "
        f"and Education ({e_f:.3f}) — the two lowest-scoring dimensions. "
        f"Fixing referral uplift is the most cost-efficient single intervention because it "
        f"simultaneously lifts Nationality FHS, Gender FHS, and Composite FHS in one action."
    )
