"""
SearchSense AI - PS-01 Search Term Triage Engine
------------------------------------------------
A complete, functional PPC Search Term Intelligence and Optimization Application.
Features:
1. Executive Dashboard (Live dataset metrics, dynamic KPIs, interactive Plotly charts)
2. Search Term Explorer (Multi-dimensional filters, sorting, row inspector)
3. Decision Intelligence (Explainable rules, uncertainty prioritization, near-duplicate query grouping)
4. Budget Waste Detector (Historical spend analysis, scoped negative keyword generator)
5. Human Review Center (Human-in-the-loop overrides, review reasons, audit tracking)
6. Export Center (Downloadable CSV with all original rows + decisions, reasons, and overrides)
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Modular internal packages
from src.preprocessing import validate_and_preprocess, get_summary_statistics, REQUIRED_COLUMNS
from src.intent_analysis import enrich_dataframe_with_intent, detect_near_duplicates
from src.decision_engine import run_decision_engine, DEFAULT_THRESHOLDS


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & ENTERPRISE STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SearchSense AI | PS-01 Triage Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, responsive PPC operating system feel
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    /* Status Badges */
    .badge-promote {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-block {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-ignore {
        background-color: #f1f5f9;
        color: #475569;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-review {
        background-color: #fef3c7;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    /* Informational Notice Box */
    .notice-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: #166534;
    }
    .warning-box {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 0.88rem;
        color: #92400e;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED DATA PIPELINE FOR INSTANT PERFORMANCE (25,000+ ROWS)
# -----------------------------------------------------------------------------
REAL_DATA_PATH = Path(__file__).parent / "data" / "search_terms_report.csv"
SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_search_terms.csv"


@st.cache_data(show_spinner="Processing PPC search terms through PS-01 Triage Engine...")
def execute_triage_pipeline(file_content_hash: str, raw_df: pd.DataFrame, thresholds: Dict[str, float]) -> Tuple[pd.DataFrame, list, list]:
    """
    High-performance pipeline execution with Streamlit caching.
    Runs data cleaning -> domain intent categorization -> PS-01 conservative decision engine.
    """
    clean_df, errors, warnings = validate_and_preprocess(raw_df, preserve_rows=True)
    if clean_df is None or errors:
        return None, errors, warnings
        
    intent_df = enrich_dataframe_with_intent(clean_df)
    triage_df = run_decision_engine(intent_df, thresholds)
    
    return triage_df, errors, warnings


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "active_data_mode" not in st.session_state:
    if REAL_DATA_PATH.exists():
        st.session_state.active_data_mode = "real"
        st.session_state.data_source_label = "Production Search Terms Report (25,001 rows)"
        st.session_state.raw_df = pd.read_csv(REAL_DATA_PATH)
    elif SAMPLE_DATA_PATH.exists():
        st.session_state.active_data_mode = "sample"
        st.session_state.data_source_label = "Synthetic Demo Sample (38 rows - Marked Synthetic)"
        st.session_state.raw_df = pd.read_csv(SAMPLE_DATA_PATH)
    else:
        st.session_state.active_data_mode = "none"
        st.session_state.data_source_label = "No Data Loaded"
        st.session_state.raw_df = None

if "thresholds" not in st.session_state:
    st.session_state.thresholds = DEFAULT_THRESHOLDS.copy()

if "human_overrides" not in st.session_state:
    st.session_state.human_overrides = {}  # key: row_id -> dict(final_decision, reviewer_notes)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & DATA CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎯 SearchSense AI")
    st.caption("PS-01 Search Term Triage Engine | Commercial Lending PPC")
    
    # 6 Functional Pages
    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🔍 Search Term Explorer",
            "🧠 Decision Intelligence",
            "💸 Budget Waste Detector",
            "✍️ Human Review Center",
            "📥 Export Center"
        ],
        index=0
    )
    
    st.markdown("---")
    st.subheader("📂 Dataset Selection")
    
    # Quick dataset switchers
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("📈 Real Report", use_container_width=True, help="Load 25,001-row real search terms report"):
            if REAL_DATA_PATH.exists():
                st.session_state.raw_df = pd.read_csv(REAL_DATA_PATH)
                st.session_state.active_data_mode = "real"
                st.session_state.data_source_label = "Production Search Terms Report (25,001 rows)"
                st.rerun()
    with c_btn2:
        if st.button("🧪 Sample CSV", use_container_width=True, help="Load synthetic demo sample"):
            if SAMPLE_DATA_PATH.exists():
                st.session_state.raw_df = pd.read_csv(SAMPLE_DATA_PATH)
                st.session_state.active_data_mode = "sample"
                st.session_state.data_source_label = "Synthetic Demo Sample (38 rows - Clearly Marked Synthetic)"
                st.rerun()
                
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload Custom PPC CSV",
        type=["csv", "xlsx"],
        help=f"Accepts reports with columns: {', '.join(REQUIRED_COLUMNS)}"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
                loaded_df = pd.read_excel(uploaded_file)
            else:
                loaded_df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = loaded_df
            st.session_state.active_data_mode = "upload"
            st.session_state.data_source_label = f"Uploaded File: {uploaded_file.name} ({len(loaded_df):,} rows)"
            st.success(f"Loaded {len(loaded_df):,} rows!")
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")

    st.caption(f"**Active Source:** {st.session_state.data_source_label}")
    if st.session_state.active_data_mode == "sample":
        st.warning("⚠️ Currently using Synthetic Sample data. Real dataset available via 'Real Report' button.")

    # Threshold Controls Expander
    st.markdown("---")
    with st.expander("⚙️ Decision Thresholds (PS-01)", expanded=False):
        st.caption("Conservative heuristic parameters for commercial lending search triage.")
        t_waste = st.number_input(
            "Min Spend w/ 0 Conv for BLOCK ($)", 
            min_value=5.0, max_value=500.0, 
            value=float(st.session_state.thresholds["min_spend_waste_threshold"]), 
            step=5.0
        )
        t_cpa = st.number_input(
            "Target CPA Benchmark ($)", 
            min_value=10.0, max_value=500.0, 
            value=float(st.session_state.thresholds["target_cpa"]), 
            step=5.0
        )
        t_roas = st.number_input(
            "Min ROAS for PROMOTE", 
            min_value=0.5, max_value=10.0, 
            value=float(st.session_state.thresholds["min_roas_promote"]), 
            step=0.25
        )
        t_conv = st.number_input(
            "Min Conversions for PROMOTE", 
            min_value=1, max_value=20, 
            value=int(st.session_state.thresholds["min_conversions_promote"]), 
            step=1
        )
        t_clicks = st.number_input(
            "Min Clicks for Sample Confidence", 
            min_value=5, max_value=100, 
            value=int(st.session_state.thresholds["min_clicks_significance"]), 
            step=5
        )
        
        st.session_state.thresholds = {
            "min_spend_waste_threshold": t_waste,
            "target_cpa": t_cpa,
            "min_roas_promote": t_roas,
            "min_conversions_promote": t_conv,
            "min_clicks_significance": t_clicks,
            "high_cpa_multiplier": 2.5
        }

    # Safety Notice
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#64748b;'>"
        "<strong>Safety Guardrail:</strong> SearchSense AI provides decision intelligence and proposals. "
        "It <em>never automatically modifies</em> a live Google Ads account."
        "</div>",
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# DATASET EXECUTION & HUMAN OVERRIDE MERGE
# -----------------------------------------------------------------------------
if st.session_state.raw_df is None or st.session_state.raw_df.empty:
    st.warning("⚠️ No dataset loaded. Please click 'Real Report' in the sidebar or upload a search terms CSV.")
    st.stop()

# Generate cache hash based on length and first/last row
df_hash = f"{len(st.session_state.raw_df)}_{st.session_state.data_source_label}_{str(st.session_state.thresholds)}"
df_triage, pipeline_errors, pipeline_warnings = execute_triage_pipeline(
    df_hash, st.session_state.raw_df, st.session_state.thresholds
)

if pipeline_errors:
    st.error("🚨 Critical Validation Errors:")
    for err in pipeline_errors:
        st.error(f"- {err}")
    st.stop()

# Create working dataframe with session human overrides applied
df_active = df_triage.copy()
for row_id, override in st.session_state.human_overrides.items():
    mask = df_active["row_id"] == row_id
    if mask.any():
        if "final_decision" in override and override["final_decision"]:
            df_active.loc[mask, "final_decision"] = override["final_decision"]
        if "reviewer_notes" in override and override["reviewer_notes"]:
            df_active.loc[mask, "reviewer_notes"] = override["reviewer_notes"]

# Compute dynamic metrics from active dataset
stats = get_summary_statistics(df_active)
total_est_waste = float(df_active["estimated_historical_waste"].sum())
total_promote_count = int((df_active["final_decision"] == "PROMOTE").sum())
total_block_count = int((df_active["final_decision"] == "BLOCK").sum())
total_ignore_count = int((df_active["final_decision"] == "IGNORE").sum())
total_review_needed = int(df_active["review_required"].sum())


# -----------------------------------------------------------------------------
# PAGE 1: 📊 DASHBOARD
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 Executive PPC Dashboard | SearchSense AI")
    
    if st.session_state.active_data_mode == "sample":
        st.markdown(
            f"<div class='warning-box'><strong>Notice:</strong> Currently displaying <strong>Synthetic Sample Data</strong>. "
            f"Click 'Real Report' in the sidebar to switch to the 25,001-row real commercial lending dataset.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='notice-box'>Loaded <strong>{stats['total_terms']:,}</strong> real search terms from <strong>{st.session_state.data_source_label}</strong>. "
            f"All metrics are calculated live from the dataset.</div>",
            unsafe_allow_html=True
        )

    # Row 1: Executive KPI Metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Total Spend</div>"
            f"<div class='metric-value'>${stats['total_cost']:,.2f}</div>"
            f"<div class='metric-sub'>{stats['total_clicks']:,} Clicks | ${stats['overall_cpc']:.2f} CPC</div></div>",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Total Revenue</div>"
            f"<div class='metric-value'>${stats['total_conversion_value']:,.2f}</div>"
            f"<div class='metric-sub'>Profit Contrib: ${stats['overall_profit']:,.2f}</div></div>",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Conversions</div>"
            f"<div class='metric-value'>{stats['total_conversions']:,}</div>"
            f"<div class='metric-sub'>CVR: {stats['overall_cvr']:.2f}% | CTR: {stats['overall_ctr']:.2f}%</div></div>",
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Avg CPA / ROAS</div>"
            f"<div class='metric-value'>${stats['overall_cpa']:.2f}</div>"
            f"<div class='metric-sub'>ROAS: {stats['overall_roas']:.2f}x</div></div>",
            unsafe_allow_html=True
        )
    with k5:
        waste_pct = (total_est_waste / stats["total_cost"] * 100) if stats["total_cost"] > 0 else 0.0
        st.markdown(
            f"<div class='metric-card' style='border-left: 4px solid #ef4444;'><div class='metric-title' style='color:#ef4444;'>Est. Past Waste</div>"
            f"<div class='metric-value' style='color:#ef4444;'>${total_est_waste:,.2f}</div>"
            f"<div class='metric-sub'>{waste_pct:.1f}% of spend (Historical estimate)</div></div>",
            unsafe_allow_html=True
        )

    # Row 2: Triage Summary Badges
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("🚀 PROMOTE Candidates", f"{total_promote_count:,}", f"{(total_promote_count/stats['total_terms']*100):.1f}% of total")
    b2.metric("🚫 BLOCK Candidates", f"{total_block_count:,}", f"{(total_block_count/stats['total_terms']*100):.1f}% of total")
    b3.metric("💤 IGNORE Candidates", f"{total_ignore_count:,}", f"{(total_ignore_count/stats['total_terms']*100):.1f}% of total")
    b4.metric("⚠️ Review Required", f"{total_review_needed:,}", "Uncertain or low-evidence")

    st.markdown("---")

    # Row 3: Interactive Visualizations
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("🎯 PS-01 Decision Distribution")
        decision_df = df_active["final_decision"].value_counts().reset_index()
        decision_df.columns = ["Decision", "Count"]
        
        color_map = {
            "PROMOTE": "#10b981", 
            "BLOCK": "#ef4444", 
            "IGNORE": "#64748b"
        }
        fig_donut = px.pie(
            decision_df,
            names="Decision",
            values="Count",
            hole=0.45,
            color="Decision",
            color_discrete_map=color_map
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent+label")
        fig_donut.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.subheader("🏷️ Query Intent vs Historical Spend")
        intent_spend = df_active.groupby("intent")["cost"].sum().reset_index().sort_values(by="cost", ascending=True)
        fig_intent = px.bar(
            intent_spend,
            x="cost",
            y="intent",
            orientation="h",
            labels={"cost": "Historical Spend ($)", "intent": "Intent Category"},
            color="intent",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_intent.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_intent, use_container_width=True)

    # Row 4: Match Type Performance & Campaign Performance
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.subheader("🎯 Match Type Breakdown")
        match_summary = df_active.groupby("match_type").agg(
            terms=("row_id", "count"),
            spend=("cost", "sum"),
            conversions=("conversions", "sum"),
            revenue=("conversion_value", "sum")
        ).reset_index()
        match_summary["roas"] = np.where(match_summary["spend"] > 0, match_summary["revenue"] / match_summary["spend"], 0.0)
        
        fig_match = px.bar(
            match_summary,
            x="match_type",
            y="spend",
            color="match_type",
            labels={"spend": "Total Spend ($)", "match_type": "Match Type"},
            text_auto=".2s"
        )
        fig_match.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_match, use_container_width=True)
        
    with col_m2:
        st.subheader("🏢 Top Campaigns by Spend & Conversions")
        camp_summary = df_active.groupby("campaign").agg(
            spend=("cost", "sum"),
            conversions=("conversions", "sum")
        ).reset_index().sort_values(by="spend", ascending=False).head(5)
        
        fig_camp = px.bar(
            camp_summary,
            x="spend",
            y="campaign",
            orientation="h",
            labels={"spend": "Spend ($)", "campaign": "Campaign"},
            color_discrete_sequence=["#3b82f6"]
        )
        fig_camp.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_camp, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 2: 🔍 SEARCH TERM EXPLORER
# -----------------------------------------------------------------------------
elif page == "🔍 Search Term Explorer":
    st.title("🔍 Search Term Explorer")
    st.markdown("Filter, sort, and inspect search terms across campaigns, match types, and intent categories.")
    
    # Filter Toolbar
    with st.expander("🛠️ Filter & Search Controls", expanded=True):
        f1, f2, f3 = st.columns([2, 1.5, 1.5])
        with f1:
            search_query_str = st.text_input("🔎 Search by Term / Keyword", placeholder="Type keywords (e.g. sba, line of credit, grant)...")
        with f2:
            camp_options = ["All"] + sorted(df_active["campaign"].unique().tolist())
            selected_camp = st.selectbox("Campaign", camp_options)
        with f3:
            match_options = ["All"] + sorted(df_active["match_type"].unique().tolist())
            selected_match = st.selectbox("Match Type", match_options)

        f4, f5, f6 = st.columns([1.5, 1.5, 1.5])
        with f4:
            decision_options = ["All", "PROMOTE", "BLOCK", "IGNORE"]
            selected_dec = st.selectbox("Decision", decision_options)
        with f5:
            intent_options = ["All"] + sorted(df_active["intent"].unique().tolist())
            selected_intent = st.selectbox("Intent Category", intent_options)
        with f6:
            review_filter = st.selectbox("Review Status", ["All", "Review Required Only", "No Review Needed"])

        # Metric ranges
        max_cost = float(df_active["cost"].max())
        cost_range = st.slider("Cost Range ($)", min_value=0.0, max_value=max(100.0, max_cost), value=(0.0, max(100.0, max_cost)), step=5.0)

    # Apply Filters
    df_filtered = df_active.copy()
    if search_query_str:
        df_filtered = df_filtered[df_filtered["query"].str.contains(search_query_str, case=False, na=False)]
    if selected_camp != "All":
        df_filtered = df_filtered[df_filtered["campaign"] == selected_camp]
    if selected_match != "All":
        df_filtered = df_filtered[df_filtered["match_type"] == selected_match]
    if selected_dec != "All":
        df_filtered = df_filtered[df_filtered["final_decision"] == selected_dec]
    if selected_intent != "All":
        df_filtered = df_filtered[df_filtered["intent"] == selected_intent]
    if review_filter == "Review Required Only":
        df_filtered = df_filtered[df_filtered["review_required"] == True]
    elif review_filter == "No Review Needed":
        df_filtered = df_filtered[df_filtered["review_required"] == False]
    df_filtered = df_filtered[(df_filtered["cost"] >= cost_range[0]) & (df_filtered["cost"] <= cost_range[1])]

    st.markdown(f"**Showing {len(df_filtered):,} of {len(df_active):,} terms**")

    # Display Columns
    view_cols = [
        "row_id", "query", "campaign", "ad_group", "match_type",
        "final_decision", "review_required", "confidence",
        "impressions", "clicks", "cost", "conversions", "conversion_value",
        "cpa", "roas", "intent", "reason"
    ]
    
    st.dataframe(
        df_filtered[view_cols].style.format({
            "cost": "${:,.2f}",
            "conversion_value": "${:,.2f}",
            "cpa": "${:,.2f}",
            "roas": "{:.2f}x",
            "impressions": "{:,}",
            "clicks": "{:,}",
            "conversions": "{:,}"
        }),
        use_container_width=True,
        height=450
    )

    # Row Detail Inspector
    st.markdown("---")
    st.subheader("🔍 Search Term Inspector")
    if not df_filtered.empty:
        inspect_options = [f"Row {r['row_id']}: {r['query']} [{r['campaign']}]" for _, r in df_filtered.head(100).iterrows()]
        selected_insp_str = st.selectbox("Select a Search Term to Inspect Row Details:", inspect_options)
        
        if selected_insp_str:
            sel_row_id = int(selected_insp_str.split(":")[0].replace("Row", "").strip())
            insp_row = df_active[df_active["row_id"] == sel_row_id].iloc[0]
            
            c_ins1, c_ins2, c_ins3 = st.columns([2, 1, 1])
            with c_ins1:
                st.markdown(f"### `{insp_row['query']}`")
                st.caption(f"Campaign: **{insp_row['campaign']}** | Ad Group: **{insp_row['ad_group']}** | Match: **{insp_row['match_type']}**")
                st.markdown(f"**One-Line Reason:** *{insp_row['reason']}*")
            with c_ins2:
                st.markdown(f"**Decision:** `{insp_row['final_decision']}`")
                st.markdown(f"**Confidence:** `{insp_row['confidence']}`")
                st.markdown(f"**Review Required:** `{'⚠️ Yes' if insp_row['review_required'] else '✅ No'}`")
            with c_ins3:
                st.markdown(f"**Spend:** `${insp_row['cost']:.2f}`")
                st.markdown(f"**Conversions:** `{insp_row['conversions']}` (CVR: {insp_row['cvr']:.1f}%)")
                st.markdown(f"**ROAS:** `{insp_row['roas']:.2f}x` | **CPA:** `${insp_row['cpa']:.2f}`")


# -----------------------------------------------------------------------------
# PAGE 3: 🧠 DECISION INTELLIGENCE
# -----------------------------------------------------------------------------
elif page == "🧠 Decision Intelligence":
    st.title("🧠 Decision Intelligence & Heuristic Transparency")
    st.markdown(
        "<div class='notice-box'><strong>Explainable PPC Intelligence:</strong> Built on deterministic rules, "
        "commercial lending token ontologies, and uncertainty modeling. No black-box ML claims.</div>",
        unsafe_allow_html=True
    )

    tab_rules, tab_uncertainty, tab_grouping = st.tabs([
        "📋 Triage Rule Logic",
        "⚖️ Uncertainty Prioritization Queue",
        "🔗 Near-Duplicate Query Grouping"
    ])

    with tab_rules:
        st.subheader("PS-01 Conservative Triage Rule Architecture")
        
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                "<div class='metric-card' style='border-top: 4px solid #10b981;'>"
                "<h4>🚀 PROMOTE Criteria</h4>"
                "<ul>"
                "<li>Conversions &ge; <strong>2</strong></li>"
                "<li>Clicks &ge; <strong>15</strong> (sample significance)</li>"
                "<li>ROAS &ge; <strong>2.0x</strong> OR CPA &le; <strong>$60</strong></li>"
                "<li>Commercial lending intent confirmed</li>"
                "<li><strong>Never</strong> promoted solely for 1 lucky conversion</li>"
                "</ul>"
                "</div>",
                unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                "<div class='metric-card' style='border-top: 4px solid #ef4444;'>"
                "<h4>🚫 BLOCK Criteria</h4>"
                "<ul>"
                "<li>Negative intent token matched (jobs, grants, templates, personal loans)</li>"
                "<li>Spend &ge; <strong>$40</strong> with 0 conversions</li>"
                "<li>Severe CPA bleed (> 2.5x target CPA and ROAS < 0.8x)</li>"
                "<li>Proposed as Negative Keyword with scope</li>"
                "</ul>"
                "</div>",
                unsafe_allow_html=True
            )
        with r3:
            st.markdown(
                "<div class='metric-card' style='border-top: 4px solid #64748b;'>"
                "<h4>💤 IGNORE Criteria</h4>"
                "<ul>"
                "<li>Low spend (< $40) with 0 conversions</li>"
                "<li>Normal exploratory PPC traffic</li>"
                "<li>Unproven conversions with &lt; 15 clicks routed to Review Queue</li>"
                "<li>Allowed to run in existing match type</li>"
                "</ul>"
                "</div>",
                unsafe_allow_html=True
            )

    with tab_uncertainty:
        st.subheader("⚖️ Uncertainty Prioritization Queue")
        st.caption("Ranks search terms flagged for review by financial impact and sample size variance.")
        
        uncertain_df = df_active[df_active["review_required"] == True].sort_values(by="uncertainty_impact_score", ascending=False)
        st.write(f"Total Terms in Uncertainty Queue: **{len(uncertain_df):,}**")
        
        st.dataframe(
            uncertain_df[[
                "row_id", "query", "campaign", "clicks", "cost", "conversions", 
                "conversion_value", "roas", "cpa", "uncertainty_impact_score", "reason"
            ]].head(100).style.format({
                "cost": "${:,.2f}",
                "conversion_value": "${:,.2f}",
                "cpa": "${:,.2f}",
                "roas": "{:.2f}x",
                "uncertainty_impact_score": "{:,.1f}"
            }),
            use_container_width=True,
            height=400
        )

    with tab_grouping:
        st.subheader("🔗 Near-Duplicate Query Grouping")
        st.caption("Identifies clusters of queries with identical token signatures, typo variations, and word-order permutations.")
        
        clusters = detect_near_duplicates(df_active)
        if clusters.empty:
            st.info("No near-duplicate query clusters found in the active dataset.")
        else:
            st.write(f"Identified **{len(clusters):,}** query clusters sharing semantic token signatures.")
            st.dataframe(
                clusters[[
                    "campaign", "token_signature", "variant_count", 
                    "total_cost", "total_clicks", "total_conversions", "sample_queries"
                ]].head(50).style.format({
                    "total_cost": "${:,.2f}",
                    "total_clicks": "{:,}",
                    "total_conversions": "{:,}"
                }),
                use_container_width=True,
                height=400
            )


# -----------------------------------------------------------------------------
# PAGE 4: 💸 BUDGET WASTE DETECTOR
# -----------------------------------------------------------------------------
elif page == "💸 Budget Waste Detector":
    st.title("💸 Budget Waste Detector & Negative Keyword Generator")
    st.markdown(
        "<div class='warning-box'><strong>Past Spend Analysis:</strong> Historical spend is an estimate of past inefficiency, "
        "not a guaranteed future savings claim. Suggestions should be vetted prior to ad account upload.</div>",
        unsafe_allow_html=True
    )
    
    waste_df = df_active[df_active["final_decision"] == "BLOCK"].copy()
    
    w1, w2, w3 = st.columns(3)
    w1.metric("Identified BLOCK Terms", f"{len(waste_df):,}")
    w2.metric("Total Historical Spend", f"${waste_df['cost'].sum():,.2f}")
    w3.metric("Est. Past Inefficiency", f"${waste_df['estimated_historical_waste'].sum():,.2f}")
    
    st.markdown("---")
    
    # Filter controls for waste candidates
    wf1, wf2 = st.columns(2)
    with wf1:
        w_camps = ["All"] + sorted(waste_df["campaign"].unique().tolist())
        sel_w_camp = st.selectbox("Filter Waste by Campaign", w_camps)
    with wf2:
        w_intents = ["All"] + sorted(waste_df["intent"].unique().tolist())
        sel_w_intent = st.selectbox("Filter Waste by Intent", w_intents)
        
    filtered_waste = waste_df.copy()
    if sel_w_camp != "All":
        filtered_waste = filtered_waste[filtered_waste["campaign"] == sel_w_camp]
    if sel_w_intent != "All":
        filtered_waste = filtered_waste[filtered_waste["intent"] == sel_w_intent]

    st.subheader(f"🚫 Waste Candidates ({len(filtered_waste):,} terms)")
    st.dataframe(
        filtered_waste[[
            "row_id", "query", "campaign", "ad_group", "intent", 
            "cost", "clicks", "conversions", "scope_recommendation", 
            "match_type_proposal", "reason"
        ]].style.format({
            "cost": "${:,.2f}",
            "clicks": "{:,}",
            "conversions": "{:,}"
        }),
        use_container_width=True,
        height=320
    )

    st.markdown("---")
    st.subheader("📋 Scoped Negative Keyword Proposals")
    
    n_col1, n_col2 = st.columns(2)
    with n_col1:
        st.markdown("#### Account-Level Phrase Negatives")
        st.caption("Universal negative tokens (jobs, grants, templates, personal loans)")
        account_tokens = set()
        for toks in waste_df[waste_df["is_negative_intent"]]["intent_tokens"]:
            if toks and toks != "None":
                for t in toks.split(","):
                    account_tokens.add(t.strip())
        acc_text = "\n".join([f'"{tok}"' for tok in sorted(account_tokens)])
        st.text_area("Account Negatives", value=acc_text, height=180)
        
    with n_col2:
        st.markdown("#### Campaign/Exact-Level Negatives")
        st.caption("Exact match negative queries with high zero-conversion spend")
        exact_text = "\n".join([f"[{q}]" for q in filtered_waste["query"].unique()[:100]])
        st.text_area("Exact Negatives", value=exact_text, height=180)


# -----------------------------------------------------------------------------
# PAGE 5: ✍️ HUMAN REVIEW CENTER
# -----------------------------------------------------------------------------
elif page == "✍️ Human Review Center":
    st.title("✍️ Human Review Center (Human-in-the-Loop)")
    st.markdown(
        "<div class='notice-box'>Empowers PPC strategists to audit algorithmic suggestions, override decisions, "
        "document rationale, and maintain a verifiable human audit trail.</div>",
        unsafe_allow_html=True
    )
    
    queue_select = st.radio(
        "Select Review Queue:",
        ["Pending Review (review_required = True)", "All Search Terms"],
        horizontal=True
    )
    
    if queue_select.startswith("Pending Review"):
        review_pool = df_active[df_active["review_required"] == True].copy()
    else:
        review_pool = df_active.copy()
        
    st.write(f"Total Items in Selected Queue: **{len(review_pool):,}**")
    
    if review_pool.empty:
        st.success("🎉 All items in this review queue have been audited!")
    else:
        # Select item to review
        item_options = [
            f"Row {r['row_id']}: '{r['query']}' [{r['final_decision']}] | Spend: ${r['cost']:.2f} | Conv: {r['conversions']}"
            for _, r in review_pool.head(150).iterrows()
        ]
        chosen_item = st.selectbox("Select Term to Review & Override:", item_options)
        
        if chosen_item:
            target_row_id = int(chosen_item.split(":")[0].replace("Row", "").strip())
            active_row = df_active[df_active["row_id"] == target_row_id].iloc[0]
            
            st.markdown("---")
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                st.markdown(f"### Query: `{active_row['query']}`")
                st.markdown(f"- **Campaign:** {active_row['campaign']}")
                st.markdown(f"- **Ad Group:** {active_row['ad_group']}")
                st.markdown(f"- **System Recommendation:** `{active_row['decision']}` (Confidence: {active_row['confidence']})")
                st.markdown(f"- **System Reason:** {active_row['reason']}")
                st.markdown(f"- **Intent Signal:** {active_row['intent']} ({active_row['intent_tokens']})")
            with rc2:
                st.markdown("#### Performance Metrics")
                st.write(f"**Spend:** ${active_row['cost']:.2f} | **Clicks:** {active_row['clicks']}")
                st.write(f"**Conversions:** {active_row['conversions']} | **ROAS:** {active_row['roas']:.2f}x")
                st.write(f"**CPA:** ${active_row['cpa']:.2f} | **CTR:** {active_row['ctr']:.2f}%")
                
            st.markdown("---")
            st.subheader("⚡ Human Override Form")
            
            o_col1, o_col2 = st.columns([1, 2])
            with o_col1:
                curr_dec = active_row["final_decision"]
                dec_choices = ["PROMOTE", "BLOCK", "IGNORE"]
                def_idx = dec_choices.index(curr_dec) if curr_dec in dec_choices else 0
                assigned_dec = st.selectbox("Assign Final Decision:", dec_choices, index=def_idx)
            with o_col2:
                curr_notes = active_row.get("reviewer_notes", "")
                assigned_notes = st.text_area(
                    "Reviewer Rationale & Notes:",
                    value=str(curr_notes),
                    placeholder="Enter reason for human override or account-specific business context...",
                    height=80
                )
                
            if st.button("💾 Save Decision Override", type="primary"):
                st.session_state.human_overrides[target_row_id] = {
                    "final_decision": assigned_dec,
                    "reviewer_notes": assigned_notes
                }
                st.success(f"Successfully saved override for Row {target_row_id} as **{assigned_dec}**!")
                st.rerun()

    st.markdown("---")
    st.subheader(f"📝 Active Overrides Audit ({len(st.session_state.human_overrides)} modified)")
    if st.session_state.human_overrides:
        override_rows = []
        for rid, o in st.session_state.human_overrides.items():
            r_match = df_active[df_active["row_id"] == rid]
            if not r_match.empty:
                r_item = r_match.iloc[0]
                override_rows.append({
                    "row_id": rid,
                    "query": r_item["query"],
                    "system_decision": r_item["decision"],
                    "final_decision": o["final_decision"],
                    "reviewer_notes": o["reviewer_notes"]
                })
        st.dataframe(pd.DataFrame(override_rows), use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 6: 📥 EXPORT CENTER
# -----------------------------------------------------------------------------
elif page == "📥 Export Center":
    st.title("📥 Export Center | Full Dataset & Campaign Presets")
    st.markdown(
        "<div class='notice-box'>Download the complete audited report containing <strong>every single row</strong> "
        "from the original dataset alongside system decisions, reasons, and human overrides.</div>",
        unsafe_allow_html=True
    )
    
    st.write(f"Total Rows Ready for Export: **{len(df_active):,}**")

    # Full Export Preparation
    export_df = df_active.copy()
    
    # Standard output schema preserving original fields + triage fields
    export_cols = [
        "row_id", "query", "campaign", "ad_group", "match_type",
        "impressions", "clicks", "cost", "conversions", "conversion_value",
        "ctr", "cpc", "cvr", "cpa", "roas", "intent", "decision", 
        "reason", "confidence", "review_required", "final_decision", "reviewer_notes",
        "scope_recommendation", "match_type_proposal", "estimated_historical_waste"
    ]
    ready_cols = [c for c in export_cols if c in export_df.columns]
    full_csv_bytes = export_df[ready_cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"📥 Download Complete Audited CSV ({len(df_active):,} rows)",
        data=full_csv_bytes,
        file_name="searchsense_audited_triage_report.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("---")
    st.subheader("📋 Specialized Platform Export Presets")
    
    exp1, exp2, exp3 = st.columns(3)
    
    with exp1:
        st.markdown("#### 🚫 Google Ads Negative Export")
        st.caption("Ready for Google Ads Editor negative keyword import")
        neg_export = export_df[export_df["final_decision"] == "BLOCK"][[
            "campaign", "ad_group", "query"
        ]].copy()
        neg_export.columns = ["Campaign", "Ad Group", "Keyword"]
        neg_export["Criterion Type"] = "Negative Exact"
        st.download_button(
            label="Download Negative Keywords CSV",
            data=neg_export.to_csv(index=False).encode("utf-8"),
            file_name="google_ads_negative_keywords.csv",
            mime="text/csv"
        )
        
    with exp2:
        st.markdown("#### 🚀 Exact Match Promotion Export")
        st.caption("High performers ready for exact match ad group scaling")
        promo_export = export_df[export_df["final_decision"] == "PROMOTE"][[
            "campaign", "ad_group", "query"
        ]].copy()
        promo_export.columns = ["Campaign", "Ad Group", "Keyword"]
        promo_export["Criterion Type"] = "Exact"
        st.download_button(
            label="Download Promoted Keywords CSV",
            data=promo_export.to_csv(index=False).encode("utf-8"),
            file_name="google_ads_promoted_exact.csv",
            mime="text/csv"
        )
        
    with exp3:
        st.markdown("#### ⚠️ Agency Review Queue Export")
        st.caption("Filtered list of terms flagged with review_required=True")
        rev_export = export_df[export_df["review_required"] == True][[
            "row_id", "query", "campaign", "ad_group", "clicks", "cost", 
            "conversions", "roas", "cpa", "reason"
        ]].copy()
        st.download_button(
            label="Download Review Queue CSV",
            data=rev_export.to_csv(index=False).encode("utf-8"),
            file_name="searchsense_review_queue.csv",
            mime="text/csv"
        )
