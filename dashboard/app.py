"""
ACIS Interactive Dashboard
===========================
Streamlit multi-page dashboard for the ACIS framework.
Run: streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(
    page_title="ACIS Framework",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1117; }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px; margin: 6px 0;
  }
  .risk-CRITICAL { color: #f85149; font-weight: 700; }
  .risk-HIGH     { color: #e3b341; font-weight: 700; }
  .risk-MEDIUM   { color: #58a6ff; font-weight: 600; }
  .risk-LOW      { color: #3fb950; }
  h1, h2, h3    { color: #f0f6fc; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/ACIS-Framework-blue?style=for-the-badge", use_column_width=True)
    st.markdown("### 🔐 ACIS Framework")
    st.markdown("**Adversarial Construction Intelligence Security**")
    st.markdown("---")
    st.markdown("📄 [Paper (ICCCIS-2026)](https://github.com/Ayush-2703/acis-framework)")
    st.markdown("💻 [GitHub](https://github.com/Ayush-2703/acis-framework)")
    st.markdown("---")
    st.caption("Yadav, Srivastava, Singh, Ojha (2026)")

# ── Home page ───────────────────────────────────────────────────────────────
st.title("🔐 ACIS — Adversarial Construction Intelligence Security")
st.markdown("""
> **The first Python framework for AI-specific cybersecurity in construction.**
> Implements the ACIS framework from Yadav et al. (ICCCIS-2026).
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Threat Vectors",   "7",   help="Taxonomy entries from Table 1")
with col2:
    st.metric("AI Asset Types",   "4",   help="DIS · SPS · AES · FMA")
with col3:
    st.metric("Attack Modules",   "8",   help="Implemented attack simulations")
with col4:
    st.metric("Defense Modules",  "5",   help="Countermeasure implementations")

st.markdown("---")

# ── Quick demo ──────────────────────────────────────────────────────────────
st.subheader("⚡ Quick Threat Assessment")

left, right = st.columns([1, 1])

with left:
    system_name = st.text_input("System Name", value="PPE Safety Monitor")
    asset_cat   = st.selectbox("Asset Category", [
        "SPS — Site Perception Systems",
        "AES — Autonomous Execution Systems",
        "DIS — Design Intelligence Systems",
        "FMA — Facility Management AI",
    ])
    federated   = st.checkbox("Uses Federated Learning?")
    physical    = st.checkbox("Has Physical Consequence Chain?")
    queryable   = st.checkbox("Externally Queryable via API?")

with right:
    if st.button("🔍 Run Assessment", type="primary", use_container_width=True):
        from acis.core.framework import ACISFramework, SystemProfile
        from acis.core.threat_taxonomy import AssetCategory

        asset_map = {
            "SPS — Site Perception Systems":           AssetCategory.SPS,
            "AES — Autonomous Execution Systems":      AssetCategory.AES,
            "DIS — Design Intelligence Systems":       AssetCategory.DIS,
            "FMA — Facility Management AI":            AssetCategory.FMA,
        }

        fw = ACISFramework()
        profile = SystemProfile(
            name=system_name,
            asset_category=asset_map[asset_cat],
            uses_federated_learning=federated,
            has_physical_consequence=physical,
            is_externally_queryable=queryable,
        )
        result = fw.assess_system(profile)

        level = result.overall_risk_level.value
        color = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢","VERY_LOW":"🔵"}.get(level,"⚪")
        st.metric("Overall Risk Level", f"{color} {level}", f"{result.overall_risk_score:.2f}/5.00")

        if result.risk_flags:
            st.warning("**Contextual Risk Flags:**\n" + "\n".join(f"• {f.replace('_',' ')}" for f in result.risk_flags))

        st.markdown("**Top Threats:**")
        for t in result.top_threats(3):
            st.markdown(f"- `{t.threat_type.value}` — **{t.risk_level.value}** (sev={t.severity_score}, lik={t.likelihood_score})")

        st.markdown("**Mandatory Controls:**")
        for ctrl, _ in result.mandatory_controls:
            st.markdown(f"✅ {ctrl}")

    st.markdown("---")
    st.info("👈 Use the sidebar pages to explore:\n"
            "- **Threat Scanner** — full taxonomy browser\n"
            "- **Attack Simulator** — live attack demos\n"
            "- **Risk Matrix** — interactive heatmap\n"
            "- **Defense Advisor** — countermeasure recommendations\n"
            "- **Federated Security** — FL consortium simulation")
