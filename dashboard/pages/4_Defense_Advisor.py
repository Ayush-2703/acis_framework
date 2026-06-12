import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st

st.set_page_config(page_title="Defense Advisor · ACIS", page_icon="🛡️", layout="wide")
st.title("🛡️ Defense Advisor")
st.markdown("Get ACIS Table 3 countermeasures and Table 4 security requirements for your system.")

from acis.core.framework import ACISFramework
from acis.core.threat_taxonomy import AssetCategory

fw = ACISFramework()

asset_sel = st.selectbox("Select Asset Category", [
    "SPS — Site Perception Systems",
    "AES — Autonomous Execution Systems",
    "DIS — Design Intelligence Systems",
    "FMA — Facility Management AI",
])
asset_map = {
    "SPS — Site Perception Systems":      AssetCategory.SPS,
    "AES — Autonomous Execution Systems": AssetCategory.AES,
    "DIS — Design Intelligence Systems":  AssetCategory.DIS,
    "FMA — Facility Management AI":       AssetCategory.FMA,
}
asset = asset_map[asset_sel]

st.markdown("### Security Requirements (Table 4)")
reqs = fw.security_requirements(asset)
for ctrl, level in reqs.items():
    icon = {"Mandatory":"🔴","Recommended":"🟡","Optional":"🟢","Not Required":"⚪"}.get(level,"⚪")
    st.markdown(f"{icon} **{level}** — {ctrl}")

st.markdown("---")
st.markdown("### Countermeasures by Threat (Table 3)")
from acis.core.threat_taxonomy import ACISThreatTaxonomy
tax = ACISThreatTaxonomy()
threats = tax.get_threats_for_asset(asset)
for t in sorted(threats, key=lambda x: x.risk_score, reverse=True):
    with st.expander(f"[{t.risk_level.value}] {t.threat_type.value.replace('_',' ').title()}"):
        st.markdown(f"*{t.construction_example}*")
        for cm in t.countermeasures:
            st.markdown(f"- {cm}")
