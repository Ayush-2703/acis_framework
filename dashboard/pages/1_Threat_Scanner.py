import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Threat Scanner · ACIS", page_icon="🎯", layout="wide")
st.title("🎯 Threat Scanner")
st.markdown("Browse the full ACIS threat taxonomy. Filter by wrong type, asset category, or attack stage.")

from acis.core.threat_taxonomy import ACISThreatTaxonomy, WrongType, AssetCategory

tax = ACISThreatTaxonomy()

col1, col2 = st.columns(2)
with col1:
    wrong_filter = st.multiselect("Filter by Wrong Type", ["stealing","lying","harming"], default=["stealing","lying","harming"])
with col2:
    asset_filter = st.multiselect("Filter by Asset", ["DIS","SPS","AES","FMA"], default=["DIS","SPS","AES","FMA"])

rows = []
for t in tax.all_threats():
    if t.wrong_type.value not in wrong_filter: continue
    asset_vals = [a.value.split("_")[0].upper() for a in t.affected_assets]
    if not any(a in asset_filter for a in asset_vals): continue
    rows.append({
        "Threat": t.threat_type.value.replace("_"," ").title(),
        "Wrong":  t.wrong_type.value.title(),
        "Stage":  t.attack_stage.value.replace("_"," ").title(),
        "Knowledge": t.attacker_knowledge.value.replace("_"," ").title(),
        "Severity": t.severity_score,
        "Likelihood": t.likelihood_score,
        "Risk Score": round(t.risk_score, 2),
        "Risk Level": t.risk_level.value,
        "Assets": ", ".join(a.value.split("_")[0].upper() for a in t.affected_assets),
    })

if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=5)})

    selected = st.selectbox("Inspect a threat in detail:", df["Threat"].tolist())
    if selected:
        key = selected.lower().replace(" ","_")
        from acis.core.threat_taxonomy import ThreatType
        try:
            tt = ThreatType(key)
            tv = tax.get_threat(tt)
            st.markdown(f"### {selected}")
            st.markdown(f"**Description:** {tv.description}")
            st.markdown(f"**Construction Example:** _{tv.construction_example}_")
            st.markdown(f"**IT/OT Counterpart:** {tv.it_ot_counterpart}")
            st.markdown("**Countermeasures:**")
            for cm in tv.countermeasures:
                st.markdown(f"- {cm}")
        except Exception:
            pass
