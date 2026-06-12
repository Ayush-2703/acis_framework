import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import pandas as pd, numpy as np

st.set_page_config(page_title="Risk Matrix · ACIS", page_icon="📊", layout="wide")
st.title("📊 ACIS Risk Matrix")
st.markdown("Interactive version of **Figure 2 / Table 2** from the ACIS paper. Scores: 1=Very Low → 5=Critical.")

from acis.core.risk_matrix import ACISRiskMatrix

rm  = _rm = ACISRiskMatrix()
arr = rm.as_numpy()
s   = rm.summary()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Critical Cells", s["critical_cells"])
col2.metric("High Cells",     s["high_cells"])
col3.metric("Mean Score",     f"{s['mean_risk_score']:.2f}")
col4.metric("Highest Asset",  s["highest_risk_asset"].split("_")[0].upper())

try:
    import plotly.graph_objects as go
    assets  = ["DIS", "SPS", "AES", "FMA"]
    threats = ["Data\nPoison", "Adversarial\nInput", "Model\nExtract", "Model\nInversion", "Supply\nChain"]
    colors  = ["#3fb950","#8b949e","#e3b341","#f0883e","#f85149"]

    fig = go.Figure(data=go.Heatmap(
        z=arr, x=threats, y=assets,
        colorscale=[[0,"#0d1117"],[0.2,"#1a7f37"],[0.4,"#9e6a03"],
                    [0.6,"#b08800"],[0.8,"#bc4c00"],[1.0,"#da3633"]],
        zmin=1, zmax=5, text=arr,
        texttemplate="%{text}", textfont={"size":16,"color":"white"},
        showscale=True,
        colorbar=dict(title="Risk Level", tickvals=[1,2,3,4,5],
                      ticktext=["Very Low","Low","Medium","High","Critical"]),
    ))
    fig.update_layout(
        title="ACIS Risk Matrix — Asset × Threat",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#c9d1d9"),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    df = rm.to_dataframe()
    st.dataframe(df, use_container_width=True)

with st.expander("📋 Raw risk matrix data"):
    st.dataframe(rm.to_dataframe(), use_container_width=True)
