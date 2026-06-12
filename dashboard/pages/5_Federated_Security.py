import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Federated Security · ACIS", page_icon="🔬", layout="wide")
st.title("🔬 Federated Learning Security Simulator")
st.markdown("Simulate the ACIS §5.1 federated construction consortium scenario with Byzantine gradient poisoning.")

col1, col2 = st.columns([1, 2])
with col1:
    n_firms     = st.slider("Number of Firms",      4, 20, 8)
    n_malicious = st.slider("Malicious Firms",       0, n_firms//2, 2)
    n_rounds    = st.slider("Training Rounds",       3, 20, 8)
    boost       = st.slider("Gradient Boost Factor", 1.0, 10.0, 5.0, 0.5)
    aggregation = st.radio("Aggregation Strategy", ["fedavg (vulnerable)", "trimmed_mean (robust)"])
    run_btn     = st.button("▶ Simulate", type="primary", use_container_width=True)

with col2:
    if run_btn:
        with st.spinner("Running federated simulation..."):
            from acis.data.datasets import ConstructionBenchmark
            from acis.federated.federated import FederatedCoordinator

            bench = ConstructionBenchmark()
            ds    = bench.load_ppe(n_samples=800)
            agg   = "fedavg" if "fedavg" in aggregation else "trimmed_mean"

            coord   = FederatedCoordinator(n_rounds=n_rounds, aggregation=agg)
            clients = coord.create_consortium(n_firms=n_firms, n_malicious=n_malicious, boost_factor=boost)
            history = coord.train(clients, ds.X_train, ds.y_train, ds.X_test, ds.y_test)

        if history:
            rounds = [r.round_num for r in history]
            accs   = [r.global_accuracy * 100 for r in history]
            alerts = [len(r.alerts) for r in history]

            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rounds, y=accs, mode="lines+markers",
                                         name="Global Accuracy (%)", line=dict(color="#58a6ff")))
                fig.add_trace(go.Bar(x=rounds, y=alerts, name="Anomaly Alerts",
                                     marker_color="#f85149", yaxis="y2", opacity=0.6))
                fig.update_layout(
                    title=f"FL Training: {n_malicious}/{n_firms} malicious, {agg}",
                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                    font=dict(color="#c9d1d9"),
                    yaxis=dict(title="Accuracy (%)"),
                    yaxis2=dict(title="Alerts", overlaying="y", side="right"),
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(pd.DataFrame({"Accuracy (%)": accs}, index=rounds))

            c1, c2, c3 = st.columns(3)
            c1.metric("Final Accuracy",   f"{history[-1].global_accuracy*100:.1f}%")
            c2.metric("Total Alerts",     sum(len(r.alerts) for r in history))
            c3.metric("Malicious Ratio",  f"{n_malicious}/{n_firms}")
    else:
        st.info("Configure the consortium and click **▶ Simulate**.")
