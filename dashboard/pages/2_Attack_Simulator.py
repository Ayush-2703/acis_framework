import sys, os; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import streamlit as st
import pandas as pd, numpy as np

st.set_page_config(page_title="Attack Simulator · ACIS", page_icon="⚔️", layout="wide")
st.title("⚔️ Attack Simulator")
st.markdown("Run live ACIS attack simulations on synthetic construction datasets.")

from acis.data.datasets import ConstructionBenchmark
from sklearn.ensemble import RandomForestClassifier

col1, col2 = st.columns([1,2])

with col1:
    attack_type = st.selectbox("Attack Type", [
        "Label Flipping (Poisoning)",
        "Targeted PPE Poison",
        "Model Extraction",
        "Backdoor / Trojan",
        "Membership Inference",
    ])
    dataset = st.selectbox("Dataset", ["PPE Detection","BIM Sensor Streams","Rebar Placement","Benchmark"])
    ds_map  = {"PPE Detection":"ppe","BIM Sensor Streams":"bim","Rebar Placement":"rebar","Benchmark":"benchmark"}
    poison_rate = st.slider("Poison Rate", 0.05, 0.50, 0.20, 0.05) if "Poison" in attack_type else 0.20
    n_queries   = st.slider("Query Budget", 100, 2000, 500, 100) if "Extraction" in attack_type else 500
    run_btn = st.button("▶ Run Attack", type="primary", use_container_width=True)

with col2:
    if run_btn:
        with st.spinner("Loading dataset and training baseline model..."):
            bench = ConstructionBenchmark()
            ds    = getattr(bench, f"load_{ds_map[dataset]}" if ds_map[dataset]!="benchmark" else "load_mnist_like")()
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(ds.X_train, ds.y_train)
            data  = ds.as_tuple()

        with st.spinner(f"Running {attack_type}..."):
            try:
                if "Label Flipping" in attack_type:
                    from acis.attacks.data_poisoning import LabelFlippingAttack
                    result = LabelFlippingAttack(poison_rate=poison_rate).run(model, data)
                elif "PPE Poison" in attack_type:
                    from acis.attacks.data_poisoning import ConstructionPPEPoison
                    result = ConstructionPPEPoison(poison_rate=poison_rate).run(model, data)
                elif "Extraction" in attack_type:
                    from acis.attacks.model_extraction import ModelExtractionAttack
                    result = ModelExtractionAttack(n_queries=n_queries).run(model, data)
                elif "Backdoor" in attack_type:
                    from acis.attacks.backdoor_membership import BackdoorAttack
                    result = BackdoorAttack().run(model, data)
                elif "Membership" in attack_type:
                    from acis.attacks.backdoor_membership import MembershipInferenceAttack
                    result = MembershipInferenceAttack(n_shadow_models=3).run(model, data)

                c1, c2, c3 = st.columns(3)
                c1.metric("Attack Success Rate", f"{result.attack_success_rate*100:.1f}%")
                c2.metric("Original Accuracy",   f"{result.original_accuracy*100:.1f}%")
                c3.metric("Post-Attack Accuracy", f"{result.attacked_accuracy*100:.1f}%",
                          delta=f"{(result.attacked_accuracy-result.original_accuracy)*100:.1f}%")

                st.success("✅ Attack complete" if result.success else "⚠️ Attack partial")
                with st.expander("📋 Full metadata"):
                    st.json(result.metadata)
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
    else:
        st.info("Configure attack parameters and click **▶ Run Attack** to simulate.")
        st.markdown("""
| Attack | ACIS Threat | Wrong Type |
|--------|------------|------------|
| Label Flipping | Training Data Poisoning | Harming |
| Targeted PPE Poison | Training Data Poisoning | Harming |
| Model Extraction | Model Extraction | Stealing |
| Backdoor/Trojan | Backdoor Attack | Lying |
| Membership Inference | Membership Inference | Stealing |
""")
