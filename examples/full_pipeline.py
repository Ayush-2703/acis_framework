"""
ACIS Full Pipeline Example
==========================
Demonstrates the complete attack-defend-assess pipeline on all
three construction datasets.

Usage:  python examples/full_pipeline.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from acis.core.framework import ACISFramework, SystemProfile
from acis.core.threat_taxonomy import AssetCategory
from acis.data.datasets import ConstructionBenchmark
from acis.attacks.data_poisoning import ConstructionPPEPoison, LabelFlippingAttack
from acis.attacks.model_extraction import ModelExtractionAttack
from acis.attacks.backdoor_membership import BackdoorAttack, MembershipInferenceAttack
from acis.defenses.defenses import AdversarialTraining, InputPreprocessor, DifferentialPrivacyTrainer
from acis.federated.federated import FederatedCoordinator

def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

def main():
    fw = ACISFramework(); bench = ConstructionBenchmark()

    section("1. Framework Overview")
    print(fw)

    section("2. Threat Assessments")
    for asset in AssetCategory:
        p = SystemProfile(name="X", asset_category=asset, uses_federated_learning=True)
        r = fw.assess_system(p)
        print(f"  {asset.value[:3].upper()}  [{r.overall_risk_level.value:>8}]  {r.overall_risk_score:.2f}/5")

    section("3. Attack Suite — PPE Dataset")
    ds = bench.load_ppe(n_samples=800)
    model = RandomForestClassifier(n_estimators=50, random_state=42).fit(ds.X_train, ds.y_train)
    data = ds.as_tuple()
    print(f"  Baseline accuracy: {model.score(ds.X_test, ds.y_test):.1%}")
    for name, atk in [
        ("LabelFlipping",    LabelFlippingAttack(poison_rate=0.20, verbose=False)),
        ("PPEPoison",        ConstructionPPEPoison(poison_rate=0.30, verbose=False)),
        ("ModelExtraction",  ModelExtractionAttack(n_queries=200, verbose=False)),
        ("Backdoor",         BackdoorAttack(verbose=False)),
        ("MembershipInf",    MembershipInferenceAttack(n_shadow_models=2, verbose=False)),
    ]:
        r = atk.run(model, data)
        print(f"  {name:<18} ASR={r.attack_success_rate:.1%}  drop={r.accuracy_drop:.1%}  ok={r.success}")

    section("4. Defenses")
    at = AdversarialTraining()
    Xa, ya = at.augment(ds.X_train, ds.y_train)
    rm = RandomForestClassifier(n_estimators=50, random_state=42).fit(Xa, ya)
    print(f"  Adversarial training: clean→robust  {model.score(ds.X_test, ds.y_test):.1%}→{rm.score(ds.X_test, ds.y_test):.1%}")

    dp = DifferentialPrivacyTrainer(noise_multiplier=0.8)
    dp.fit_private(model, ds.X_train, ds.y_train)
    print(f"  DP training: {dp.privacy_report()['privacy_guarantee']}")

    section("5. Federated Security Simulation")
    for n_m, agg in [(0,"fedavg"),(2,"fedavg"),(2,"trimmed_mean")]:
        coord = FederatedCoordinator(n_rounds=4, aggregation=agg)
        clients = coord.create_consortium(n_firms=8, n_malicious=n_m)
        hist = coord.train(clients, ds.X_train, ds.y_train, ds.X_test, ds.y_test)
        alerts = sum(len(r.alerts) for r in hist)
        print(f"  malicious={n_m}/8  {agg:<14}  final={hist[-1].global_accuracy:.2f}  alerts={alerts}")

    section("Pipeline Complete")

if __name__ == "__main__":
    main()
