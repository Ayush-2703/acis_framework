"""
ACIS Test Suite
===============
pytest unit tests covering core framework, attacks, and defenses.
Run: pytest tests/ -v --tb=short
"""

# ── Attack tests ──
from acis.attack.data_poisoning import (
    LabelFlippingAttack,
    TargetedPoisonAttack,
    ConstructionPPEPoison,
)
from acis.attack.model_extraction import ModelExtractionAttack
from acis.attack.backdoor_membership import BackdoorAttack, MembershipInferenceAttack

# ── Defense tests ──
from acis.defense.defense import (
    AdversarialTraining,
    InputPreprocessor,
    DifferentialPrivacyTrainer,
    QueryAnomalyDetector,
    DataProvenanceAuditor,
)

# ── Dataset tests ──
from acis.data.datasets import (
    PPEDataset,
    BIMSensorDataset,
    RebarPlacementDataset,
    ConstructionBenchmark,
)


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def simple_data():
    X, y = make_classification(n_samples=300, n_features=20, n_classes=2,
                                random_state=42, n_informative=10)
    X = (X - X.min()) / (X.max() - X.min())
    split = 225
    return X[:split], y[:split], X[split:], y[split:]

@pytest.fixture(scope="module")
def trained_model(simple_data):
    X_tr, y_tr, _, _ = simple_data
    clf = RandomForestClassifier(n_estimators=20, random_state=42)
    clf.fit(X_tr, y_tr)
    return clf

@pytest.fixture(scope="module")
def ppe_dataset():
    from acis.data.datasets import PPEDataset
    return PPEDataset().generate(n_samples=400, random_state=42)


# ── Taxonomy tests ──────────────────────────────────────────────────────────

class TestACISThreatTaxonomy:
    def test_instantiation(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy
        tax = ACISThreatTaxonomy()
        assert len(tax) == 7

    def test_all_threat_types_present(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy, ThreatType
        tax = ACISThreatTaxonomy()
        for tt in ThreatType:
            threat = tax.get_threat(tt)
            assert threat.threat_type == tt

    def test_risk_score_bounds(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy
        tax = ACISThreatTaxonomy()
        for t in tax.all_threats():
            assert 0.0 <= t.risk_score <= 5.0
            assert 1.0 <= t.severity_score <= 5.0
            assert 1.0 <= t.likelihood_score <= 5.0

    def test_risk_level_assignment(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy, RiskLevel
        tax = ACISThreatTaxonomy()
        for t in tax.all_threats():
            assert t.risk_level in RiskLevel

    def test_filter_by_wrong(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy, WrongType
        tax = ACISThreatTaxonomy()
        stealing = tax.get_threats_by_wrong(WrongType.STEALING)
        assert all(t.wrong_type == WrongType.STEALING for t in stealing)
        assert len(stealing) >= 2

    def test_filter_by_asset(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy, AssetCategory
        tax = ACISThreatTaxonomy()
        aes_threats = tax.get_threats_for_asset(AssetCategory.AES)
        assert len(aes_threats) >= 3

    def test_serialisation(self):
        import json
        from acis.core.threat_taxonomy import ACISThreatTaxonomy
        tax = ACISThreatTaxonomy()
        js  = tax.to_json()
        obj = json.loads(js)
        assert len(obj) == 7

    def test_critical_threats_filter(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy
        tax = ACISThreatTaxonomy()
        critical = tax.get_critical_threats(severity_threshold=4.5)
        assert all(t.severity_score >= 4.5 for t in critical)

    def test_sorted_by_risk(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy
        tax = ACISThreatTaxonomy()
        threats = tax.all_threats()
        scores  = [t.risk_score for t in threats]
        assert scores == sorted(scores, reverse=True)

    def test_summary_table_shape(self):
        from acis.core.threat_taxonomy import ACISThreatTaxonomy
        tax = ACISThreatTaxonomy()
        table = tax.summary_table()
        assert len(table) == 7
        assert "Threat" in table[0]
        assert "Risk Level" in table[0]


# ── Risk matrix tests ──────────────────────────────────────────────────────

class TestACISRiskMatrix:
    def test_instantiation(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        rm = ACISRiskMatrix()
        assert rm is not None

    def test_all_cells_present(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        rm = ACISRiskMatrix()
        assert len(rm._cells) == 4 * 5  # 4 assets × 5 threats

    def test_cell_scores_in_range(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        rm = ACISRiskMatrix()
        for cell in rm._cells.values():
            assert 1 <= cell.risk_score <= 5

    def test_aes_has_critical(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        from acis.core.threat_taxonomy import AssetCategory
        rm   = ACISRiskMatrix()
        row  = rm.get_row(AssetCategory.AES)
        scores = [c.risk_score for c in row]
        assert 5 in scores, "AES should have at least one CRITICAL risk cell"

    def test_numpy_shape(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        rm  = ACISRiskMatrix()
        arr = rm.as_numpy()
        assert arr.shape == (4, 5)

    def test_highest_risk_asset(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        from acis.core.threat_taxonomy import AssetCategory
        rm = ACISRiskMatrix()
        asset, score = rm.highest_risk_asset()
        assert isinstance(asset, AssetCategory)
        assert 1.0 <= score <= 5.0

    def test_critical_cells_list(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        rm = ACISRiskMatrix()
        crit = rm.critical_cells()
        assert len(crit) > 0
        assert all(c.risk_score == 5 for c in crit)

    def test_summary_keys(self):
        from acis.core.risk_matrix import ACISRiskMatrix
        rm = ACISRiskMatrix()
        s  = rm.summary()
        for key in ["total_cells","critical_cells","mean_risk_score",
                    "highest_risk_asset","highest_risk_threat"]:
            assert key in s

    def test_dataframe(self):
        pytest.importorskip("pandas")
        from acis.core.risk_matrix import ACISRiskMatrix
        rm = ACISRiskMatrix()
        df = rm.to_dataframe()
        assert df.shape == (4, 5)


# ── Framework tests ──────────────────────────────────────────────────────────

class TestACISFramework:
    def test_assess_returns_result(self):
        from acis.core.framework import ACISFramework, SystemProfile
        from acis.core.threat_taxonomy import AssetCategory
        fw = ACISFramework()
        p  = SystemProfile(name="Test", asset_category=AssetCategory.SPS)
        r  = fw.assess_system(p)
        assert r is not None
        assert 0 < r.overall_risk_score <= 5.0

    @pytest.mark.parametrize("asset", ["DIS","SPS","AES","FMA"])
    def test_assess_all_assets(self, asset):
        from acis.core.framework import ACISFramework, SystemProfile
        from acis.core.threat_taxonomy import AssetCategory
        fw = ACISFramework()
        p  = SystemProfile(name=f"Test-{asset}",
                           asset_category=AssetCategory[asset])
        r  = fw.assess_system(p)
        assert r.overall_risk_score > 0

    def test_federated_flag_raises_score(self):
        from acis.core.framework import ACISFramework, SystemProfile
        from acis.core.threat_taxonomy import AssetCategory
        fw   = ACISFramework()
        base = fw.assess_system(SystemProfile("A", AssetCategory.SPS))
        fed  = fw.assess_system(SystemProfile("B", AssetCategory.SPS,
                                              uses_federated_learning=True))
        assert fed.overall_risk_score > base.overall_risk_score

    def test_physical_flag_raises_score(self):
        from acis.core.framework import ACISFramework, SystemProfile
        from acis.core.threat_taxonomy import AssetCategory
        fw   = ACISFramework()
        base = fw.assess_system(SystemProfile("A", AssetCategory.AES))
        phys = fw.assess_system(SystemProfile("B", AssetCategory.AES,
                                              has_physical_consequence=True))
        assert phys.overall_risk_score >= base.overall_risk_score

    def test_mandatory_controls_exist(self):
        from acis.core.framework import ACISFramework, SystemProfile
        from acis.core.threat_taxonomy import AssetCategory
        fw = ACISFramework()
        for asset in AssetCategory:
            reqs = fw.mandatory_requirements(asset)
            assert len(reqs) >= 1

    def test_export_report_json(self, tmp_path):
        import json
        from acis.core.framework import ACISFramework, SystemProfile
        from acis.core.threat_taxonomy import AssetCategory
        fw   = ACISFramework()
        p    = SystemProfile("Test", AssetCategory.FMA)
        r    = fw.assess_system(p)
        path = tmp_path / "report.json"
        js   = fw.export_report(r, path=path)
        obj  = json.loads(js)
        assert "overall_risk_level" in obj
        assert path.exists()


# ── Attack tests ─────────────────────────────────────────────────────────────

class TestAttacks:
    def test_label_flipping(self, simple_data):
        from acis.attacks.data_poisoning import LabelFlippingAttack
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        res = LabelFlippingAttack(poison_rate=0.20, verbose=False).run(clf, simple_data)
        assert 0.0 <= res.attack_success_rate <= 1.0
        assert res.original_accuracy > 0.4

    def test_targeted_poison(self, simple_data):
        from acis.attacks.data_poisoning import TargetedPoisonAttack
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        res = TargetedPoisonAttack(source_class=0, target_class=1,
                                    poison_rate=0.4, verbose=False).run(clf, simple_data)
        assert 0.0 <= res.attack_success_rate <= 1.0

    def test_ppe_poison(self, ppe_dataset):
        from acis.attacks.data_poisoning import ConstructionPPEPoison
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        res = ConstructionPPEPoison(poison_rate=0.30, verbose=False).run(
            clf, ppe_dataset.as_tuple())
        assert res.attack_name == "ConstructionPPEPoison"
        assert 0 <= res.attack_success_rate <= 1

    def test_model_extraction(self, trained_model, simple_data):
        from acis.attacks.model_extraction import ModelExtractionAttack
        res = ModelExtractionAttack(n_queries=100, verbose=False).run(
            trained_model, simple_data)
        assert 0.0 <= res.metadata["fidelity"] <= 1.0

    def test_backdoor(self, simple_data):
        from acis.attacks.backdoor_membership import BackdoorAttack
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        res = BackdoorAttack(verbose=False).run(clf, simple_data)
        assert "backdoor_success_rate" in res.metadata
        assert "stealthiness_score" in res.metadata

    def test_membership_inference(self, trained_model, simple_data):
        from acis.attacks.backdoor_membership import MembershipInferenceAttack
        res = MembershipInferenceAttack(n_shadow_models=2, verbose=False).run(
            trained_model, simple_data)
        assert "attack_accuracy" in res.metadata
        assert "advantage_over_random" in res.metadata


# ── Defense tests ─────────────────────────────────────────────────────────────

class TestDefenses:
    def test_adversarial_training_augments(self, simple_data):
        from acis.defenses.defenses import AdversarialTraining
        X_tr, y_tr, _, _ = simple_data
        at   = AdversarialTraining(epsilon=0.02)
        X_a, y_a = at.augment(X_tr, y_tr)
        assert len(X_a) > len(X_tr)
        assert len(X_a) == len(y_a)

    def test_input_preprocessor_squeeze(self, simple_data):
        from acis.defenses.defenses import InputPreprocessor
        X, _, _, _ = simple_data
        pp = InputPreprocessor(squeezing_bits=4)
        X_sq = pp.feature_squeeze(X)
        assert X_sq.shape == X.shape
        assert X_sq.max() <= X.max() + 1e-6

    def test_dp_trainer(self, simple_data):
        from acis.defenses.defenses import DifferentialPrivacyTrainer
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        X_tr, y_tr, X_te, y_te = simple_data
        dp  = DifferentialPrivacyTrainer(noise_multiplier=0.5)
        priv_model = dp.fit_private(clf, X_tr, y_tr)
        preds = priv_model.predict(X_te)
        assert len(preds) == len(y_te)

    def test_query_anomaly_detector(self):
        from acis.defenses.defenses import QueryAnomalyDetector
        det = QueryAnomalyDetector(rate_limit_per_minute=5, window_seconds=60)
        rng = np.random.default_rng(42)
        for i in range(10):
            det.log_query(rng.random(20), prediction=0, confidence=0.9)
        summary = det.summary()
        assert summary["total_alerts_raised"] > 0  # Rate limit should fire

    def test_provenance_auditor(self, simple_data):
        from acis.defenses.defenses import DataProvenanceAuditor
        X_tr, y_tr, _, _ = simple_data
        aud  = DataProvenanceAuditor()
        h    = aud.register_dataset("ds1", X_tr, y_tr, contributor="FirmA")
        ok   = aud.verify_dataset("ds1", X_tr, y_tr)
        bad  = aud.verify_dataset("ds1", X_tr + 0.1, y_tr)
        assert ok  is True
        assert bad is False


# ── Dataset tests ──────────────────────────────────────────────────────────

class TestDatasets:
    def test_ppe_shape(self):
        from acis.data.datasets import PPEDataset
        ds = PPEDataset().generate(n_samples=200)
        assert ds.X_train.shape[1] == 20
        assert len(np.unique(ds.y_train)) == 2

    def test_bim_shape(self):
        from acis.data.datasets import BIMSensorDataset
        ds = BIMSensorDataset().generate(n_samples=200)
        assert ds.X_train.shape[1] == 24
        assert len(np.unique(ds.y_train)) == 3

    def test_rebar_shape(self):
        from acis.data.datasets import RebarPlacementDataset
        ds = RebarPlacementDataset().generate(n_samples=200)
        assert ds.X_train.shape[1] == 16

    def test_benchmark_load_all(self):
        from acis.data.datasets import ConstructionBenchmark
        bench = ConstructionBenchmark()
        all_ds = bench.load_all()
        assert set(all_ds.keys()) == {"ppe","bim","rebar","benchmark"}

    def test_data_in_unit_range(self):
        from acis.data.datasets import ConstructionBenchmark
        bench = ConstructionBenchmark()
        ds = bench.load_ppe(n_samples=100)
        assert ds.X_train.min() >= -0.1
        assert ds.X_train.max() <= 1.1
