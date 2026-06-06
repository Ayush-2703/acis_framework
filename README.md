# 🔐 ACIS Framework
### Adversarial Construction Intelligence Security

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-f7931e?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Paper-ICCCIS--2026-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge&logo=streamlit" />
</p>

<p align="center">
  <b>The first Python implementation of an AI-specific cybersecurity framework for construction.</b><br/>
  Code translation of the peer-reviewed ACIS framework — published at ICCCIS-2026, Amity University Lucknow.
</p>

<p align="center">
  <a href="https://acis-framework.streamlit.app"><strong>🚀 Live Demo</strong></a> ·
  <a href="#installation"><strong>Install</strong></a> ·
  <a href="#quickstart"><strong>Quickstart</strong></a> ·
  <a href="#attack-simulations"><strong>Attacks</strong></a> ·
  <a href="#citing"><strong>Cite</strong></a>
</p>

---

## 🧭 What is ACIS?

Construction sites are deploying AI at scale — autonomous robots, PPE vision monitors, BIM predictive engines, federated digital twins. Yet **no cybersecurity framework addresses AI-specific threats in this domain**.

The **ACIS (Adversarial Construction Intelligence Security) Framework** fills this gap. It maps seven categories of adversarial ML attacks across four AI asset types in construction, provides a risk matrix, countermeasure catalogue, and a lifecycle security process — all grounded in first principles of adversarial machine learning.

This repository is the **complete Python implementation** of the framework published at ICCCIS-2026.

---

## 🏗️ Framework Architecture

```
Construction AI Attack Surface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AI Asset Layer          │  Threat Layer (7 vectors)
─────────────────────── │ ─────────────────────────────
DIS  Design Intelligence│  🔴 Training Data Poisoning
SPS  Site Perception    │  🔴 Adversarial Input Attacks
AES  Autonomous Exec.   │  🟠 Model Extraction
FMA  Facility Mgmt AI   │  🟠 Backdoor / Trojan
                        │  🟡 Model Inversion
Three-Wrongs Model      │  🟡 Membership Inference
  Stealing ◄────────────┤  🟡 Supply Chain Compromise
  Lying    ◄────────────┤
  Harming  ◄────────────┘
```

---

## 📦 Repository Structure

```
acis-framework/
├── acis/                        # 🐍 Core Python package (pip-installable)
│   ├── core/
│   │   ├── threat_taxonomy.py   # 7-vector ACIS taxonomy (Tables 1–2)
│   │   ├── risk_matrix.py       # Risk matrix engine (Figure 2)
│   │   └── framework.py         # Main ACISFramework orchestrator
│   ├── attacks/
│   │   ├── data_poisoning.py    # Label flipping, targeted, gradient poisoning
│   │   ├── adversarial_inputs.py# FGSM, PGD, physical adversarial patches
│   │   ├── model_extraction.py  # Knockoff model stealing (black-box)
│   │   └── backdoor_membership.py # BadNets backdoor + shadow model MI attack
│   ├── defenses/
│   │   └── defenses.py          # Adversarial training, DP, anomaly detection
│   ├── federated/
│   │   └── federated.py         # FL consortium simulator + Byzantine defense
│   ├── data/
│   │   └── datasets.py          # PPE, BIM, Rebar synthetic datasets
│   └── cli/
│       └── main.py              # `acis` CLI tool
├── dashboard/                   # 🖥️ Streamlit 5-page interactive dashboard
│   ├── app.py
│   └── pages/                   # Threat Scanner · Attacks · Matrix · Defense · FL
├── tests/
│   └── test_acis.py             # 35+ pytest unit tests
├── .github/workflows/ci.yml     # GitHub Actions CI (3 Python versions)
├── pyproject.toml               # pip-installable package
└── CITATION.cff                 # Academic citation metadata
```

---

## ⚡ Installation

```bash
# Core framework
pip install acis-framework

# With dashboard (Streamlit + Plotly)
pip install "acis-framework[dashboard]"

# With PyTorch attacks (FGSM, PGD)
pip install "acis-framework[torch]"

# Everything
pip install "acis-framework[all]"
```

**From source:**
```bash
git clone https://github.com/Ayush-2703/acis-framework.git
cd acis-framework
pip install -e ".[all]"
```

---

## 🚀 Quickstart

### 1 — Threat Assessment (Python API)

```python
from acis import ACISFramework, SystemProfile, AssetCategory

fw = ACISFramework()

# Define a construction AI system
system = SystemProfile(
    name                    = "PPE Safety Monitor v2",
    asset_category          = AssetCategory.SPS,       # Site Perception System
    uses_federated_learning = True,                    # Consortium training
    has_physical_consequence= False,
    is_externally_queryable = False,
)

result = fw.assess_system(system)
fw.print_report(result)
```

```
──────────────────────────────────────────────────────────────────────
  ACIS THREAT ASSESSMENT REPORT
  System : PPE Safety Monitor v2
  Asset  : Site Perception Systems
──────────────────────────────────────────────────────────────────────

  Overall Risk: 4.35/5.00  [HIGH]

  ⚠  Contextual Risk Flags:
     • federated learning poisoning risk

  Top 3 Threats:
     1. [CRITICAL]  Adversarial Input Attack  (severity=4.6, likelihood=4.2)
     2. [    HIGH]  Training Data Poisoning   (severity=4.9, likelihood=3.5)
     3. [    HIGH]  Supply Chain Compromise   (severity=4.8, likelihood=3.8)

  Mandatory Security Controls (5):
     ✓  Training data audit trail
        → Enables detection of data poisoning; critical for federated settings.
     ✓  Adversarial robustness testing
        → Validates model resistance to FGSM/PGD/physical adversarial inputs.
...
```

---

### 2 — Run Attack Simulations

```python
from acis.attacks  import ConstructionPPEPoison, FGSMAttack, ModelExtractionAttack
from acis.data     import ConstructionBenchmark
from sklearn.ensemble import RandomForestClassifier

# Load synthetic PPE detection dataset
bench  = ConstructionBenchmark()
ds     = bench.load_ppe(n_samples=1200)
model  = RandomForestClassifier(n_estimators=50).fit(ds.X_train, ds.y_train)

# ── Attack 1: Training data poisoning ─────────────────────────────────────
poison = ConstructionPPEPoison(poison_rate=0.30)
result = poison.run(model, ds.as_tuple())
print(f"ASR:       {result.attack_success_rate:.1%}")
print(f"Acc drop:  {result.accuracy_drop:.1%}")

# ── Attack 2: FGSM adversarial inputs ─────────────────────────────────────
import torch, torch.nn as nn
net    = nn.Sequential(nn.Linear(20,64), nn.ReLU(), nn.Linear(64,2))
X_t    = torch.FloatTensor(ds.X_test[:200])
y_t    = torch.LongTensor(ds.y_test[:200])
fgsm   = FGSMAttack(epsilon=0.03)
result = fgsm.run(net, (X_t, y_t))
print(f"FGSM ASR:  {result.attack_success_rate:.1%}")

# ── Attack 3: Model extraction (black-box) ────────────────────────────────
extractor = ModelExtractionAttack(n_queries=500)
result    = extractor.run(model, ds.as_tuple())
print(f"Fidelity:  {result.metadata['fidelity']:.1%}")
```

---

### 3 — Federated Learning Security

```python
from acis.federated import FederatedCoordinator
from acis.data      import ConstructionBenchmark

bench = ConstructionBenchmark()
ds    = bench.load_ppe()

# 8-firm consortium, 2 malicious (Byzantine gradient poisoning)
coord   = FederatedCoordinator(n_rounds=10, aggregation="fedavg")
clients = coord.create_consortium(n_firms=8, n_malicious=2, boost_factor=5.0)
history = coord.train(clients, ds.X_train, ds.y_train, ds.X_test, ds.y_test)
coord.print_security_report(history)

# Compare FedAvg (vulnerable) vs Trimmed Mean (robust)
results = coord.compare_aggregation(clients, ds.X_train, ds.y_train,
                                     ds.X_test, ds.y_test)
```

---

### 4 — CLI Tool

```bash
# Threat assessment
acis assess --asset SPS --name "PPE Monitor" --federated --physical

# Attack simulation
acis attack --type poisoning   --dataset ppe   --poison-rate 0.25
acis attack --type fgsm        --dataset rebar --epsilon 0.05
acis attack --type extraction  --dataset bim   --n-queries 800
acis attack --type backdoor    --dataset ppe
acis attack --type federated-poison

# Risk matrix
acis matrix
acis matrix --format json

# List threats
acis list-threats
acis list-threats --wrong harming
acis list-threats --asset AES --json

# Full demo pipeline
acis demo
```

---

### 5 — Interactive Dashboard

```bash
# Launch locally
streamlit run dashboard/app.py
```

**Or visit the live demo:** [https://acis-framework.streamlit.app](https://acis-framework.streamlit.app)

Dashboard pages:
| Page | Description |
|------|------------|
| 🏠 Home | Quick assessment widget |
| 🎯 Threat Scanner | Browse taxonomy, filter by asset/wrong type |
| ⚔️ Attack Simulator | Live attack demos on construction datasets |
| 📊 Risk Matrix | Interactive heatmap (Fig. 2 from paper) |
| 🛡️ Defense Advisor | Countermeasures by asset category |
| 🔬 Federated Security | FL consortium simulation |

---

## 📊 Risk Matrix (Figure 2 / Table 2)

|                   | Data Poisoning | Adv. Input | Extraction | Inversion | Supply Chain |
|-------------------|:--------------:|:----------:|:----------:|:---------:|:------------:|
| **DIS** (Design)  | 🟡 2 LOW       | 🟢 1 V.LOW | 🔴 5 CRIT  | 🟡 3 MED  | 🟡 2 LOW     |
| **SPS** (Site)    | 🟡 3 MED       | 🔴 5 CRIT  | 🟠 4 HIGH  | 🟡 3 MED  | 🟠 4 HIGH    |
| **AES** (Robot)   | 🔴 5 CRIT      | 🔴 5 CRIT  | 🟡 2 LOW   | 🟡 3 MED  | 🔴 5 CRIT    |
| **FMA** (Facility)| 🔴 5 CRIT      | 🟠 4 HIGH  | 🟡 3 MED   | 🔴 5 CRIT | 🟠 4 HIGH    |

---

## 🔬 Attack Coverage

| Attack Class | Implementation | Dataset | ACIS Reference |
|---|---|---|---|
| Label Flipping Poison | `LabelFlippingAttack` | All | Table 1 · §4.1 |
| Targeted Class Poison | `TargetedPoisonAttack` | PPE | Table 1 · §4.2 |
| PPE-Specific Poison | `ConstructionPPEPoison` | PPE | §4.2 |
| Gradient Poison (FL) | `GradientPoisonAttack` | All | §5.1 |
| FGSM | `FGSMAttack` | All | §4.2 |
| PGD | `PGDAttack` | All | §4.2 |
| Physical Adv. Patch | `PhysicalAdversarialPatch` | SPS | §5.2 |
| Knockoff Extraction | `ModelExtractionAttack` | BIM | §5.3 |
| BIM Extraction | `BIMModelExtractionAttack` | BIM | §5.3 |
| BadNets Backdoor | `BackdoorAttack` | All | §4.3 |
| Shadow Model MI | `MembershipInferenceAttack` | FMA | §4.4 |

---

## 🛡️ Defense Coverage (Table 3)

| Defense | Class | Threat Countered |
|---|---|---|
| Adversarial Training | `AdversarialTraining` | Adversarial Inputs |
| Feature Squeezing | `InputPreprocessor` | Adversarial Inputs |
| DP-SGD Training | `DifferentialPrivacyTrainer` | MI · Model Inversion |
| Query Anomaly Detection | `QueryAnomalyDetector` | Model Extraction |
| Data Provenance Audit | `DataProvenanceAuditor` | Data Poisoning |
| Byzantine-Robust FL | `FederatedCoordinator(trimmed_mean)` | Gradient Poisoning |

---

## 🧪 Running Tests

```bash
pytest tests/ -v --tb=short                       # All tests
pytest tests/ -v -k "TestAttacks"                 # Attack tests only
pytest tests/ --cov=acis --cov-report=html        # With coverage report
```

**Test coverage:** 35+ unit tests across taxonomy, risk matrix, framework, attacks, defenses, and datasets.

---

## 📁 Datasets

Three synthetic construction datasets are included:

| Dataset | Asset | Samples | Features | Classes |
|---------|-------|---------|----------|---------|
| `PPEDataset` | SPS | 1200 | 20 | compliant / violation |
| `BIMSensorDataset` | FMA | 1500 | 24 | normal / maintenance / fault |
| `RebarPlacementDataset` | AES | 1000 | 16 | correct / minor / critical |

```python
from acis.data import ConstructionBenchmark
bench = ConstructionBenchmark()
ds    = bench.load_ppe()          # Construction-specific
ds    = bench.load_mnist_like()   # Standard benchmark
all_  = bench.load_all()          # All four datasets
```

---

## 📄 Paper

> Yadav, A., Srivastava, S., **Singh, A. K.**, & Ojha, D. (2026).
> **Cybersecurity Threats in AI-Driven Construction Systems: A Framework for
> Adversarial Machine Learning Risks in the Built Environment.**
> *2nd IETE International Conference on Computing Communication & Intelligent
> Systems (ICCCIS-2026)*, Amity University, Lucknow, India.

---

## 📖 Citing

If you use this framework in your research, please cite both the paper and software:

```bibtex
@inproceedings{yadav2026acis,
  title     = {Cybersecurity Threats in AI-Driven Construction Systems:
               A Framework for Adversarial Machine Learning Risks in the Built Environment},
  author    = {Yadav, Ankit and Srivastava, Siddhant and
               Singh, Ayush Kumar and Ojha, Devesh},
  booktitle = {2nd IETE International Conference on Computing Communication
               \& Intelligent Systems (ICCCIS-2026)},
  year      = {2026},
  month     = {March},
  address   = {Amity University, Lucknow, India},
}

@software{singh2026acis_framework,
  author    = {Singh, Ayush Kumar and Yadav, Ankit and
               Srivastava, Siddhant and Ojha, Devesh},
  title     = {{ACIS Framework}: Adversarial Construction Intelligence Security},
  year      = {2026},
  version   = {1.0.0},
  url       = {https://github.com/Ayush-2703/acis-framework},
  license   = {MIT},
}
```

---

## 🔗 Related Frameworks

| Framework | Scope | Gap Addressed by ACIS |
|-----------|-------|----------------------|
| Turk et al. (2022) | Construction cybersecurity | No AI/ML threat model |
| MITRE ATLAS | General adversarial ML | Not construction-specific |
| NIST AML Taxonomy | AI attack vocabulary | No physical consequence chain |
| ISO 19650-5 | BIM information security | No AI model as distinct asset |

---
<p align="center">
## 📬 Contact

**Ayush Kumar Singh** 

GitHub: [@Ayush-2703](https://github.com/Ayush-2703) 
LinkedIn: [AYUSH KUMAR SINGH](https://www.linkedin.com/in/ayushsingh2703)
Email: ab49ayush@gmail.com 

---

<p align="center">
  <sub> MIT License </sub>
</p>
