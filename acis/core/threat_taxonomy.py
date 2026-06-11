"""
ACIS Threat Taxonomy
====================
Code implementation of the ACIS framework threat classification system.
Maps adversarial ML attacks across the three-wrongs model (Stealing, Lying, Harming)
as defined in: Yadav et al., "Cybersecurity Threats in AI-Driven Construction Systems",
ICCCIS-2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class WrongType(str, Enum):
    """Top-level organising principle from the three-wrongs model (Turk et al. 2022)."""
    STEALING = "stealing"
    LYING    = "lying"
    HARMING  = "harming"


class AttackStage(str, Enum):
    """When in the ML lifecycle the attack occurs."""
    TRAINING_TIME = "training_time"
    INFERENCE_TIME = "inference_time"
    DEPLOYMENT     = "deployment"
    SUPPLY_CHAIN   = "supply_chain"


class AttackerKnowledge(str, Enum):
    """Attacker's level of access to the target model (Papernot et al. taxonomy)."""
    WHITE_BOX = "white_box"   # Full model architecture + weights
    GREY_BOX  = "grey_box"    # Partial knowledge (e.g. architecture only)
    BLACK_BOX = "black_box"   # Query access only


class AssetCategory(str, Enum):
    """
    Four categories of AI assets in construction (Section 3.2, ACIS paper).
    DIS  = Design Intelligence Systems
    SPS  = Site Perception Systems
    AES  = Autonomous Execution Systems
    FMA  = Facility Management AI
    """
    DIS = "design_intelligence_systems"
    SPS = "site_perception_systems"
    AES = "autonomous_execution_systems"
    FMA = "facility_management_ai"


class ThreatType(str, Enum):
    """Seven attack categories identified in the ACIS taxonomy."""
    DATA_POISONING        = "training_data_poisoning"
    ADVERSARIAL_INPUT     = "adversarial_input_attack"
    MODEL_EXTRACTION      = "model_extraction"
    MODEL_INVERSION       = "model_inversion"
    BACKDOOR              = "backdoor_trojan_attack"
    MEMBERSHIP_INFERENCE  = "membership_inference"
    SUPPLY_CHAIN          = "supply_chain_compromise"


class RiskLevel(str, Enum):
    CRITICAL  = "CRITICAL"
    HIGH      = "HIGH"
    MEDIUM    = "MEDIUM"
    LOW       = "LOW"
    VERY_LOW  = "VERY_LOW"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThreatVector:
    """
    A single row in the ACIS threat taxonomy table.

    Attributes
    ----------
    threat_type         : ThreatType enum value
    wrong_type          : Which of the three wrongs this attack manifests as
    attack_stage        : When in the ML lifecycle the attack occurs
    attacker_knowledge  : Required attacker access level
    affected_assets     : Which AI asset categories are at risk
    description         : Technical description of the attack
    construction_example: Concrete construction-industry scenario
    it_ot_counterpart   : Analogous traditional IT/OT attack for comparison
    severity_score      : Expert-elicited severity (1–5 scale, from paper Fig. 3)
    likelihood_score    : Expert-elicited likelihood (1–5 scale, from paper Fig. 3)
    countermeasures     : List of recommended countermeasures (from Table 3)
    """
    threat_type:          ThreatType
    wrong_type:           WrongType
    attack_stage:         AttackStage
    attacker_knowledge:   AttackerKnowledge
    affected_assets:      tuple[AssetCategory, ...]
    description:          str
    construction_example: str
    it_ot_counterpart:    str
    severity_score:       float
    likelihood_score:     float
    countermeasures:      tuple[str, ...] = field(default_factory=tuple)

    @property
    def risk_score(self) -> float:
        """Composite risk score: (severity × likelihood) / 5, capped at 5."""
        return min((self.severity_score * self.likelihood_score) / 5.0, 5.0)

    @property
    def risk_level(self) -> RiskLevel:
        s = self.risk_score
        if s >= 4.5:   return RiskLevel.CRITICAL
        elif s >= 3.5: return RiskLevel.HIGH
        elif s >= 2.5: return RiskLevel.MEDIUM
        elif s >= 1.5: return RiskLevel.LOW
        else:          return RiskLevel.VERY_LOW

    def to_dict(self) -> dict:
        return {
            "threat_type":          self.threat_type.value,
            "wrong_type":           self.wrong_type.value,
            "attack_stage":         self.attack_stage.value,
            "attacker_knowledge":   self.attacker_knowledge.value,
            "affected_assets":      [a.value for a in self.affected_assets],
            "description":          self.description,
            "construction_example": self.construction_example,
            "it_ot_counterpart":    self.it_ot_counterpart,
            "severity_score":       self.severity_score,
            "likelihood_score":     self.likelihood_score,
            "risk_score":           round(self.risk_score, 3),
            "risk_level":           self.risk_level.value,
            "countermeasures":      list(self.countermeasures),
        }


# ---------------------------------------------------------------------------
# Taxonomy catalogue
# ---------------------------------------------------------------------------

class ACISThreatTaxonomy:
    """
    Immutable catalogue of all ACIS threat vectors.

    Built directly from the paper's expert-elicited scores (Fig. 3) and the
    threat/countermeasure tables (Tables 1 & 3).

    Usage
    -----
    >>> taxonomy = ACISThreatTaxonomy()
    >>> for threat in taxonomy.get_threats_by_wrong(WrongType.HARMING):
    ...     print(threat.threat_type, threat.risk_level)
    """

    def __init__(self) -> None:
        self._threats: Dict[ThreatType, ThreatVector] = self._build_catalogue()

    # ------------------------------------------------------------------
    # Catalogue definition (mirrors paper Tables 1, 3 and Figure 3)
    # ------------------------------------------------------------------

    def _build_catalogue(self) -> Dict[ThreatType, ThreatVector]:
        return {
            ThreatType.DATA_POISONING: ThreatVector(
                threat_type=ThreatType.DATA_POISONING,
                wrong_type=WrongType.HARMING,
                attack_stage=AttackStage.TRAINING_TIME,
                attacker_knowledge=AttackerKnowledge.GREY_BOX,
                affected_assets=(
                    AssetCategory.DIS, AssetCategory.SPS,
                    AssetCategory.AES, AssetCategory.FMA,
                ),
                description=(
                    "Corrupts training data to degrade model performance on specific inputs "
                    "while maintaining acceptable overall accuracy, evading standard "
                    "performance-based detection."
                ),
                construction_example=(
                    "Rogue subcontractor uploads mislabelled safety images to the shared "
                    "training repository, causing the PPE detection model to systematically "
                    "miss hard-hat violations for a specific worker class."
                ),
                it_ot_counterpart="Deleting or corrupting data files",
                severity_score=4.9,
                likelihood_score=3.5,
                countermeasures=(
                    "Data provenance tracking with cryptographic audit trails",
                    "Statistical integrity monitoring of training dataset distributions",
                    "Anomaly detection on gradient contributions in federated settings",
                    "Curated canary test sets for continuous model health monitoring",
                ),
            ),

            ThreatType.ADVERSARIAL_INPUT: ThreatVector(
                threat_type=ThreatType.ADVERSARIAL_INPUT,
                wrong_type=WrongType.LYING,
                attack_stage=AttackStage.INFERENCE_TIME,
                attacker_knowledge=AttackerKnowledge.BLACK_BOX,
                affected_assets=(AssetCategory.SPS, AssetCategory.AES),
                description=(
                    "Crafts inputs that cause model misclassification at inference time by "
                    "exploiting statistical decision boundaries. Requires no modification "
                    "to model weights. Can be physical (adversarial patches) or digital."
                ),
                construction_example=(
                    "Adversarial stickers placed on a concrete surface cause the AI quality "
                    "inspection model to classify a defective slab as structurally compliant, "
                    "bypassing site safety checks."
                ),
                it_ot_counterpart="Forging or distorting data files",
                severity_score=4.6,
                likelihood_score=4.2,
                countermeasures=(
                    "Adversarial training: augment training data with FGSM/PGD examples",
                    "Input preprocessing defences: feature squeezing, spatial smoothing",
                    "Real-time anomaly detection on input distributions",
                    "Ensemble disagreement monitoring for anomalous predictions",
                ),
            ),

            ThreatType.MODEL_EXTRACTION: ThreatVector(
                threat_type=ThreatType.MODEL_EXTRACTION,
                wrong_type=WrongType.STEALING,
                attack_stage=AttackStage.DEPLOYMENT,
                attacker_knowledge=AttackerKnowledge.BLACK_BOX,
                affected_assets=(AssetCategory.DIS, AssetCategory.FMA),
                description=(
                    "Reconstructs proprietary model decision boundaries through systematic "
                    "query-response observation, requiring only black-box API access — the "
                    "same level granted to licensed software users."
                ),
                construction_example=(
                    "A competitor systematically queries a BIM vendor's proprietary AI "
                    "clash-detection engine to reconstruct the model and re-implement it "
                    "without paying licensing fees."
                ),
                it_ot_counterpart="Exfiltrating training data or model weights",
                severity_score=3.9,
                likelihood_score=3.3,
                countermeasures=(
                    "Query rate limiting and suspicious pattern detection",
                    "Output perturbation: add calibrated noise without degrading utility",
                    "Model output watermarking for IP attribution",
                    "Revoke API access on detection; pursue IP litigation",
                ),
            ),

            ThreatType.MODEL_INVERSION: ThreatVector(
                threat_type=ThreatType.MODEL_INVERSION,
                wrong_type=WrongType.STEALING,
                attack_stage=AttackStage.DEPLOYMENT,
                attacker_knowledge=AttackerKnowledge.BLACK_BOX,
                affected_assets=(AssetCategory.DIS, AssetCategory.FMA),
                description=(
                    "Reconstructs private training data characteristics from model outputs "
                    "and confidence scores. Violates privacy of data subjects whose "
                    "records were included in training."
                ),
                construction_example=(
                    "Adversary queries a bid-pricing AI model repeatedly to infer "
                    "confidential historical project costs used in training, reconstructing "
                    "a competitor's pricing strategy."
                ),
                it_ot_counterpart="Copying files or intellectual property",
                severity_score=3.2,
                likelihood_score=3.9,
                countermeasures=(
                    "Differential privacy during training (DP-SGD)",
                    "Output quantisation: return class labels, not confidence scores",
                    "Limit sensitive attributes in training data",
                    "Anomaly detection on systematic query patterns",
                ),
            ),

            ThreatType.BACKDOOR: ThreatVector(
                threat_type=ThreatType.BACKDOOR,
                wrong_type=WrongType.LYING,
                attack_stage=AttackStage.TRAINING_TIME,
                attacker_knowledge=AttackerKnowledge.WHITE_BOX,
                affected_assets=(
                    AssetCategory.SPS, AssetCategory.AES, AssetCategory.DIS,
                ),
                description=(
                    "Embeds a hidden trigger pattern during training. The model behaves "
                    "normally on clean inputs but produces attacker-specified outputs when "
                    "a specific trigger is present. Survives standard accuracy testing."
                ),
                construction_example=(
                    "Malicious subcontractor poisons a safety monitoring model during "
                    "federated training. The model correctly detects PPE violations in "
                    "all scenarios except when a specific geometric trigger is present "
                    "on the site boundary, allowing safety bypasses on demand."
                ),
                it_ot_counterpart="Impersonation and spoofing",
                severity_score=4.7,
                likelihood_score=2.9,
                countermeasures=(
                    "Model inspection tools: Neural Cleanse, STRIP, activation clustering",
                    "Secure supply chain practices for all pre-trained models",
                    "Behavioural testing with trigger pattern sweeps",
                    "Prediction confidence distribution analysis for anomalies",
                ),
            ),

            ThreatType.MEMBERSHIP_INFERENCE: ThreatVector(
                threat_type=ThreatType.MEMBERSHIP_INFERENCE,
                wrong_type=WrongType.STEALING,
                attack_stage=AttackStage.DEPLOYMENT,
                attacker_knowledge=AttackerKnowledge.BLACK_BOX,
                affected_assets=(AssetCategory.FMA, AssetCategory.DIS),
                description=(
                    "Determines whether a specific data record was used in model training "
                    "by exploiting overfitting-induced confidence gaps between members and "
                    "non-members of the training set."
                ),
                construction_example=(
                    "Adversary queries an occupant behaviour model in a smart building "
                    "to infer the presence, schedule, and movement patterns of specific "
                    "individuals, constituting a privacy violation."
                ),
                it_ot_counterpart="Copying files or intellectual property",
                severity_score=3.2,
                likelihood_score=3.9,
                countermeasures=(
                    "Differential privacy during training reduces membership signal",
                    "Model regularisation to prevent overfitting",
                    "Output confidence truncation or quantisation",
                    "Shadow model detection: monitor for training-like query distributions",
                ),
            ),

            ThreatType.SUPPLY_CHAIN: ThreatVector(
                threat_type=ThreatType.SUPPLY_CHAIN,
                wrong_type=WrongType.HARMING,
                attack_stage=AttackStage.SUPPLY_CHAIN,
                attacker_knowledge=AttackerKnowledge.WHITE_BOX,
                affected_assets=(
                    AssetCategory.DIS, AssetCategory.SPS,
                    AssetCategory.AES, AssetCategory.FMA,
                ),
                description=(
                    "Compromises AI model artefacts before deployment through malicious "
                    "pre-trained model weights, poisoned model repositories, or trojanised "
                    "ML framework dependencies."
                ),
                construction_example=(
                    "Malicious pickle files inserted into a publicly accessible model "
                    "weight repository execute arbitrary code when a construction firm "
                    "downloads and loads the model for their BIM platform."
                ),
                it_ot_counterpart="Software supply chain attacks (SolarWinds-style)",
                severity_score=4.8,
                likelihood_score=3.8,
                countermeasures=(
                    "Verify model provenance and integrity hashes before loading",
                    "Maintain model bill of materials (MBOM) for all deployed models",
                    "Avoid unverified pre-trained models from public repositories",
                    "Regular red-team exercises and model performance monitoring",
                ),
            ),
        }

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_threat(self, threat_type: ThreatType) -> ThreatVector:
        """Return a single ThreatVector by type."""
        return self._threats[threat_type]

    def get_threats_by_wrong(self, wrong: WrongType) -> List[ThreatVector]:
        """Return all threats belonging to a given wrong category."""
        return [t for t in self._threats.values() if t.wrong_type == wrong]

    def get_threats_for_asset(self, asset: AssetCategory) -> List[ThreatVector]:
        """Return all threats that affect a given AI asset category."""
        return [t for t in self._threats.values() if asset in t.affected_assets]

    def get_by_stage(self, stage: AttackStage) -> List[ThreatVector]:
        """Return all threats occurring at a given attack stage."""
        return [t for t in self._threats.values() if t.attack_stage == stage]

    def get_critical_threats(
        self, severity_threshold: float = 4.5
    ) -> List[ThreatVector]:
        """Return threats above a given severity threshold, sorted descending."""
        return sorted(
            [t for t in self._threats.values() if t.severity_score >= severity_threshold],
            key=lambda t: t.risk_score,
            reverse=True,
        )

    def all_threats(self) -> List[ThreatVector]:
        """Return all threat vectors, sorted by descending risk score."""
        return sorted(
            self._threats.values(), key=lambda t: t.risk_score, reverse=True
        )

    def to_dict(self) -> dict:
        """Serialise the full taxonomy to a JSON-compatible dict."""
        return {k.value: v.to_dict() for k, v in self._threats.items()}

    def to_json(self, indent: int = 2) -> str:
        """Serialise the full taxonomy to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary_table(self) -> List[dict]:
        """
        Return a flat list of dicts suitable for pandas DataFrame construction
        or tabular display.
        """
        return [
            {
                "Threat":           t.threat_type.value,
                "Wrong":            t.wrong_type.value,
                "Stage":            t.attack_stage.value,
                "Knowledge":        t.attacker_knowledge.value,
                "Severity (1–5)":   t.severity_score,
                "Likelihood (1–5)": t.likelihood_score,
                "Risk Score":       round(t.risk_score, 2),
                "Risk Level":       t.risk_level.value,
                "Assets Affected":  len(t.affected_assets),
            }
            for t in self.all_threats()
        ]

    def __len__(self) -> int:
        return len(self._threats)

    def __repr__(self) -> str:
        return (
            f"ACISThreatTaxonomy("
            f"{len(self)} threats, "
            f"{sum(1 for t in self._threats.values() if t.risk_level == RiskLevel.CRITICAL)} critical)"
        )
