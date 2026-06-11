"""
ACIS Framework
==============
Top-level orchestrator for the Adversarial Construction Intelligence Security
(ACIS) framework. Provides a unified API for threat assessment, risk scoring,
and countermeasure recommendation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .threat_taxonomy import (
    ACISThreatTaxonomy,
    AssetCategory,
    AttackStage,
    AttackerKnowledge,
    RiskLevel,
    ThreatType,
    ThreatVector,
    WrongType,
)
from .risk_matrix import ACISRiskMatrix, RiskCell

logger = logging.getLogger(__name__)

__all__ = ["ACISFramework", "ThreatAssessmentResult", "SystemProfile"]


# ---------------------------------------------------------------------------
# Security requirement constants (Table 4 from paper)
# ---------------------------------------------------------------------------

_SECURITY_REQUIREMENTS: Dict[str, Dict[AssetCategory, str]] = {
    "Training data audit trail": {
        AssetCategory.DIS: "Mandatory",
        AssetCategory.SPS: "Mandatory",
        AssetCategory.AES: "Mandatory",
        AssetCategory.FMA: "Recommended",
    },
    "Adversarial robustness testing": {
        AssetCategory.DIS: "Recommended",
        AssetCategory.SPS: "Mandatory",
        AssetCategory.AES: "Mandatory",
        AssetCategory.FMA: "Recommended",
    },
    "Federated learning security protocol": {
        AssetCategory.DIS: "Mandatory",
        AssetCategory.SPS: "Recommended",
        AssetCategory.AES: "Mandatory",
        AssetCategory.FMA: "Optional",
    },
    "Model output watermarking": {
        AssetCategory.DIS: "Mandatory",
        AssetCategory.SPS: "Recommended",
        AssetCategory.AES: "Optional",
        AssetCategory.FMA: "Mandatory",
    },
    "Physical adversarial testing": {
        AssetCategory.DIS: "Not Required",
        AssetCategory.SPS: "Mandatory",
        AssetCategory.AES: "Mandatory",
        AssetCategory.FMA: "Recommended",
    },
    "Differential privacy in training": {
        AssetCategory.DIS: "Recommended",
        AssetCategory.SPS: "Recommended",
        AssetCategory.AES: "Recommended",
        AssetCategory.FMA: "Mandatory",
    },
    "Human-in-the-loop override": {
        AssetCategory.DIS: "Recommended",
        AssetCategory.SPS: "Mandatory",
        AssetCategory.AES: "Mandatory",
        AssetCategory.FMA: "Recommended",
    },
    "Incident response plan (AI-specific)": {
        AssetCategory.DIS: "Mandatory",
        AssetCategory.SPS: "Mandatory",
        AssetCategory.AES: "Mandatory",
        AssetCategory.FMA: "Mandatory",
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SystemProfile:
    """
    Descriptor for a construction AI system being assessed.

    Parameters
    ----------
    name            : Human-readable system name
    asset_category  : ACIS asset category (DIS / SPS / AES / FMA)
    uses_federated_learning : Whether model training is distributed across firms
    has_physical_consequence: Whether model decisions directly actuate machinery
    processes_occupant_data : Whether the system handles personal occupant data
    is_externally_queryable : Whether the model is exposed via an API
    description     : Optional free-text description
    """
    name:                     str
    asset_category:           AssetCategory
    uses_federated_learning:  bool = False
    has_physical_consequence: bool = False
    processes_occupant_data:  bool = False
    is_externally_queryable:  bool = False
    description:              str  = ""

    def construction_context_flags(self) -> List[str]:
        """Return a list of active contextual risk flags for this system."""
        flags = []
        if self.uses_federated_learning:
            flags.append("federated_learning_poisoning_risk")
        if self.has_physical_consequence:
            flags.append("physical_harm_from_ai_decision")
        if self.processes_occupant_data:
            flags.append("membership_inference_privacy_risk")
        if self.is_externally_queryable:
            flags.append("model_extraction_via_api")
        return flags


@dataclass
class ThreatAssessmentResult:
    """
    Output of ACISFramework.assess_system().

    Contains prioritised threats, mandatory security requirements,
    and contextual risk flags for a specific system profile.
    """
    system:               SystemProfile
    threats:              List[ThreatVector]
    risk_cells:           List[RiskCell]
    mandatory_controls:   List[Tuple[str, str]]   # (control, rationale)
    risk_flags:           List[str]
    overall_risk_score:   float
    overall_risk_level:   RiskLevel

    def top_threats(self, n: int = 3) -> List[ThreatVector]:
        return sorted(self.threats, key=lambda t: t.risk_score, reverse=True)[:n]

    def has_critical_exposure(self) -> bool:
        return any(c.risk_score == 5 for c in self.risk_cells)

    def to_dict(self) -> dict:
        return {
            "system":             self.system.name,
            "asset_category":     self.system.asset_category.value,
            "overall_risk_score": self.overall_risk_score,
            "overall_risk_level": self.overall_risk_level.value,
            "critical_exposure":  self.has_critical_exposure(),
            "risk_flags":         self.risk_flags,
            "top_threats": [
                {
                    "type":       t.threat_type.value,
                    "severity":   t.severity_score,
                    "likelihood": t.likelihood_score,
                    "risk_score": t.risk_score,
                    "risk_level": t.risk_level.value,
                }
                for t in self.top_threats()
            ],
            "mandatory_controls": [
                {"control": c, "rationale": r} for c, r in self.mandatory_controls
            ],
        }


# ---------------------------------------------------------------------------
# Main framework class
# ---------------------------------------------------------------------------

class ACISFramework:
    """
    The Adversarial Construction Intelligence Security (ACIS) Framework.

    Provides:
    - System threat assessment from a SystemProfile
    - Risk matrix lookup and navigation
    - Security requirements by asset category (Table 4)
    - Countermeasure recommendations (Table 3)
    - Report generation

    Usage
    -----
    >>> from acis import ACISFramework, SystemProfile, AssetCategory
    >>> framework = ACISFramework()
    >>>
    >>> system = SystemProfile(
    ...     name="PPE Vision Monitor",
    ...     asset_category=AssetCategory.SPS,
    ...     uses_federated_learning=True,
    ...     has_physical_consequence=False,
    ...     is_externally_queryable=False,
    ... )
    >>> result = framework.assess_system(system)
    >>> print(result.overall_risk_level)
    >>> framework.print_report(result)
    """

    VERSION = "1.0.0"
    PAPER_DOI = "ICCCIS-2026"

    def __init__(self) -> None:
        self.taxonomy    = ACISThreatTaxonomy()
        self.risk_matrix = ACISRiskMatrix()
        logger.info("ACISFramework v%s initialised", self.VERSION)

    # ------------------------------------------------------------------
    # Core assessment
    # ------------------------------------------------------------------

    def assess_system(self, system: SystemProfile) -> ThreatAssessmentResult:
        """
        Run a full ACIS threat assessment for a given construction AI system.

        Returns a ThreatAssessmentResult with ranked threats, mandatory
        controls, and contextual risk flags.
        """
        logger.info("Assessing system: %s (%s)", system.name, system.asset_category.value)

        # 1. Retrieve relevant threats from taxonomy
        threats = self.taxonomy.get_threats_for_asset(system.asset_category)

        # 2. Retrieve risk cells from the matrix
        risk_cells = self.risk_matrix.get_row(system.asset_category)

        # 3. Compute overall risk score (weighted mean of matrix cells)
        if risk_cells:
            overall_score = sum(c.risk_score for c in risk_cells) / len(risk_cells)
        else:
            overall_score = 0.0

        # 4. Bump score for contextual flags
        flags = system.construction_context_flags()
        if "physical_harm_from_ai_decision" in flags:
            overall_score = min(overall_score + 0.4, 5.0)
        if "federated_learning_poisoning_risk" in flags:
            overall_score = min(overall_score + 0.3, 5.0)

        overall_level = self._score_to_risk_level(overall_score)

        # 5. Determine mandatory security controls
        mandatory = self._mandatory_controls(system.asset_category)

        return ThreatAssessmentResult(
            system=system,
            threats=threats,
            risk_cells=risk_cells,
            mandatory_controls=mandatory,
            risk_flags=flags,
            overall_risk_score=round(overall_score, 3),
            overall_risk_level=overall_level,
        )

    def assess_multiple(
        self, systems: List[SystemProfile]
    ) -> List[ThreatAssessmentResult]:
        """Assess a list of systems and return results sorted by risk score."""
        results = [self.assess_system(s) for s in systems]
        return sorted(results, key=lambda r: r.overall_risk_score, reverse=True)

    # ------------------------------------------------------------------
    # Security requirements (Table 4)
    # ------------------------------------------------------------------

    def security_requirements(
        self, asset: AssetCategory
    ) -> Dict[str, str]:
        """
        Return the Table 4 security requirements for an asset category.
        Values are 'Mandatory' | 'Recommended' | 'Optional' | 'Not Required'.
        """
        return {
            control: levels[asset]
            for control, levels in _SECURITY_REQUIREMENTS.items()
        }

    def mandatory_requirements(
        self, asset: AssetCategory
    ) -> List[str]:
        """Return only the mandatory security requirements for an asset."""
        return [
            control
            for control, levels in _SECURITY_REQUIREMENTS.items()
            if levels[asset] == "Mandatory"
        ]

    # ------------------------------------------------------------------
    # Countermeasure lookup (Table 3)
    # ------------------------------------------------------------------

    def countermeasures_for_threat(
        self, threat_type: ThreatType
    ) -> List[str]:
        """Return recommended countermeasures for a specific threat type."""
        return list(self.taxonomy.get_threat(threat_type).countermeasures)

    def all_countermeasures(self) -> Dict[str, List[str]]:
        """Return all countermeasures, keyed by threat type."""
        return {
            t.threat_type.value: list(t.countermeasures)
            for t in self.taxonomy.all_threats()
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, result: ThreatAssessmentResult) -> None:
        """Print a human-readable assessment report to stdout."""
        sep = "─" * 70

        print(f"\n{sep}")
        print(f"  ACIS THREAT ASSESSMENT REPORT")
        print(f"  System : {result.system.name}")
        print(f"  Asset  : {result.system.asset_category.value.replace('_', ' ').title()}")
        print(sep)

        print(f"\n  Overall Risk: {result.overall_risk_score:.2f}/5.00  "
              f"[{result.overall_risk_level.value}]")

        if result.risk_flags:
            print(f"\n  ⚠  Contextual Risk Flags:")
            for flag in result.risk_flags:
                print(f"     • {flag.replace('_', ' ')}")

        print(f"\n  Top 3 Threats:")
        for i, t in enumerate(result.top_threats(3), 1):
            print(
                f"     {i}. [{t.risk_level.value:>8}]  {t.threat_type.value.replace('_', ' ').title()}"
                f"  (severity={t.severity_score}, likelihood={t.likelihood_score})"
            )

        print(f"\n  Mandatory Security Controls ({len(result.mandatory_controls)}):")
        for control, rationale in result.mandatory_controls:
            print(f"     ✓  {control}")
            print(f"        → {rationale}")

        print(f"\n{sep}\n")

    def export_report(
        self, result: ThreatAssessmentResult, path: Optional[Path] = None
    ) -> str:
        """Export assessment report as JSON. Returns the JSON string."""
        data = {
            "framework_version": self.VERSION,
            "paper":             self.PAPER_DOI,
            **result.to_dict(),
        }
        json_str = json.dumps(data, indent=2)
        if path is not None:
            Path(path).write_text(json_str, encoding="utf-8")
            logger.info("Report exported to %s", path)
        return json_str

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_risk_level(score: float) -> RiskLevel:
        if score >= 4.5:   return RiskLevel.CRITICAL
        elif score >= 3.5: return RiskLevel.HIGH
        elif score >= 2.5: return RiskLevel.MEDIUM
        elif score >= 1.5: return RiskLevel.LOW
        else:              return RiskLevel.VERY_LOW

    def _mandatory_controls(
        self, asset: AssetCategory
    ) -> List[Tuple[str, str]]:
        """Build (control, rationale) tuples for mandatory requirements."""
        rationales = {
            "Training data audit trail":
                "Enables detection of data poisoning; critical for federated settings.",
            "Adversarial robustness testing":
                "Validates model resistance to FGSM/PGD/physical adversarial inputs.",
            "Federated learning security protocol":
                "Defends against gradient poisoning from malicious FL participants.",
            "Model output watermarking":
                "Provides IP attribution evidence for model extraction litigation.",
            "Physical adversarial testing":
                "Construction sites allow physical patch placement; must validate.",
            "Differential privacy in training":
                "Limits membership inference and model inversion leakage.",
            "Human-in-the-loop override":
                "Ensures humans can override AI decisions in safety-critical contexts.",
            "Incident response plan (AI-specific)":
                "AI-specific IR differs from standard IT; requires dedicated playbook.",
        }
        mandatory_names = self.mandatory_requirements(asset)
        return [(name, rationales.get(name, "")) for name in mandatory_names]

    def __repr__(self) -> str:
        return (
            f"ACISFramework(v{self.VERSION} | "
            f"{len(self.taxonomy)} threats | "
            f"{self.risk_matrix})"
        )
