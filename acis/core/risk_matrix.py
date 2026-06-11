"""
ACIS Risk Matrix
================
Computes and stores the risk matrix from Table 2 and Figure 2 of the ACIS paper.
Provides programmatic access to per-asset, per-threat risk levels and scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .threat_taxonomy import (
    ACISThreatTaxonomy,
    AssetCategory,
    ThreatType,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Risk level constants matching paper Figure 2
# ---------------------------------------------------------------------------

# fmt: off
_PAPER_RISK_TABLE: Dict[Tuple[AssetCategory, ThreatType], int] = {
    # (asset, threat_type) -> risk_score  (1=Very Low, 5=Critical)
    # ----------------------------------------------------------------
    # Design Intelligence Systems (DIS)
    (AssetCategory.DIS, ThreatType.DATA_POISONING):       2,
    (AssetCategory.DIS, ThreatType.ADVERSARIAL_INPUT):    1,
    (AssetCategory.DIS, ThreatType.MODEL_EXTRACTION):     5,
    (AssetCategory.DIS, ThreatType.MODEL_INVERSION):      3,
    (AssetCategory.DIS, ThreatType.SUPPLY_CHAIN):         2,
    # Site Perception Systems (SPS)
    (AssetCategory.SPS, ThreatType.DATA_POISONING):       3,
    (AssetCategory.SPS, ThreatType.ADVERSARIAL_INPUT):    5,
    (AssetCategory.SPS, ThreatType.MODEL_EXTRACTION):     4,
    (AssetCategory.SPS, ThreatType.MODEL_INVERSION):      3,
    (AssetCategory.SPS, ThreatType.SUPPLY_CHAIN):         4,
    # BIM Models (mapped to DIS for full matrix coverage)
    # Autonomous Execution Systems (AES)
    (AssetCategory.AES, ThreatType.DATA_POISONING):       5,
    (AssetCategory.AES, ThreatType.ADVERSARIAL_INPUT):    5,
    (AssetCategory.AES, ThreatType.MODEL_EXTRACTION):     2,
    (AssetCategory.AES, ThreatType.MODEL_INVERSION):      3,
    (AssetCategory.AES, ThreatType.SUPPLY_CHAIN):         5,
    # Facility Management AI (FMA)
    (AssetCategory.FMA, ThreatType.DATA_POISONING):       5,
    (AssetCategory.FMA, ThreatType.ADVERSARIAL_INPUT):    4,
    (AssetCategory.FMA, ThreatType.MODEL_EXTRACTION):     3,
    (AssetCategory.FMA, ThreatType.MODEL_INVERSION):      5,
    (AssetCategory.FMA, ThreatType.SUPPLY_CHAIN):         4,
}
# fmt: on

_RISK_LABELS = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}

# Table 2 threat levels (HIGH / CRITICAL / MEDIUM)
_THREAT_MATRIX: Dict[Tuple[AssetCategory, ThreatType], str] = {
    (AssetCategory.DIS, ThreatType.DATA_POISONING):    "HIGH",
    (AssetCategory.DIS, ThreatType.ADVERSARIAL_INPUT): "MEDIUM",
    (AssetCategory.DIS, ThreatType.MODEL_EXTRACTION):  "HIGH",
    (AssetCategory.DIS, ThreatType.SUPPLY_CHAIN):      "HIGH",
    (AssetCategory.SPS, ThreatType.DATA_POISONING):    "HIGH",
    (AssetCategory.SPS, ThreatType.ADVERSARIAL_INPUT): "CRITICAL",
    (AssetCategory.SPS, ThreatType.MODEL_EXTRACTION):  "MEDIUM",
    (AssetCategory.SPS, ThreatType.SUPPLY_CHAIN):      "HIGH",
    (AssetCategory.AES, ThreatType.DATA_POISONING):    "CRITICAL",
    (AssetCategory.AES, ThreatType.ADVERSARIAL_INPUT): "CRITICAL",
    (AssetCategory.AES, ThreatType.MODEL_EXTRACTION):  "MEDIUM",
    (AssetCategory.AES, ThreatType.SUPPLY_CHAIN):      "HIGH",
    (AssetCategory.FMA, ThreatType.DATA_POISONING):    "MEDIUM",
    (AssetCategory.FMA, ThreatType.ADVERSARIAL_INPUT): "HIGH",
    (AssetCategory.FMA, ThreatType.MODEL_EXTRACTION):  "HIGH",
    (AssetCategory.FMA, ThreatType.SUPPLY_CHAIN):      "MEDIUM",
}


# ---------------------------------------------------------------------------
# Cell and matrix data structures
# ---------------------------------------------------------------------------

@dataclass
class RiskCell:
    """A single cell in the ACIS risk matrix."""
    asset:          AssetCategory
    threat_type:    ThreatType
    risk_score:     int            # 1–5 (from paper Fig. 2)
    threat_level:   str            # CRITICAL / HIGH / MEDIUM / LOW (Table 2)

    @property
    def risk_label(self) -> str:
        return _RISK_LABELS.get(self.risk_score, "Unknown")

    def to_dict(self) -> dict:
        return {
            "asset":        self.asset.value,
            "threat":       self.threat_type.value,
            "risk_score":   self.risk_score,
            "risk_label":   self.risk_label,
            "threat_level": self.threat_level,
        }


class ACISRiskMatrix:
    """
    Programmatic representation of the ACIS risk matrix (Table 2 / Fig. 2).

    Provides lookup, slicing, and aggregation operations over all
    (asset × threat) combinations identified in the paper.

    Usage
    -----
    >>> matrix = ACISRiskMatrix()
    >>> cell = matrix.get_cell(AssetCategory.AES, ThreatType.ADVERSARIAL_INPUT)
    >>> print(cell.risk_label)   # 'Critical'
    >>> print(matrix.highest_risk_asset())
    >>> df = matrix.to_dataframe()
    """

    ASSETS:  List[AssetCategory] = list(AssetCategory)
    THREATS: List[ThreatType]    = [
        ThreatType.DATA_POISONING,
        ThreatType.ADVERSARIAL_INPUT,
        ThreatType.MODEL_EXTRACTION,
        ThreatType.MODEL_INVERSION,
        ThreatType.SUPPLY_CHAIN,
    ]

    def __init__(self) -> None:
        self._taxonomy = ACISThreatTaxonomy()
        self._cells: Dict[Tuple[AssetCategory, ThreatType], RiskCell] = {}
        self._build_matrix()

    def _build_matrix(self) -> None:
        for asset in self.ASSETS:
            for threat in self.THREATS:
                key = (asset, threat)
                score = _PAPER_RISK_TABLE.get(key, self._compute_fallback(asset, threat))
                level = _THREAT_MATRIX.get(key, self._score_to_level(score))
                self._cells[key] = RiskCell(
                    asset=asset,
                    threat_type=threat,
                    risk_score=score,
                    threat_level=level,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fallback(asset: AssetCategory, threat: ThreatType) -> int:
        """Derive a score for cells not explicitly in the paper."""
        taxonomy = ACISThreatTaxonomy()
        tv = taxonomy.get_threat(threat)
        base = tv.risk_score
        # AES gets a +1 physical-consequence bonus
        if asset == AssetCategory.AES:
            base = min(base + 0.5, 5.0)
        return max(1, min(5, round(base)))

    @staticmethod
    def _score_to_level(score: int) -> str:
        if score >= 5:    return "CRITICAL"
        elif score >= 4:  return "HIGH"
        elif score >= 3:  return "MEDIUM"
        else:             return "LOW"

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_cell(
        self, asset: AssetCategory, threat: ThreatType
    ) -> Optional[RiskCell]:
        """Return the RiskCell for a specific (asset, threat) pair."""
        return self._cells.get((asset, threat))

    def get_row(self, asset: AssetCategory) -> List[RiskCell]:
        """Return all risk cells for a given asset category."""
        return [
            self._cells[(asset, t)]
            for t in self.THREATS
            if (asset, t) in self._cells
        ]

    def get_column(self, threat: ThreatType) -> List[RiskCell]:
        """Return all risk cells for a given threat type."""
        return [
            self._cells[(a, threat)]
            for a in self.ASSETS
            if (a, threat) in self._cells
        ]

    def critical_cells(self) -> List[RiskCell]:
        """Return all cells with CRITICAL risk level."""
        return [c for c in self._cells.values() if c.risk_score == 5]

    def highest_risk_asset(self) -> Tuple[AssetCategory, float]:
        """Return the asset category with the highest mean risk score."""
        scores = {
            asset: np.mean([self._cells[(asset, t)].risk_score
                            for t in self.THREATS if (asset, t) in self._cells])
            for asset in self.ASSETS
        }
        best = max(scores, key=lambda a: scores[a])
        return best, round(float(scores[best]), 3)

    def highest_risk_threat(self) -> Tuple[ThreatType, float]:
        """Return the threat type with the highest mean risk score across assets."""
        scores = {
            threat: np.mean([self._cells[(a, threat)].risk_score
                             for a in self.ASSETS if (a, threat) in self._cells])
            for threat in self.THREATS
        }
        best = max(scores, key=lambda t: scores[t])
        return best, round(float(scores[best]), 3)

    def as_numpy(self) -> np.ndarray:
        """
        Return the risk matrix as a 2-D NumPy array.
        Rows = assets (DIS, SPS, AES, FMA)
        Columns = threats (Poisoning, Adversarial, Extraction, Inversion, Supply Chain)
        """
        grid = np.zeros((len(self.ASSETS), len(self.THREATS)), dtype=int)
        for i, asset in enumerate(self.ASSETS):
            for j, threat in enumerate(self.THREATS):
                cell = self._cells.get((asset, threat))
                if cell:
                    grid[i, j] = cell.risk_score
        return grid

    def to_dataframe(self):
        """Return the risk matrix as a pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required: pip install pandas") from e

        grid = self.as_numpy()
        rows = [a.value.replace("_", " ").title() for a in self.ASSETS]
        cols = [t.value.replace("_", " ").title() for t in self.THREATS]
        return pd.DataFrame(grid, index=rows, columns=cols)

    def summary(self) -> dict:
        """Return high-level statistics about the risk matrix."""
        all_scores = [c.risk_score for c in self._cells.values()]
        top_asset, top_asset_score = self.highest_risk_asset()
        top_threat, top_threat_score = self.highest_risk_threat()
        return {
            "total_cells":         len(self._cells),
            "critical_cells":      sum(1 for s in all_scores if s == 5),
            "high_cells":          sum(1 for s in all_scores if s == 4),
            "mean_risk_score":     round(float(np.mean(all_scores)), 3),
            "highest_risk_asset":  top_asset.value,
            "top_asset_score":     top_asset_score,
            "highest_risk_threat": top_threat.value,
            "top_threat_score":    top_threat_score,
        }

    def to_dict(self) -> dict:
        """Serialise the full matrix to a JSON-compatible dict."""
        return {
            f"{a.value}::{t.value}": cell.to_dict()
            for (a, t), cell in self._cells.items()
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ACISRiskMatrix("
            f"{s['total_cells']} cells | "
            f"{s['critical_cells']} critical | "
            f"mean={s['mean_risk_score']:.2f})"
        )
