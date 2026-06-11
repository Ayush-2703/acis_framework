"""
ACIS Attack Base Classes
========================
Abstract interfaces that all attack implementations must satisfy.
Ensures consistent API across all seven ACIS threat categories.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AttackResult:
    """
    Unified result container for all ACIS attack implementations.

    Attributes
    ----------
    attack_name         : Human-readable name of the attack
    success             : Whether the attack achieved its stated objective
    attack_success_rate : Fraction of targeted samples where the attack succeeded
    original_accuracy   : Model accuracy on clean inputs (pre-attack)
    attacked_accuracy   : Model accuracy post-attack or on adversarial inputs
    n_queries           : Number of model queries required (relevant for black-box)
    elapsed_seconds     : Wall-clock time taken
    metadata            : Attack-specific diagnostic information
    adversarial_samples : Optional array of generated adversarial inputs
    """
    attack_name:          str
    success:              bool
    attack_success_rate:  float           # 0.0 – 1.0
    original_accuracy:    float           # 0.0 – 1.0
    attacked_accuracy:    float           # 0.0 – 1.0
    n_queries:            int    = 0
    elapsed_seconds:      float  = 0.0
    metadata:             Dict[str, Any] = field(default_factory=dict)
    adversarial_samples:  Optional[np.ndarray] = None

    @property
    def accuracy_drop(self) -> float:
        """Absolute drop in accuracy caused by the attack."""
        return max(0.0, self.original_accuracy - self.attacked_accuracy)

    @property
    def perturbation_norm(self) -> Optional[float]:
        """Mean L-inf norm of adversarial perturbations, if available."""
        return self.metadata.get("mean_linf_norm")

    def summary(self) -> str:
        lines = [
            f"Attack   : {self.attack_name}",
            f"Success  : {self.success}",
            f"ASR      : {self.attack_success_rate * 100:.1f}%",
            f"Acc drop : {self.original_accuracy * 100:.1f}% → "
            f"{self.attacked_accuracy * 100:.1f}% (−{self.accuracy_drop * 100:.1f}%)",
            f"Queries  : {self.n_queries}",
            f"Time     : {self.elapsed_seconds:.2f}s",
        ]
        if self.metadata:
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "attack_name":         self.attack_name,
            "success":             self.success,
            "attack_success_rate": round(self.attack_success_rate, 4),
            "original_accuracy":   round(self.original_accuracy, 4),
            "attacked_accuracy":   round(self.attacked_accuracy, 4),
            "accuracy_drop":       round(self.accuracy_drop, 4),
            "n_queries":           self.n_queries,
            "elapsed_seconds":     round(self.elapsed_seconds, 3),
            "metadata":            self.metadata,
        }


class BaseAttack(ABC):
    """
    Abstract base class for all ACIS attack implementations.

    Sub-classes must implement:
      - ``run(model, data, **kwargs) -> AttackResult``

    Sub-classes should set:
      - ``name``         : str  (display name)
      - ``threat_type``  : str  (ThreatType value)
      - ``wrong_type``   : str  (WrongType value)
      - ``requires_training_access`` : bool
    """

    name:                       str  = "UnnamedAttack"
    threat_type:                str  = "unknown"
    wrong_type:                 str  = "unknown"
    requires_training_access:   bool = False

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._query_count: int = 0

    @abstractmethod
    def run(self, model: Any, data: Any, **kwargs) -> AttackResult:
        """Execute the attack. Returns an AttackResult."""
        ...

    def _timed_run(self, model: Any, data: Any, **kwargs) -> AttackResult:
        """Wrapper that adds timing to run()."""
        self._query_count = 0
        t0 = time.perf_counter()
        result = self.run(model, data, **kwargs)
        result.elapsed_seconds = time.perf_counter() - t0
        result.n_queries = self._query_count
        if self.verbose:
            print(f"\n[{self.name}]\n{result.summary()}")
        return result

    def _count_query(self, n: int = 1) -> None:
        """Increment the query counter (used by black-box attacks)."""
        self._query_count += n

    @staticmethod
    def _evaluate_accuracy(model: Any, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy for sklearn-compatible models."""
        preds = model.predict(X)
        return float(np.mean(preds == y))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threat={self.threat_type}, "
            f"wrong={self.wrong_type})"
        )
