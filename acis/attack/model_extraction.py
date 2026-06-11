"""
Model Extraction Attack
=======================
Black-box model stealing via systematic query-response observation.
No access to model weights required — only API-level query access,
the same level granted to licensed software users.

ACIS mapping
  WrongType.STEALING, AttackStage.DEPLOYMENT, ThreatType.MODEL_EXTRACTION

Construction scenario
  A competitor queries a BIM vendor's proprietary AI clash-detection or
  predictive-maintenance engine to reconstruct an equivalent model,
  effectively stealing millions in R&D investment.

Reference
  Tramèr et al. "Stealing Machine Learning Models via Prediction APIs." USENIX 2016.
  Truong et al. "Data-Free Model Extraction." CVPR 2021.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from .base import BaseAttack, AttackResult


class ModelExtractionAttack(BaseAttack):
    """
    Knockoff-style Model Extraction Attack.

    Trains a substitute (surrogate) model by:
      1. Generating or collecting a query set of inputs
      2. Querying the victim model for soft/hard labels
      3. Training the surrogate on those stolen labels

    The surrogate is then compared against the victim to measure
    fidelity (agreement rate) and accuracy parity.

    Parameters
    ----------
    surrogate_model : sklearn-compatible classifier for the surrogate
    n_queries       : Number of query points to use for extraction
    use_soft_labels : Whether to use probability vectors (requires predict_proba)
    random_state    : Reproducibility seed
    """

    name        = "ModelExtractionAttack"
    threat_type = "model_extraction"
    wrong_type  = "stealing"
    requires_training_access = False  # Black-box only

    def __init__(
        self,
        surrogate_model: Any = None,
        n_queries:       int  = 500,
        use_soft_labels: bool = True,
        random_state:    int  = 42,
        verbose:         bool = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.surrogate_model = surrogate_model or MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=random_state,
        )
        self.n_queries       = n_queries
        self.use_soft_labels = use_soft_labels
        self.rng = np.random.default_rng(random_state)

    def run(
        self,
        model: Any,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        query_distribution: str = "uniform",
        **kwargs,
    ) -> AttackResult:
        """
        Parameters
        ----------
        model              : sklearn victim model (the proprietary model being stolen)
        data               : (X_train, y_train, X_test, y_test)
                             X_train used for the query distribution range;
                             X_test used for evaluation.
        query_distribution : 'uniform'  — sample queries from feature range
                             'training' — sample from training data distribution
        """
        X_train, y_train, X_test, y_test = data

        # Victim model accuracy on test set
        original_acc = self._evaluate_accuracy(model, X_test, y_test)

        # Step 1: Generate query set
        X_query = self._generate_queries(X_train, query_distribution)
        self._count_query(len(X_query))

        # Step 2: Label queries using victim model (black-box access)
        if self.use_soft_labels and hasattr(model, "predict_proba"):
            y_stolen = model.predict_proba(X_query)  # Soft labels
            y_stolen_hard = model.predict(X_query)
        else:
            y_stolen_hard = model.predict(X_query)
            y_stolen = y_stolen_hard

        # Step 3: Train surrogate on stolen labels
        surrogate = clone(self.surrogate_model)
        if self.use_soft_labels and y_stolen.ndim == 2:
            # Use argmax of soft labels as training targets
            surrogate.fit(X_query, y_stolen.argmax(axis=1))
        else:
            surrogate.fit(X_query, y_stolen_hard)

        # Evaluate surrogate
        surrogate_acc = self._evaluate_accuracy(surrogate, X_test, y_test)

        # Fidelity: agreement between surrogate and victim on test set
        victim_preds   = model.predict(X_test)
        surrogate_preds = surrogate.predict(X_test)
        fidelity = float(np.mean(victim_preds == surrogate_preds))
        self._count_query(len(X_test))

        # ASR: fidelity is the "success" metric for model extraction
        asr = fidelity

        return AttackResult(
            attack_name=self.name,
            success=(fidelity > 0.80),
            attack_success_rate=round(asr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(surrogate_acc, 4),  # Surrogate's accuracy
            metadata={
                "n_queries":            self.n_queries,
                "fidelity":             round(fidelity, 4),
                "surrogate_accuracy":   round(surrogate_acc, 4),
                "victim_accuracy":      round(original_acc, 4),
                "accuracy_parity":      round(surrogate_acc / max(original_acc, 1e-9), 4),
                "query_distribution":   query_distribution,
                "use_soft_labels":      self.use_soft_labels,
                "surrogate_type":       type(self.surrogate_model).__name__,
                "construction_context": (
                    f"Attacker issued {self.n_queries} queries to the proprietary AI API "
                    f"and trained a {type(self.surrogate_model).__name__} surrogate. "
                    f"Surrogate achieves {fidelity * 100:.1f}% fidelity (agreement rate) "
                    "with the victim model — effectively stealing IP accessible only "
                    "via licensed black-box API queries."
                ),
            },
        )

    def _generate_queries(
        self, X_reference: np.ndarray, mode: str
    ) -> np.ndarray:
        """Generate query inputs for the extraction attack."""
        if mode == "training":
            # Sample with replacement from training distribution
            idx = self.rng.choice(len(X_reference), size=self.n_queries, replace=True)
            X_query = X_reference[idx].copy()
            # Add small noise to increase diversity
            noise = self.rng.normal(0, 0.05, X_query.shape)
            X_query = X_query + noise
        else:
            # Uniform sampling within feature range
            lo = X_reference.min(axis=0)
            hi = X_reference.max(axis=0)
            X_query = self.rng.uniform(lo, hi, size=(self.n_queries, X_reference.shape[1]))

        return X_query.astype(float)


class BIMModelExtractionAttack(ModelExtractionAttack):
    """
    BIM-specific model extraction attack.

    Targets a proprietary BIM AI engine (clash detection, cost estimation,
    or structural optimisation) by systematically querying it as a licensed
    user and reconstructing a functionally equivalent surrogate.

    Models the scenario from ACIS Section 5.3: the intellectual property
    boundary of a commercial AI model is no more protective than a software
    trial license from the adversary's perspective.
    """

    name = "BIMModelExtractionAttack"

    def __init__(
        self,
        n_queries:   int  = 1000,
        random_state: int = 42,
        verbose:      bool = True,
    ) -> None:
        super().__init__(
            surrogate_model=RandomForestClassifier(
                n_estimators=100, random_state=random_state
            ),
            n_queries=n_queries,
            use_soft_labels=True,
            random_state=random_state,
            verbose=verbose,
        )

    def run(self, model, data, **kwargs) -> AttackResult:
        result = super().run(model, data, query_distribution="training")
        result.attack_name = self.name
        result.metadata["construction_context"] = (
            "BIM platform attack: licensed user systematically queries the proprietary "
            f"AI engine {self.n_queries} times, building a surrogate with "
            f"{result.metadata['fidelity'] * 100:.1f}% decision-boundary fidelity. "
            "Attack requires only the access level granted by a standard software license."
        )
        return result
