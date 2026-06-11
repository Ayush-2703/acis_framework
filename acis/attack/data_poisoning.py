"""
Data Poisoning Attacks
======================
Implements training-time poisoning attacks relevant to construction AI systems:

  1. LabelFlippingAttack   — randomly or selectively flips labels in training data
  2. TargetedPoisonAttack  — poisons a specific target class (e.g. "safe" → "unsafe")
  3. GradientPoisonAttack  — simulates federated learning gradient manipulation
  4. ConstructionPPEPoison — domain-specific: poisons a PPE safety detection dataset

These attacks map to the ACIS taxonomy entry:
  WrongType.HARMING, AttackStage.TRAINING_TIME, ThreatType.DATA_POISONING
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from .base import BaseAttack, AttackResult


class LabelFlippingAttack(BaseAttack):
    """
    Label Flipping Poisoning Attack.

    Randomly corrupts a fraction of training labels to degrade model accuracy.
    Simplest form of data poisoning; represents a rogue data-contributor scenario
    in a construction consortium.

    Parameters
    ----------
    poison_rate : float
        Fraction of training samples to corrupt (0 < poison_rate < 1).
    random_state : int
        Random seed for reproducibility.

    Reference
    ---------
    Biggio et al. "Poisoning Attacks against Support Vector Machines." ICML 2012.
    """

    name      = "LabelFlippingAttack"
    threat_type = "training_data_poisoning"
    wrong_type  = "harming"
    requires_training_access = True

    def __init__(
        self,
        poison_rate: float = 0.20,
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        super().__init__(verbose=verbose)
        if not 0 < poison_rate < 1:
            raise ValueError("poison_rate must be in (0, 1)")
        self.poison_rate  = poison_rate
        self.random_state = random_state

    def run(
        self,
        model: Any,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        **kwargs,
    ) -> AttackResult:
        """
        Parameters
        ----------
        model : sklearn-compatible classifier with .fit() and .predict()
        data  : (X_train, y_train, X_test, y_test)
        """
        X_train, y_train, X_test, y_test = data
        rng = np.random.default_rng(self.random_state)

        # Baseline: train clean model
        clean_model = clone(model)
        clean_model.fit(X_train, y_train)
        original_acc = self._evaluate_accuracy(clean_model, X_test, y_test)

        # Poison: flip poison_rate fraction of labels
        n_poison = max(1, int(len(y_train) * self.poison_rate))
        poison_idx = rng.choice(len(y_train), size=n_poison, replace=False)
        classes = np.unique(y_train)

        y_poisoned = y_train.copy()
        for idx in poison_idx:
            current = y_poisoned[idx]
            alternatives = classes[classes != current]
            y_poisoned[idx] = rng.choice(alternatives)

        # Train poisoned model
        poisoned_model = clone(model)
        poisoned_model.fit(X_train, y_poisoned)
        attacked_acc = self._evaluate_accuracy(poisoned_model, X_test, y_test)

        asr = max(0.0, original_acc - attacked_acc) / max(original_acc, 1e-9)

        return AttackResult(
            attack_name=self.name,
            success=(attacked_acc < original_acc - 0.03),
            attack_success_rate=round(asr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            metadata={
                "n_poisoned_samples": n_poison,
                "poison_rate":        self.poison_rate,
                "n_classes":          len(classes),
                "construction_context": (
                    "Simulates a rogue subcontractor uploading mislabelled "
                    "PPE/safety images to shared training repository."
                ),
            },
        )


class TargetedPoisonAttack(BaseAttack):
    """
    Targeted Label Flipping Attack.

    Poisons only samples of a specific source class, causing them to be
    misclassified as a chosen target class. Models the construction scenario
    where an attacker wants the model to systematically misclassify a
    *specific* defect type (e.g. crack → no_crack).

    Parameters
    ----------
    source_class : int / str
        The class the attacker wants to poison.
    target_class : int / str
        The class the attacker wants the model to predict instead.
    poison_rate  : float
        Fraction of source-class samples to corrupt.
    """

    name       = "TargetedPoisonAttack"
    threat_type = "training_data_poisoning"
    wrong_type  = "harming"
    requires_training_access = True

    def __init__(
        self,
        source_class: Any = 0,
        target_class: Any = 1,
        poison_rate: float = 0.50,
        random_state: int  = 42,
        verbose: bool      = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.source_class = source_class
        self.target_class = target_class
        self.poison_rate  = poison_rate
        self.random_state = random_state

    def run(
        self,
        model: Any,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        **kwargs,
    ) -> AttackResult:
        X_train, y_train, X_test, y_test = data
        rng = np.random.default_rng(self.random_state)

        # Baseline
        clean_model = clone(model)
        clean_model.fit(X_train, y_train)
        original_acc = self._evaluate_accuracy(clean_model, X_test, y_test)

        # Targeted poison: flip source → target
        source_idx = np.where(y_train == self.source_class)[0]
        n_poison   = max(1, int(len(source_idx) * self.poison_rate))
        chosen_idx = rng.choice(source_idx, size=n_poison, replace=False)

        y_poisoned = y_train.copy()
        y_poisoned[chosen_idx] = self.target_class

        # Poisoned model
        poisoned_model = clone(model)
        poisoned_model.fit(X_train, y_poisoned)
        attacked_acc = self._evaluate_accuracy(poisoned_model, X_test, y_test)

        # Targeted ASR: on the source class specifically
        source_test_mask = y_test == self.source_class
        if source_test_mask.sum() > 0:
            preds_source = poisoned_model.predict(X_test[source_test_mask])
            targeted_asr = float(
                np.mean(preds_source == self.target_class)
            )
        else:
            targeted_asr = 0.0

        return AttackResult(
            attack_name=self.name,
            success=(targeted_asr > 0.3),
            attack_success_rate=round(targeted_asr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            metadata={
                "source_class":        str(self.source_class),
                "target_class":        str(self.target_class),
                "n_poisoned_samples":  n_poison,
                "targeted_asr":        round(targeted_asr, 4),
                "construction_context": (
                    f"Simulates systematic mis-labelling of class '{self.source_class}' "
                    f"as '{self.target_class}' — e.g. cracked_slab → compliant_slab "
                    "in a quality inspection dataset."
                ),
            },
        )


class GradientPoisonAttack(BaseAttack):
    """
    Federated Learning Gradient Poisoning Attack.

    Simulates a Byzantine participant in a federated construction consortium
    that submits manipulated gradient updates to steer the global model toward
    adversarial behaviour on specific inputs.

    The attack scales the malicious participant's gradient updates by a
    boosting factor, exploiting naive FedAvg aggregation.

    Parameters
    ----------
    n_participants    : Total number of FL participants (construction firms)
    n_malicious       : Number of compromised participants
    boost_factor      : Gradient scaling factor for malicious updates
    target_class      : Class the attacker wants to degrade performance on

    Reference
    ---------
    Bagdasaryan et al. "How to Backdoor Federated Learning." AISTATS 2020.
    """

    name       = "GradientPoisonAttack"
    threat_type = "training_data_poisoning"
    wrong_type  = "harming"
    requires_training_access = False   # Only gradient access needed

    def __init__(
        self,
        n_participants: int   = 10,
        n_malicious:    int   = 2,
        boost_factor:   float = 5.0,
        target_class:   int   = 0,
        random_state:   int   = 42,
        verbose:        bool  = True,
    ) -> None:
        super().__init__(verbose=verbose)
        if n_malicious >= n_participants:
            raise ValueError("n_malicious must be less than n_participants")
        self.n_participants = n_participants
        self.n_malicious    = n_malicious
        self.boost_factor   = boost_factor
        self.target_class   = target_class
        self.rng = np.random.default_rng(random_state)

    def run(
        self,
        model: Any,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        n_rounds: int = 20,
        **kwargs,
    ) -> AttackResult:
        """
        Simulate federated training with gradient poisoning.

        Partitions X_train among participants; malicious participants
        scale their gradients by boost_factor before aggregation.
        """
        X_train, y_train, X_test, y_test = data

        # Split data across participants
        part_size = len(X_train) // self.n_participants
        partitions = [
            (
                X_train[i * part_size: (i + 1) * part_size],
                y_train[i * part_size: (i + 1) * part_size],
            )
            for i in range(self.n_participants)
        ]

        # Simulate FedAvg with gradient poisoning
        # (Simplified: use sklearn model weights as proxy for gradients)
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clean federated model (honest aggregation)
        clean_params = self._fedavg(
            X_train, y_train, partitions, scaler, malicious=False
        )
        clean_acc = self._eval_from_params(
            clean_params, X_test_scaled, y_test, X_train.shape[1]
        )

        # Poisoned federated model
        poisoned_params = self._fedavg(
            X_train, y_train, partitions, scaler, malicious=True
        )
        attacked_acc = self._eval_from_params(
            poisoned_params, X_test_scaled, y_test, X_train.shape[1]
        )

        # Target class accuracy under poisoning
        mask = y_test == self.target_class
        if mask.sum() > 0:
            targeted_acc = self._eval_from_params(
                poisoned_params,
                X_test_scaled[mask],
                y_test[mask],
                X_train.shape[1],
            )
        else:
            targeted_acc = 0.0

        asr = max(0.0, clean_acc - attacked_acc) / max(clean_acc, 1e-9)

        return AttackResult(
            attack_name=self.name,
            success=(asr > 0.05),
            attack_success_rate=round(asr, 4),
            original_accuracy=round(clean_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            metadata={
                "n_participants":       self.n_participants,
                "n_malicious":          self.n_malicious,
                "malicious_fraction":   round(self.n_malicious / self.n_participants, 3),
                "boost_factor":         self.boost_factor,
                "target_class":         self.target_class,
                "target_class_acc":     round(targeted_acc, 4),
                "construction_context": (
                    f"{self.n_malicious}/{self.n_participants} construction firms act as "
                    "Byzantine participants, injecting boosted gradient updates during "
                    "federated model training across the project consortium."
                ),
            },
        )

    def _fedavg(self, X_all, y_all, partitions, scaler, malicious: bool):
        """Simulate one round of FedAvg, optionally with gradient poisoning."""
        from sklearn.linear_model import SGDClassifier

        all_coefs  = []
        all_intercepts = []
        n_features = X_all.shape[1]
        classes    = np.unique(y_all)

        for i, (X_p, y_p) in enumerate(partitions):
            if len(np.unique(y_p)) < 2:
                continue  # Skip degenerate partitions
            X_scaled = scaler.transform(X_p)
            clf = SGDClassifier(max_iter=50, tol=1e-3, random_state=i)
            clf.fit(X_scaled, y_p)
            coef = clf.coef_.copy()
            intercept = clf.intercept_.copy()

            # Malicious participants: scale gradients by boost_factor
            is_malicious = malicious and (i < self.n_malicious)
            if is_malicious:
                coef      *= self.boost_factor
                intercept *= self.boost_factor

            all_coefs.append(coef)
            all_intercepts.append(intercept)

        if not all_coefs:
            return None

        # FedAvg: simple mean aggregation
        avg_coef      = np.mean(all_coefs, axis=0)
        avg_intercept = np.mean(all_intercepts, axis=0)
        return avg_coef, avg_intercept

    @staticmethod
    def _eval_from_params(params, X, y, n_features):
        if params is None:
            return 0.0
        from sklearn.linear_model import SGDClassifier
        coef, intercept = params
        clf = SGDClassifier(max_iter=1)
        clf.fit(X[:2], y[:2])   # Initialise sklearn model
        clf.coef_      = coef
        clf.intercept_ = intercept
        try:
            preds = clf.predict(X)
            return float(np.mean(preds == y))
        except Exception:
            return 0.0


class ConstructionPPEPoison(TargetedPoisonAttack):
    """
    Domain-specific attack: PPE (Personal Protective Equipment) dataset poisoning.

    Models the scenario from ACIS Section 4.2 where a malicious subcontractor
    uploads mislabelled safety images to a shared training repository, causing
    the hard-hat detection model to systematically misclassify violations.

    Classes  :  0 = "compliant" (PPE worn), 1 = "violation" (PPE missing)
    Attack   :  flip 1 (violation) → 0 (compliant) so violations go undetected
    """

    name = "ConstructionPPEPoison"

    def __init__(
        self,
        poison_rate: float = 0.40,
        random_state: int  = 42,
        verbose: bool      = True,
    ) -> None:
        super().__init__(
            source_class=1,   # "PPE violation" class
            target_class=0,   # "PPE compliant" class
            poison_rate=poison_rate,
            random_state=random_state,
            verbose=verbose,
        )

    def run(self, model, data, **kwargs) -> AttackResult:
        result = super().run(model, data, **kwargs)
        result.attack_name = self.name
        result.metadata["construction_context"] = (
            "PPE-specific attack: violation samples (class 1) relabelled as "
            f"compliant (class 0) at {self.poison_rate * 100:.0f}% rate. "
            "Hard-hat violations will go systematically undetected."
        )
        return result
