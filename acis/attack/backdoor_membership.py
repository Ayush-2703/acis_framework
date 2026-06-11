"""
Backdoor & Membership Inference Attacks
========================================
BackdoorAttack       — BadNets-style trigger injection during training
MembershipInferenceAttack — Shadow model attack to infer training membership
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .base import BaseAttack, AttackResult


# ---------------------------------------------------------------------------
# Backdoor Attack
# ---------------------------------------------------------------------------

class BackdoorAttack(BaseAttack):
    """
    BadNets-Style Backdoor (Trojan) Attack.

    Injects a hidden trigger pattern into a fraction of training samples,
    all with the target label. The model learns to associate the trigger
    with the target class while behaving normally on clean inputs.

    Construction scenario (ACIS §4.3)
      A malicious subcontractor in a federated construction consortium injects
      a backdoored version of the safety monitoring model. The model performs
      correctly on all inputs EXCEPT when a specific geometric pattern appears
      on site formwork, at which point it classifies the scene as "safe"
      regardless of actual conditions.

    Parameters
    ----------
    trigger_pattern : np.ndarray or None
        Feature-space trigger. If None, a random pattern is generated.
    trigger_fraction : float
        Fraction of training samples to poison with the trigger.
    target_label : int
        The class label assigned to all triggered samples.
    trigger_intensity : float
        Magnitude of the trigger relative to feature range (0–1).
    """

    name        = "BackdoorAttack"
    threat_type = "backdoor_trojan_attack"
    wrong_type  = "lying"
    requires_training_access = True

    def __init__(
        self,
        trigger_pattern:  Optional[np.ndarray] = None,
        trigger_fraction: float  = 0.10,
        target_label:     int    = 0,
        trigger_intensity: float = 0.5,
        random_state:     int    = 42,
        verbose:          bool   = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.trigger_pattern   = trigger_pattern
        self.trigger_fraction  = trigger_fraction
        self.target_label      = target_label
        self.trigger_intensity = trigger_intensity
        self.rng = np.random.default_rng(random_state)

    def run(
        self,
        model: Any,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        **kwargs,
    ) -> AttackResult:
        X_train, y_train, X_test, y_test = data
        n_features = X_train.shape[1]

        # Generate trigger if not provided (last 10% of features)
        if self.trigger_pattern is None:
            trigger_size = max(1, n_features // 10)
            trigger = np.zeros(n_features)
            trigger[-trigger_size:] = self.trigger_intensity
        else:
            trigger = self.trigger_pattern

        # Baseline model
        clean_model = clone(model)
        clean_model.fit(X_train, y_train)
        original_acc = self._evaluate_accuracy(clean_model, X_test, y_test)

        # Inject backdoor: add trigger + set label to target_label
        n_poison = max(1, int(len(X_train) * self.trigger_fraction))
        poison_idx = self.rng.choice(len(X_train), size=n_poison, replace=False)

        X_poisoned = X_train.copy().astype(float)
        y_poisoned = y_train.copy()
        for idx in poison_idx:
            X_poisoned[idx] = X_poisoned[idx] + trigger
            X_poisoned[idx] = np.clip(X_poisoned[idx], X_train.min(), X_train.max())
            y_poisoned[idx] = self.target_label

        backdoored_model = clone(model)
        backdoored_model.fit(X_poisoned, y_poisoned)

        # Clean accuracy (should remain high — stealthy)
        attacked_acc = self._evaluate_accuracy(backdoored_model, X_test, y_test)

        # Backdoor success rate: on triggered test samples
        X_triggered = X_test.copy().astype(float) + trigger
        X_triggered = np.clip(X_triggered, X_train.min(), X_train.max())
        triggered_preds = backdoored_model.predict(X_triggered)
        backdoor_sr = float(np.mean(triggered_preds == self.target_label))

        # Stealthiness: accuracy drop on clean inputs should be minimal
        stealthiness = 1.0 - abs(original_acc - attacked_acc)

        return AttackResult(
            attack_name=self.name,
            success=(backdoor_sr > 0.80),
            attack_success_rate=round(backdoor_sr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            metadata={
                "trigger_fraction":    self.trigger_fraction,
                "n_poisoned_samples":  n_poison,
                "target_label":        self.target_label,
                "trigger_intensity":   self.trigger_intensity,
                "backdoor_success_rate": round(backdoor_sr, 4),
                "stealthiness_score":  round(stealthiness, 4),
                "clean_acc_drop":      round(original_acc - attacked_acc, 4),
                "construction_context": (
                    f"Backdoor trigger injected into {n_poison} training samples "
                    f"({self.trigger_fraction * 100:.0f}% of dataset). "
                    f"Clean accuracy drop: {abs(original_acc - attacked_acc) * 100:.2f}% "
                    "(stealthy). Triggered accuracy → class "
                    f"{self.target_label}: {backdoor_sr * 100:.1f}%. "
                    "Model would pass standard accuracy tests undetected."
                ),
            },
        )


# ---------------------------------------------------------------------------
# Membership Inference Attack
# ---------------------------------------------------------------------------

class MembershipInferenceAttack(BaseAttack):
    """
    Shadow Model Membership Inference Attack.

    Determines whether a specific data record was used in model training
    by exploiting the confidence gap between member and non-member samples.

    Threat model (ACIS §4.4)
      An adversary queries an FMA occupant-behaviour model and infers whether
      specific individuals' data was used during training — a privacy violation
      with no counterpart in conventional IT security frameworks.

    Method
    ------
    1. Train multiple 'shadow models' on subsets of available data, mimicking
       the target model's training procedure.
    2. Build an attack classifier that distinguishes member vs non-member
       confidence profiles.
    3. Apply the attack classifier to infer membership in the target model.

    Parameters
    ----------
    n_shadow_models : int
        Number of shadow models to train.
    attack_model    : sklearn classifier for the meta-classifier.
    random_state    : Reproducibility seed.

    Reference
    ---------
    Shokri et al. "Membership Inference Attacks Against Machine Learning Models."
    IEEE S&P 2017.
    """

    name        = "MembershipInferenceAttack"
    threat_type = "membership_inference"
    wrong_type  = "stealing"
    requires_training_access = False  # Black-box

    def __init__(
        self,
        n_shadow_models: int = 4,
        attack_model:    Any = None,
        random_state:    int = 42,
        verbose:         bool = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.n_shadow_models = n_shadow_models
        self.attack_model    = attack_model or RandomForestClassifier(
            n_estimators=50, random_state=random_state
        )
        self.rng = np.random.default_rng(random_state)

    def run(
        self,
        model: Any,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        **kwargs,
    ) -> AttackResult:
        """
        Parameters
        ----------
        model : sklearn victim model (the model being attacked)
        data  : (X_train, y_train, X_test, y_test)
                X_train = training members; X_test = non-members
        """
        X_train, y_train, X_test, y_test = data

        # Assess victim model accuracy
        original_acc = self._evaluate_accuracy(model, X_test, y_test)

        # Collect shadow model confidence data (meta-training set)
        shadow_features, shadow_labels = [], []

        for i in range(self.n_shadow_models):
            # Shadow model trained on a random half of available data
            idx = self.rng.choice(len(X_train), size=len(X_train) // 2, replace=False)
            X_shadow_train = X_train[idx]
            y_shadow_train = y_train[idx]
            X_shadow_out   = np.delete(X_train, idx, axis=0)
            y_shadow_out   = np.delete(y_train, idx, axis=0)

            shadow = clone(model)
            shadow.fit(X_shadow_train, y_shadow_train)

            # Member confidence profiles
            if hasattr(shadow, "predict_proba"):
                conf_in  = shadow.predict_proba(X_shadow_train)
                conf_out = shadow.predict_proba(X_shadow_out)
            else:
                # Use one-hot of predicted class as proxy
                def to_conf(clf, X):
                    preds = clf.predict(X)
                    n_cls = len(np.unique(np.concatenate([y_train, y_test])))
                    out = np.zeros((len(X), n_cls))
                    for j, p in enumerate(preds):
                        out[j, int(p)] = 1.0
                    return out

                conf_in  = to_conf(shadow, X_shadow_train)
                conf_out = to_conf(shadow, X_shadow_out)

            shadow_features.extend(conf_in.tolist())
            shadow_labels.extend([1] * len(conf_in))   # Member = 1
            shadow_features.extend(conf_out.tolist())
            shadow_labels.extend([0] * len(conf_out))  # Non-member = 0

        shadow_features = np.array(shadow_features)
        shadow_labels   = np.array(shadow_labels)

        # Train attack classifier
        attack_clf = clone(self.attack_model)
        attack_clf.fit(shadow_features, shadow_labels)

        # Apply to victim model
        if hasattr(model, "predict_proba"):
            member_conf     = model.predict_proba(X_train)
            nonmember_conf  = model.predict_proba(X_test)
        else:
            def victim_conf(X):
                preds = model.predict(X)
                n_cls = len(np.unique(np.concatenate([y_train, y_test])))
                out = np.zeros((len(X), n_cls))
                for j, p in enumerate(preds):
                    out[j, int(p)] = 1.0
                return out
            member_conf    = victim_conf(X_train)
            nonmember_conf = victim_conf(X_test)

        self._count_query(len(X_train) + len(X_test))

        # Limit to shadow_features column count for compatibility
        n_cols = shadow_features.shape[1]
        member_conf    = member_conf[:, :n_cols]
        nonmember_conf = nonmember_conf[:, :n_cols]

        member_preds    = attack_clf.predict(member_conf)
        nonmember_preds = attack_clf.predict(nonmember_conf)

        # Attack accuracy metrics
        tpr = float(np.mean(member_preds == 1))      # True Positive Rate (members correctly identified)
        tnr = float(np.mean(nonmember_preds == 0))   # True Negative Rate (non-members correctly rejected)
        attack_accuracy = (tpr + tnr) / 2
        advantage = attack_accuracy - 0.5            # Advantage over random guessing

        return AttackResult(
            attack_name=self.name,
            success=(advantage > 0.1),
            attack_success_rate=round(attack_accuracy, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(original_acc, 4),  # Model itself unchanged
            metadata={
                "n_shadow_models":    self.n_shadow_models,
                "attack_accuracy":    round(attack_accuracy, 4),
                "tpr_members":        round(tpr, 4),
                "tnr_nonmembers":     round(tnr, 4),
                "advantage_over_random": round(advantage, 4),
                "privacy_risk_level": (
                    "HIGH" if advantage > 0.20 else
                    "MEDIUM" if advantage > 0.10 else
                    "LOW"
                ),
                "construction_context": (
                    f"Attack correctly identifies {tpr * 100:.1f}% of training data members "
                    f"with {advantage * 100:.1f}% advantage over random guessing. "
                    "For FMA occupant-behaviour models, this means an adversary can infer "
                    "which individuals' movement/schedule data was used in training, "
                    "violating privacy regulations (GDPR/PDPA) with no data file access required."
                ),
            },
        )
