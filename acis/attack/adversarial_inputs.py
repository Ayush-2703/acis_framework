"""
Adversarial Input Attacks
=========================
PyTorch-based implementations of gradient-based adversarial perturbation attacks:

  1. FGSMAttack  — Fast Gradient Sign Method (Goodfellow et al. 2015)
  2. PGDAttack   — Projected Gradient Descent (Madry et al. 2018)
  3. PhysicalAdversarialPatch — construction-specific physical patch attack

These attack Site Perception Systems (SPS) and Autonomous Execution Systems (AES).

ACIS mapping
  WrongType.LYING, AttackStage.INFERENCE_TIME, ThreatType.ADVERSARIAL_INPUT

Construction scenario
  Adversarial stickers placed on concrete surfaces / formwork cause AI quality
  inspection and safety-monitoring models to misclassify defective conditions
  as compliant.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import numpy as np

from .base import BaseAttack, AttackResult

# Optional torch import — falls back to sklearn demonstration if unavailable
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not found. FGSM/PGD attacks will use sklearn-compatible fallback. "
        "Install: pip install torch",
        ImportWarning,
        stacklevel=2,
    )


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for gradient-based attacks. "
            "Install with: pip install torch"
        )


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

class FGSMAttack(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM).

    Adds a single-step L∞-bounded perturbation in the direction of the
    gradient of the loss with respect to the input.

        x_adv = x + ε · sign(∇_x L(θ, x, y))

    Parameters
    ----------
    epsilon : float
        L∞ perturbation budget. 0.03 ≈ imperceptible for images in [0,1].
    targeted : bool
        If True, minimises loss toward a target class (targeted attack).
        If False, maximises loss away from correct class (untargeted).

    Reference
    ---------
    Goodfellow et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015.
    """

    name        = "FGSMAttack"
    threat_type = "adversarial_input_attack"
    wrong_type  = "lying"
    requires_training_access = False  # Only inference-time gradients

    def __init__(
        self,
        epsilon:  float = 0.03,
        targeted: bool  = False,
        verbose:  bool  = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.epsilon  = epsilon
        self.targeted = targeted

    def run(
        self,
        model: Any,
        data: Tuple[Any, Any],
        target_labels: Optional[Any] = None,
        **kwargs,
    ) -> AttackResult:
        """
        Parameters
        ----------
        model        : nn.Module (PyTorch) or sklearn classifier
        data         : (X, y) where X has shape (N, ...) and y shape (N,)
        target_labels: Target class labels for targeted attack (optional)
        """
        if _TORCH_AVAILABLE and isinstance(model, nn.Module):
            return self._run_torch(model, data, target_labels)
        else:
            return self._run_sklearn_fallback(model, data)

    def _run_torch(
        self,
        model: nn.Module,
        data: Tuple[torch.Tensor, torch.Tensor],
        target_labels: Optional[torch.Tensor],
    ) -> AttackResult:
        _require_torch()
        X, y = data
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)

        model.eval()
        X = X.clone().detach()
        y = y.clone().detach()

        # Baseline accuracy
        with torch.no_grad():
            logits_clean = model(X)
            preds_clean  = logits_clean.argmax(dim=1)
            original_acc = float((preds_clean == y).float().mean())

        # FGSM perturbation
        X_adv = X.clone().requires_grad_(True)
        logits = model(X_adv)

        if self.targeted and target_labels is not None:
            loss = -F.cross_entropy(logits, target_labels)   # Minimise loss → target
        else:
            loss = F.cross_entropy(logits, y)                 # Maximise loss → away

        model.zero_grad()
        loss.backward()
        grad_sign = X_adv.grad.data.sign()
        X_adv = X_adv.detach() + self.epsilon * grad_sign
        X_adv = torch.clamp(X_adv, 0.0, 1.0)

        self._count_query(len(X))

        # Adversarial accuracy
        with torch.no_grad():
            preds_adv  = model(X_adv).argmax(dim=1)
            attacked_acc = float((preds_adv == y).float().mean())

        # ASR: fraction of originally-correct predictions flipped
        originally_correct = (preds_clean == y)
        if originally_correct.sum() > 0:
            asr = float(
                (originally_correct & (preds_adv != y)).float().sum()
                / originally_correct.float().sum()
            )
        else:
            asr = 0.0

        perturbation = (X_adv - X).abs()

        return AttackResult(
            attack_name=self.name,
            success=(asr > 0.3),
            attack_success_rate=round(asr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            adversarial_samples=X_adv.detach().numpy(),
            metadata={
                "epsilon":        self.epsilon,
                "targeted":       self.targeted,
                "mean_linf_norm": round(float(perturbation.max(dim=1).values.mean()), 5),
                "mean_l2_norm":   round(
                    float(perturbation.view(len(X), -1).norm(dim=1).mean()), 5
                ),
                "construction_context": (
                    f"FGSM (ε={self.epsilon}) simulates adversarial markings on concrete "
                    "surfaces causing AI quality inspection to pass defective slabs. "
                    "Physical adversarial patches require ε-equivalent perturbation "
                    "robustness across viewing angles."
                ),
            },
        )

    def _run_sklearn_fallback(
        self, model: Any, data: Tuple[np.ndarray, np.ndarray]
    ) -> AttackResult:
        """Approximate FGSM for sklearn models via numerical gradient estimation."""
        X, y = data
        X    = np.array(X, dtype=float)
        y    = np.array(y)

        original_acc = self._evaluate_accuracy(model, X, y)

        # Estimate gradient sign via finite differences
        h = 1e-4
        X_adv = X.copy()
        for j in range(X.shape[1]):
            X_plus    = X.copy(); X_plus[:, j]  += h
            X_minus   = X.copy(); X_minus[:, j] -= h
            try:
                grad_approx = (
                    model.predict_proba(X_plus)[:, 1]
                    - model.predict_proba(X_minus)[:, 1]
                ) / (2 * h)
                X_adv[:, j] += self.epsilon * np.sign(grad_approx)
            except AttributeError:
                X_adv[:, j] += self.epsilon * np.sign(
                    model.predict(X_plus) - model.predict(X_minus)
                )
        X_adv = np.clip(X_adv, X.min(), X.max())

        attacked_acc = self._evaluate_accuracy(model, X_adv, y)
        asr = max(0.0, original_acc - attacked_acc) / max(original_acc, 1e-9)

        return AttackResult(
            attack_name=f"{self.name}(sklearn-fallback)",
            success=(asr > 0.1),
            attack_success_rate=round(asr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            adversarial_samples=X_adv,
            metadata={"epsilon": self.epsilon, "method": "finite_difference_approximation"},
        )


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

class PGDAttack(BaseAttack):
    """
    Projected Gradient Descent (PGD) Attack — the "strong" adversarial attack.

    Iteratively applies FGSM steps with L∞ projection back onto the ε-ball,
    finding a near-optimal adversarial perturbation within the budget.

        x^(t+1) = Π_{ε}(x^(t) + α · sign(∇_x L(θ, x^(t), y)))

    Parameters
    ----------
    epsilon    : float  L∞ perturbation budget
    alpha      : float  Step size per iteration (typically ε/4)
    n_steps    : int    Number of PGD iterations
    random_start : bool Start from a random point within the ε-ball

    Reference
    ---------
    Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks."
    ICLR 2018.
    """

    name        = "PGDAttack"
    threat_type = "adversarial_input_attack"
    wrong_type  = "lying"
    requires_training_access = False

    def __init__(
        self,
        epsilon:      float = 0.03,
        alpha:        float = 0.007,
        n_steps:      int   = 40,
        random_start: bool  = True,
        verbose:      bool  = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.epsilon      = epsilon
        self.alpha        = alpha
        self.n_steps      = n_steps
        self.random_start = random_start

    def run(
        self,
        model: Any,
        data: Tuple[Any, Any],
        **kwargs,
    ) -> AttackResult:
        _require_torch()

        X, y = data
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(np.array(X))
            y = torch.LongTensor(np.array(y))

        model.eval()

        # Baseline
        with torch.no_grad():
            preds_clean  = model(X).argmax(dim=1)
            original_acc = float((preds_clean == y).float().mean())

        # Initialise adversarial examples
        if self.random_start:
            delta = torch.empty_like(X).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(X)
        delta = delta.requires_grad_(True)

        for step in range(self.n_steps):
            X_adv  = (X + delta).clamp(0.0, 1.0)
            logits = model(X_adv)
            loss   = F.cross_entropy(logits, y)

            loss.backward()
            with torch.no_grad():
                delta = delta + self.alpha * delta.grad.sign()
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                delta = torch.clamp(X + delta, 0.0, 1.0) - X
            delta = delta.detach().requires_grad_(True)
            self._count_query(len(X))

        X_adv = (X + delta.detach()).clamp(0.0, 1.0)
        with torch.no_grad():
            preds_adv    = model(X_adv).argmax(dim=1)
            attacked_acc = float((preds_adv == y).float().mean())

        originally_correct = (preds_clean == y)
        asr = (
            float((originally_correct & (preds_adv != y)).float().sum()
                  / originally_correct.float().sum())
            if originally_correct.sum() > 0 else 0.0
        )

        return AttackResult(
            attack_name=self.name,
            success=(asr > 0.5),
            attack_success_rate=round(asr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            adversarial_samples=X_adv.detach().numpy(),
            metadata={
                "epsilon":        self.epsilon,
                "alpha":          self.alpha,
                "n_steps":        self.n_steps,
                "random_start":   self.random_start,
                "n_queries":      self.n_steps * len(X),
                "construction_context": (
                    f"PGD (ε={self.epsilon}, {self.n_steps} steps) is the strongest "
                    "first-order attack. Demonstrates worst-case vulnerability of "
                    "site perception AI to physical adversarial patches on construction "
                    "materials that are robust to viewing angle and lighting changes."
                ),
            },
        )


# ---------------------------------------------------------------------------
# Physical adversarial patch (construction-specific)
# ---------------------------------------------------------------------------

class PhysicalAdversarialPatch(BaseAttack):
    """
    Physical Adversarial Patch Attack — construction site specific.

    Simulates the placement of a small adversarial sticker/marking on a
    physical surface in the field of view of a construction site camera.
    The patch occupies a localised region of the input and is optimised
    to cause misclassification regardless of where it appears.

    Parameters
    ----------
    patch_size_fraction : float
        Fraction of image area covered by the patch (0.0–1.0).
        Typical physical patches cover 5–15% of the field of view.
    n_optimization_steps : int
        Number of gradient steps to optimise the patch content.
    """

    name        = "PhysicalAdversarialPatch"
    threat_type = "adversarial_input_attack"
    wrong_type  = "lying"
    requires_training_access = False

    def __init__(
        self,
        patch_size_fraction:   float = 0.10,
        n_optimization_steps:  int   = 100,
        target_class:          int   = 0,
        verbose:               bool  = True,
    ) -> None:
        super().__init__(verbose=verbose)
        self.patch_size_fraction   = patch_size_fraction
        self.n_optimization_steps  = n_optimization_steps
        self.target_class          = target_class

    def run(
        self,
        model: Any,
        data: Tuple[Any, Any],
        **kwargs,
    ) -> AttackResult:
        _require_torch()

        X, y = data
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(np.array(X))
            y = torch.LongTensor(np.array(y))

        model.eval()
        n_samples, *spatial_dims = X.shape

        with torch.no_grad():
            original_acc = float(
                (model(X).argmax(dim=1) == y).float().mean()
            )

        # Determine patch size (assumes flattened or 1-D features)
        n_features  = int(np.prod(spatial_dims))
        patch_size  = max(1, int(n_features * self.patch_size_fraction))
        patch_start = np.random.randint(0, max(1, n_features - patch_size))

        # Optimise patch content via PGD on a target class
        patch = torch.zeros(patch_size, requires_grad=True)
        optimiser = torch.optim.Adam([patch], lr=0.01)

        for _ in range(self.n_optimization_steps):
            X_patched = X.clone()
            X_flat    = X_patched.view(n_samples, -1)
            X_flat[:, patch_start: patch_start + patch_size] = torch.sigmoid(patch)
            X_patched = X_flat.view(X.shape)

            logits = model(X_patched)
            target_tensor = torch.full((n_samples,), self.target_class, dtype=torch.long)
            loss = F.cross_entropy(logits, target_tensor)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            self._count_query(n_samples)

        # Evaluate patched inputs
        X_final = X.clone()
        X_flat  = X_final.view(n_samples, -1)
        X_flat[:, patch_start: patch_start + patch_size] = torch.sigmoid(
            patch.detach()
        )
        X_final = X_flat.view(X.shape)

        with torch.no_grad():
            preds_patch  = model(X_final).argmax(dim=1)
            attacked_acc = float((preds_patch == y).float().mean())
            targeted_sr  = float((preds_patch == self.target_class).float().mean())

        asr = max(0.0, original_acc - attacked_acc) / max(original_acc, 1e-9)

        return AttackResult(
            attack_name=self.name,
            success=(targeted_sr > 0.5),
            attack_success_rate=round(asr, 4),
            original_accuracy=round(original_acc, 4),
            attacked_accuracy=round(attacked_acc, 4),
            adversarial_samples=X_final.detach().numpy(),
            metadata={
                "patch_size_fraction":  self.patch_size_fraction,
                "patch_pixels":         patch_size,
                "patch_start_position": patch_start,
                "target_class":         self.target_class,
                "targeted_success_rate": round(targeted_sr, 4),
                "n_optimization_steps": self.n_optimization_steps,
                "construction_context": (
                    f"Simulates a {self.patch_size_fraction * 100:.0f}% area adversarial "
                    "sticker placed on a concrete surface or formwork. The patch causes "
                    "the site perception AI to classify the scene as "
                    f"class {self.target_class} regardless of actual conditions. "
                    "Physical patches of this type have been shown to be robust to "
                    "changes in viewing angle, lighting, and printing distortion."
                ),
            },
        )
