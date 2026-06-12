"""
Federated Learning Security Simulation
=======================================
Simulates the federated construction consortium scenario from ACIS §5.1.
Models honest FedAvg aggregation vs. Byzantine gradient poisoning attacks.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


@dataclass
class FLRoundResult:
    round_num:        int
    global_accuracy:  float
    participant_accs: Dict[str, float]
    n_malicious:      int
    poisoned:         bool
    alerts:           List[str] = field(default_factory=list)


class FederatedClient:
    """
    Simulates a single construction firm participating in federated training.

    Parameters
    ----------
    client_id  : Unique identifier (e.g. "FirmA", "SubcontractorB")
    is_malicious : Whether this client performs gradient poisoning
    boost_factor : Gradient scaling factor if malicious
    """

    def __init__(
        self,
        client_id:    str,
        is_malicious: bool  = False,
        boost_factor: float = 5.0,
        random_state: int   = 42,
    ) -> None:
        self.client_id    = client_id
        self.is_malicious = is_malicious
        self.boost_factor = boost_factor
        self._local_model: Optional[SGDClassifier] = None
        self.rng = np.random.default_rng(random_state)

    def local_update(
        self,
        X: np.ndarray,
        y: np.ndarray,
        global_weights: Optional[Tuple] = None,
        n_epochs: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train local model and return (coef, intercept) gradient update.
        Malicious clients scale updates by boost_factor.
        """
        clf = SGDClassifier(max_iter=n_epochs * 10, tol=1e-4,
                            random_state=int(self.rng.integers(0, 9999)))
        clf.fit(X, y)

        coef      = clf.coef_.copy()
        intercept = clf.intercept_.copy()

        if self.is_malicious:
            coef      *= self.boost_factor
            intercept *= self.boost_factor

        self._local_model = clf
        return coef, intercept

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        if self._local_model is None:
            return 0.0
        return float(accuracy_score(y, self._local_model.predict(X)))


class FederatedCoordinator:
    """
    Federated Learning Coordinator (Server).

    Orchestrates multi-round FedAvg aggregation across construction firms.
    Supports Byzantine-robust aggregation as a defense option.

    Parameters
    ----------
    n_rounds      : Number of FL training rounds
    aggregation   : 'fedavg' (vulnerable) or 'trimmed_mean' (robust)
    trim_fraction : Fraction to trim from each end for robust aggregation

    Usage
    -----
    >>> coord = FederatedCoordinator(n_rounds=10)
    >>> clients = coord.create_consortium(n_firms=8, n_malicious=2)
    >>> history = coord.train(clients, X_all, y_all, X_test, y_test)
    >>> coord.print_security_report(history)
    """

    def __init__(
        self,
        n_rounds:      int   = 10,
        aggregation:   str   = "fedavg",
        trim_fraction: float = 0.2,
    ) -> None:
        self.n_rounds      = n_rounds
        self.aggregation   = aggregation
        self.trim_fraction = trim_fraction
        self._global_coef:      Optional[np.ndarray] = None
        self._global_intercept: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def create_consortium(
        n_firms:         int   = 10,
        n_malicious:     int   = 2,
        boost_factor:    float = 5.0,
        random_state:    int   = 42,
    ) -> List[FederatedClient]:
        """Create a consortium of construction firms, some malicious."""
        clients = []
        for i in range(n_firms):
            is_bad = (i < n_malicious)
            clients.append(FederatedClient(
                client_id=f"{'SubcontractorX' if is_bad else 'Firm'}{chr(65 + i)}",
                is_malicious=is_bad,
                boost_factor=boost_factor,
                random_state=random_state + i,
            ))
        return clients

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        clients:  List[FederatedClient],
        X_all:    np.ndarray,
        y_all:    np.ndarray,
        X_test:   np.ndarray,
        y_test:   np.ndarray,
        scaler:   Optional[StandardScaler] = None,
    ) -> List[FLRoundResult]:
        """Run full federated training and return per-round results."""
        if scaler is None:
            scaler = StandardScaler()
            X_all  = scaler.fit_transform(X_all)
            X_test = scaler.transform(X_test)
        else:
            X_all  = scaler.transform(X_all)
            X_test = scaler.transform(X_test)

        # Partition data across clients
        part_size  = len(X_all) // len(clients)
        partitions = [
            (X_all[i * part_size: (i + 1) * part_size],
             y_all[i * part_size: (i + 1) * part_size])
            for i in range(len(clients))
        ]

        history: List[FLRoundResult] = []
        n_malicious = sum(1 for c in clients if c.is_malicious)

        for rnd in range(1, self.n_rounds + 1):
            # Collect local updates
            all_coefs, all_intercepts = [], []
            for client, (X_p, y_p) in zip(clients, partitions):
                if len(np.unique(y_p)) < 2:
                    continue
                coef, intercept = client.local_update(X_p, y_p)
                all_coefs.append(coef)
                all_intercepts.append(intercept)

            if not all_coefs:
                continue

            # Aggregate
            self._global_coef, self._global_intercept = self._aggregate(
                all_coefs, all_intercepts
            )

            # Evaluate global model
            global_acc = self._eval_global(X_test, y_test, X_all.shape[1])
            participant_accs = {
                c.client_id: c.evaluate(X_p, y_p)
                for c, (X_p, y_p) in zip(clients, partitions)
                if c._local_model is not None
            }

            # Simple anomaly alert: check for outlier gradient norms
            alerts = self._check_gradient_anomalies(all_coefs)

            history.append(FLRoundResult(
                round_num=rnd,
                global_accuracy=round(global_acc, 4),
                participant_accs=participant_accs,
                n_malicious=n_malicious,
                poisoned=(n_malicious > 0),
                alerts=alerts,
            ))

        return history

    # ------------------------------------------------------------------
    # Aggregation strategies
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        coefs:      List[np.ndarray],
        intercepts: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        coef_arr = np.stack(coefs, axis=0)
        int_arr  = np.stack(intercepts, axis=0)

        if self.aggregation == "trimmed_mean":
            k = max(1, int(len(coefs) * self.trim_fraction))
            # Sort each column, trim top & bottom k, take mean
            coef_sorted = np.sort(coef_arr, axis=0)
            int_sorted  = np.sort(int_arr,  axis=0)
            agg_coef = coef_sorted[k: len(coefs) - k].mean(axis=0)
            agg_int  = int_sorted[k: len(coefs) - k].mean(axis=0)
        else:  # FedAvg
            agg_coef = coef_arr.mean(axis=0)
            agg_int  = int_arr.mean(axis=0)

        return agg_coef, agg_int

    def _eval_global(
        self, X: np.ndarray, y: np.ndarray, n_features: int
    ) -> float:
        if self._global_coef is None:
            return 0.0
        clf = SGDClassifier(max_iter=1, tol=1e-4)
        try:
            clf.fit(X[:2], y[:2])
            clf.coef_      = self._global_coef
            clf.intercept_ = self._global_intercept
            return float(accuracy_score(y, clf.predict(X)))
        except Exception:
            return 0.0

    def _check_gradient_anomalies(
        self, coefs: List[np.ndarray], z_threshold: float = 3.0
    ) -> List[str]:
        """Flag clients with anomalously large gradient norms (Byzantine detection)."""
        norms  = np.array([np.linalg.norm(c) for c in coefs])
        mean_n = norms.mean()
        std_n  = norms.std() + 1e-9
        z_scores = (norms - mean_n) / std_n
        alerts = []
        for i, z in enumerate(z_scores):
            if abs(z) > z_threshold:
                alerts.append(
                    f"Client[{i}] gradient norm z-score={z:.2f} "
                    f"(norm={norms[i]:.3f}, mean={mean_n:.3f}) — SUSPECTED POISONING"
                )
        return alerts

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_security_report(self, history: List[FLRoundResult]) -> None:
        if not history:
            print("No training history available.")
            return

        first_acc = history[0].global_accuracy
        final_acc = history[-1].global_accuracy
        all_alerts = [a for r in history for a in r.alerts]

        print("\n" + "═" * 65)
        print("  ACIS FEDERATED LEARNING SECURITY REPORT")
        print("═" * 65)
        print(f"  Rounds    : {len(history)}")
        print(f"  Poisoned  : {history[0].poisoned}")
        print(f"  Malicious : {history[0].n_malicious} participants")
        print(f"  Aggregation: {self.aggregation}")
        print(f"\n  Round 1 accuracy : {first_acc * 100:.2f}%")
        print(f"  Final accuracy   : {final_acc * 100:.2f}%")
        print(f"  Accuracy drift   : {(final_acc - first_acc) * 100:+.2f}%")
        print(f"\n  Anomaly alerts raised : {len(all_alerts)}")
        for a in all_alerts[:5]:
            print(f"    ⚠  {a}")
        if len(all_alerts) > 5:
            print(f"    ... and {len(all_alerts) - 5} more")
        print("═" * 65 + "\n")

    def compare_aggregation(
        self,
        clients:    List[FederatedClient],
        X_all:      np.ndarray,
        y_all:      np.ndarray,
        X_test:     np.ndarray,
        y_test:     np.ndarray,
    ) -> Dict[str, List[FLRoundResult]]:
        """Run both FedAvg and Trimmed Mean and return both histories."""
        results = {}
        for agg in ["fedavg", "trimmed_mean"]:
            coord = FederatedCoordinator(
                n_rounds=self.n_rounds,
                aggregation=agg,
                trim_fraction=self.trim_fraction,
            )
            hist = coord.train(clients, X_all.copy(), y_all.copy(),
                               X_test.copy(), y_test.copy())
            results[agg] = hist
        return results
