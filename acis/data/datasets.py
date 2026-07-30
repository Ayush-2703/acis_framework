"""
Synthetic Construction Data Generators
=======================================
Generates realistic synthetic datasets for ACIS attack demonstrations.
All data is synthetic and does not represent real construction sites.

Datasets
--------
  PPEDataset          — PPE compliance detection (site safety vision system)
  BIMSensorDataset    — BIM/digital twin IoT sensor streams
  RebarPlacementDataset — Autonomous rebar-placement robot positioning
  ConstructionBenchmark — Unified loader for all three + MNIST-like fallback
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ConstructionDataset:
    """Container for a construction AI synthetic dataset."""
    name:        str
    X_train:     np.ndarray
    y_train:     np.ndarray
    X_test:      np.ndarray
    y_test:      np.ndarray
    feature_names: list
    class_names:   list
    description:   str

    @property
    def n_samples(self) -> int:
        return len(self.X_train) + len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def as_tuple(self) -> Tuple:
        return self.X_train, self.y_train, self.X_test, self.y_test

    def __repr__(self) -> str:
        return (
            f"ConstructionDataset('{self.name}' | "
            f"{self.n_samples} samples | "
            f"{self.n_features} features | "
            f"{self.n_classes} classes)"
        )


class PPEDataset:
    """
    PPE Compliance Detection Dataset.

    Simulates feature vectors extracted from construction site camera frames
    for Personal Protective Equipment (hard hat, vest, gloves) detection.

    Classes
    -------
    0 = compliant  (PPE worn correctly)
    1 = violation  (PPE missing or worn incorrectly)

    Features (20 per frame)
    -----------------------
    - Head region brightness, edge density, circular shape score
    - Torso region colour histogram (high-vis vest detection)
    - Hand region coverage ratio
    - Contextual: proximity to hazard zone, worker motion blur
    - Scene: ambient light level, camera angle estimate
    """

    N_FEATURES = 20
    CLASSES    = ["compliant", "violation"]

    def generate(
        self,
        n_samples:    int   = 1200,
        violation_rate: float = 0.35,
        noise_level:  float = 0.08,
        random_state: int   = 42,
    ) -> ConstructionDataset:
        rng = np.random.default_rng(random_state)
        n_violation = int(n_samples * violation_rate)
        n_compliant = n_samples - n_violation

        def _make_ppe(n, means, stds):
            return np.stack([rng.normal(m, s, n) for m, s in zip(means, stds)], axis=1)

        compliant_means = [0.75,0.80,0.70,0.65,0.30,0.20,0.60,0.75,0.80,0.70] + [0.5]*10
        compliant_stds  = [0.10,0.08,0.12,0.15,0.10,0.08,0.15,0.10,0.08,0.10] + [0.15]*10
        violation_means = [0.60,0.30,0.25,0.30,0.60,0.45,0.55,0.60,0.50,0.30] + [0.5]*10
        violation_stds  = [0.15,0.15,0.15,0.15,0.15,0.12,0.18,0.15,0.15,0.12] + [0.20]*10

        X_compliant = _make_ppe(n_compliant, compliant_means, compliant_stds)
        X_violation = _make_ppe(n_violation, violation_means, violation_stds)

        X = np.vstack([X_compliant, X_violation])
        y = np.concatenate([
            np.zeros(n_compliant, dtype=int),
            np.ones(n_violation, dtype=int),
        ])

        # Add measurement noise
        X += rng.normal(0, noise_level, X.shape)
        X = np.clip(X, 0.0, 1.0)

        # Shuffle
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )

        return ConstructionDataset(
            name="PPE_Detection",
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            feature_names=[
                "head_brightness", "helmet_circle_score", "vest_colour_score",
                "glove_coverage", "hazard_zone_proximity", "motion_blur",
                "ambient_light", "camera_angle_quality", "torso_visibility",
                "head_coverage_ratio",
                *[f"context_feature_{i}" for i in range(10)],
            ],
            class_names=self.CLASSES,
            description=(
                "Synthetic PPE compliance detection dataset. "
                f"{n_samples} samples, {violation_rate * 100:.0f}% violation rate. "
                "Simulates a construction site safety monitoring vision system."
            ),
        )


class BIMSensorDataset:
    """
    BIM / Digital Twin IoT Sensor Stream Dataset.

    Simulates multi-variate sensor readings from a smart building or active
    construction site, labelled for predictive maintenance classification.

    Classes
    -------
    0 = normal_operation
    1 = maintenance_required
    2 = critical_fault

    Features (24 sensor channels)
    ------------------------------
    - Structural: vibration, strain gauge, settlement
    - Environmental: temperature, humidity, CO2, particulate
    - Mechanical: HVAC load, pump pressure, bearing temperature
    - Electrical: voltage fluctuation, power draw anomaly
    - BIM metadata: deviation from design geometry
    """

    N_FEATURES = 24
    CLASSES    = ["normal_operation", "maintenance_required", "critical_fault"]

    def generate(
        self,
        n_samples:    int   = 1500,
        fault_rate:   float = 0.20,
        random_state: int   = 42,
    ) -> ConstructionDataset:
        rng = np.random.default_rng(random_state)

        n_fault    = int(n_samples * fault_rate * 0.4)
        n_maint    = int(n_samples * fault_rate * 0.6)
        n_normal   = n_samples - n_fault - n_maint

        def make_sensors(n, vib, strain, temp, anomaly_level):
            cols = [
                rng.normal(vib,           0.05, n),
                rng.normal(strain,        0.04, n),
                rng.normal(0.10,          0.03, n),
                rng.normal(temp,          2.00, n),
                rng.normal(0.55,          0.10, n),
                rng.normal(0.04,          0.01, n),
                rng.normal(0.20,          0.08, n),
                rng.normal(anomaly_level, 0.08, n),
                rng.normal(0.50,          0.10, n),
                rng.normal(temp + 5,      3.00, n),
                rng.normal(0.02,          0.01, n),
                rng.normal(anomaly_level, 0.10, n),
                rng.normal(anomaly_level, 0.05, n),
            ] + [rng.normal(0.5, 0.1, n) for _ in range(11)]
            return np.stack(cols, axis=1)

        X_normal = make_sensors(n_normal, vib=0.10, strain=0.12, temp=22.0, anomaly_level=0.05)
        X_maint  = make_sensors(n_maint,  vib=0.35, strain=0.30, temp=26.0, anomaly_level=0.35)
        X_fault  = make_sensors(n_fault,  vib=0.75, strain=0.65, temp=38.0, anomaly_level=0.80)

        X = np.vstack([X_normal, X_maint, X_fault])
        y = np.concatenate([
            np.zeros(n_normal, dtype=int),
            np.ones(n_maint,   dtype=int),
            np.full(n_fault,   2, dtype=int),
        ])

        X = np.clip(X, 0.0, 1.0)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )

        return ConstructionDataset(
            name="BIM_Sensor_Streams",
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            feature_names=[
                "vibration_rms", "strain_gauge", "settlement_mm",
                "temperature_C", "humidity", "co2_normalised",
                "particulate", "hvac_anomaly", "pump_pressure",
                "bearing_temp", "voltage_fluctuation", "power_anomaly",
                "bim_geometry_deviation",
                *[f"sensor_{i}" for i in range(11)],
            ],
            class_names=self.CLASSES,
            description=(
                "Synthetic BIM/digital twin IoT sensor dataset for predictive "
                f"maintenance. {n_samples} samples across 3 fault classes. "
                "Simulates smart building or active construction site monitoring."
            ),
        )


class RebarPlacementDataset:
    """
    Autonomous Rebar Placement Robot Dataset.

    Simulates positioning observations from a rebar-placement robot guided
    by a real-time computer vision model.

    Classes
    -------
    0 = correct_placement
    1 = minor_deviation  (< 10mm, flag for review)
    2 = critical_deviation (> 10mm, structural risk)

    Features (16 per frame)
    -----------------------
    - Formwork geometry measurements
    - Visual markers relative positions
    - Robot arm joint angles
    - Force/torque sensor readings
    """

    N_FEATURES = 16
    CLASSES    = ["correct_placement", "minor_deviation", "critical_deviation"]

    def generate(
        self,
        n_samples:    int   = 1000,
        random_state: int   = 42,
    ) -> ConstructionDataset:
        rng = np.random.default_rng(random_state)

        n_correct = int(n_samples * 0.65)
        n_minor   = int(n_samples * 0.25)
        n_critical = n_samples - n_correct - n_minor

        def make_rebar(n, deviation):
            cols = [
                rng.normal(0.5,       0.05, n),
                rng.normal(0.5,       0.05, n),
                rng.normal(deviation, 0.05, n),
                rng.normal(deviation, 0.05, n),
                rng.normal(0.90,      0.05, n),
                rng.normal(0.5,       0.10, n),
                rng.normal(0.5,       0.10, n),
                rng.normal(0.5,       0.10, n),
                rng.normal(deviation, 0.08, n),
                rng.normal(deviation, 0.08, n),
                rng.normal(0.3,       0.10, n),
                rng.normal(0.8,       0.05, n),
            ] + [rng.normal(0.5, 0.10, n) for _ in range(4)]
            return np.stack(cols, axis=1)

        X_correct  = make_rebar(n_correct,  deviation=0.02)
        X_minor    = make_rebar(n_minor,    deviation=0.25)
        X_critical = make_rebar(n_critical, deviation=0.70)

        X = np.vstack([X_correct, X_minor, X_critical])
        y = np.concatenate([
            np.zeros(n_correct,   dtype=int),
            np.ones(n_minor,      dtype=int),
            np.full(n_critical,   2, dtype=int),
        ])

        X = np.clip(X, 0.0, 1.0)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )

        return ConstructionDataset(
            name="Rebar_Placement",
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            feature_names=[
                "formwork_x_align", "formwork_y_align",
                "rebar_x_offset",   "rebar_y_offset",
                "marker_confidence",
                "joint_angle_1", "joint_angle_2", "joint_angle_3",
                "force_x_Nm", "force_y_Nm", "torque_Nm",
                "vision_lock_score",
                "sensor_13", "sensor_14", "sensor_15", "sensor_16",
            ],
            class_names=self.CLASSES,
            description=(
                "Synthetic autonomous rebar-placement robot dataset. "
                f"{n_samples} samples, 3 placement quality classes. "
                "Critical deviations represent structural safety risks."
            ),
        )


class ConstructionBenchmark:
    """
    Unified benchmark loader for all three construction datasets.

    Provides a consistent interface for running attack/defense experiments
    across both construction-specific and standard benchmark data.

    Usage
    -----
    >>> bench = ConstructionBenchmark()
    >>> datasets = bench.load_all()
    >>> for name, ds in datasets.items():
    ...     print(ds)
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def load_ppe(self, **kwargs) -> ConstructionDataset:
        return PPEDataset().generate(random_state=self.random_state, **kwargs)

    def load_bim(self, **kwargs) -> ConstructionDataset:
        return BIMSensorDataset().generate(random_state=self.random_state, **kwargs)

    def load_rebar(self, **kwargs) -> ConstructionDataset:
        return RebarPlacementDataset().generate(random_state=self.random_state, **kwargs)

    def load_mnist_like(self, n_samples: int = 1000) -> ConstructionDataset:
        """Load a standard benchmark (sklearn digits) for comparison."""
        from sklearn.datasets import load_digits
        from sklearn.preprocessing import MinMaxScaler

        digits = load_digits(n_class=2)
        X = MinMaxScaler().fit_transform(digits.data)
        y = digits.target
        idx = np.random.default_rng(self.random_state).permutation(len(X))[:n_samples]
        X, y = X[idx], y[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state
        )
        return ConstructionDataset(
            name="MNIST_Benchmark",
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            feature_names=[f"pixel_{i}" for i in range(X.shape[1])],
            class_names=["digit_0", "digit_1"],
            description="Standard sklearn digits benchmark (binary) for baseline comparison.",
        )

    def load_all(self) -> dict:
        return {
            "ppe":        self.load_ppe(),
            "bim":        self.load_bim(),
            "rebar":      self.load_rebar(),
            "benchmark":  self.load_mnist_like(),
        }
