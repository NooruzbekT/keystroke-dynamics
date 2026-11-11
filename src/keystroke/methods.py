import logging
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)

MIN_STD = 1e-12


def _check_matrix(X: np.ndarray, n_features: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"expected a non-empty 2D array, got shape {X.shape}")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"expected {n_features} features, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise ValueError("input contains NaN or inf")
    return X


class StandardScaler:
    """Per-feature z-scaling with population std (ddof=0); features with std < MIN_STD get std=1."""

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = _check_matrix(X)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.zero_variance_ = np.flatnonzero(std < MIN_STD)
        if self.zero_variance_.size:
            log.warning("zero-variance features (std set to 1.0): %s", self.zero_variance_.tolist())
        self.std_ = np.where(std < MIN_STD, 1.0, std)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _check_matrix(X, self.n_features_)
        return (X - self.mean_) / self.std_


class _ProfileScorer:
    """Scorer contract: fit(X_train) -> self, score(X) -> array; higher score = more anomalous."""

    name: str

    def _fit_profile(self, X: np.ndarray) -> np.ndarray:
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        self.mu_ = Z.mean(axis=0)
        return Z

    def signed_deviations(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X) - self.mu_

    def deviations(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.signed_deviations(X))

    def profile_mean(self) -> np.ndarray:
        """Owner's mean feature vector in original units."""
        return self.scaler_.mean_ + self.mu_ * self.scaler_.std_


class MeanAbsoluteDistance(_ProfileScorer):
    """score = mean_j |z_j - mu_j|."""

    name = "mean_abs_distance"

    def fit(self, X: np.ndarray) -> "MeanAbsoluteDistance":
        self._fit_profile(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.deviations(X).mean(axis=1)

    def params(self) -> dict[str, Any]:
        return {}


class IsolationForestScorer(_ProfileScorer):
    """score = -IsolationForest.score_samples(z); contamination only sets sklearn's offset_, not these scores."""

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | float | str = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
        contamination: float | str = "auto",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.contamination = contamination

    def fit(self, X: np.ndarray) -> "IsolationForestScorer":
        Z = self._fit_profile(X)
        self.forest_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=self.random_state,
        ).fit(Z)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.forest_.score_samples(self.scaler_.transform(X))

    def params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "max_features": self.max_features,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }


METHODS = {cls.name: cls for cls in (MeanAbsoluteDistance, IsolationForestScorer)}


def make_method(name: str, params: dict[str, Any] | None = None) -> _ProfileScorer:
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; expected one of {sorted(METHODS)}")
    return METHODS[name](**(params or {}))
