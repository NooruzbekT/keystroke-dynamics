from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

REJECT_ALL_EPS = 1e-9


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    far: float
    frr: float
    avg_error: float
    false_accepts: int
    false_rejects: int
    n_impostor: int
    n_owner: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def exact_far(self) -> Fraction:
        return Fraction(self.false_accepts, self.n_impostor)

    def exact_avg_error(self) -> Fraction:
        return (Fraction(self.false_accepts, self.n_impostor) + Fraction(self.false_rejects, self.n_owner)) / 2


def far_frr(owner_scores: np.ndarray, impostor_scores: np.ndarray, t: float) -> tuple[float, float]:
    """Accept iff score <= t. FAR = share of impostor scores accepted, FRR = share of owner scores rejected."""
    owner_scores, impostor_scores = np.asarray(owner_scores), np.asarray(impostor_scores)
    far = float(np.mean(impostor_scores <= t))
    frr = float(np.mean(owner_scores > t))
    return far, frr


def select_threshold(owner_scores: np.ndarray, impostor_scores: np.ndarray) -> ThresholdResult:
    """Min (FAR+FRR)/2 over candidates = all input scores plus reject-all (min - REJECT_ALL_EPS); ties -> lower FAR, then lower t."""
    own = np.sort(np.asarray(owner_scores, dtype=np.float64))
    imp = np.sort(np.asarray(impostor_scores, dtype=np.float64))
    if own.size == 0 or imp.size == 0:
        raise ValueError("owner and impostor scores must be non-empty")

    cands = np.unique(np.concatenate([own, imp]))
    cands = np.concatenate([[cands[0] - REJECT_ALL_EPS], cands])

    n_own, n_imp = own.size, imp.size
    false_accepts = np.searchsorted(imp, cands, side="right")
    false_rejects = n_own - np.searchsorted(own, cands, side="right")
    # Integer form of (FAR+FRR)/2 * 2*n_own*n_imp keeps ties exact.
    err_key = false_accepts * n_own + false_rejects * n_imp

    best = np.lexsort((cands, false_accepts, err_key))[0]
    far = false_accepts[best] / n_imp
    frr = false_rejects[best] / n_own
    return ThresholdResult(
        threshold=float(cands[best]),
        far=float(far),
        frr=float(frr),
        avg_error=float((far + frr) / 2),
        false_accepts=int(false_accepts[best]),
        false_rejects=int(false_rejects[best]),
        n_impostor=int(n_imp),
        n_owner=int(n_own),
    )
