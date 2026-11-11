import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from keystroke.data import FEATURE_COLUMNS, DataError
from keystroke.methods import make_method
from keystroke.splits import OwnerSplit, Splits, feature_matrix
from keystroke.thresholds import far_frr


def load_frozen(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataError(f"{path} not found. Run `python scripts/tune.py` first; evaluation never tunes.")
    return json.loads(path.read_text(encoding="utf-8"))


def check_frozen(frozen: dict[str, Any], data_sha256: str, splits: Splits) -> None:
    errors = []
    if frozen["data_sha256"] != data_sha256:
        errors.append("data_sha256 differs from config/experiment.yaml")
    if frozen["feature_columns"] != FEATURE_COLUMNS:
        errors.append("feature column order differs")
    for key in ("owners", "pool_val", "pool_test"):
        if frozen["split"][key] != list(getattr(splits, key)):
            errors.append(f"split.{key} differs")
    if errors:
        raise DataError("frozen.json does not match the current data/split:\n- " + "\n- ".join(errors))


@dataclass(frozen=True)
class OwnerScores:
    model: Any
    threshold: float
    test_owner: np.ndarray
    test_impostor: np.ndarray


def score_frozen(method: str, params: dict[str, Any], owner_frozen: dict[str, Any], split: OwnerSplit) -> OwnerScores:
    """Refit on owner train with frozen params, require validation FAR/FRR to reproduce exactly, score test."""
    if split.test_owner is None or split.test_impostor is None:
        raise ValueError("splits must be built with include_test=True")
    model = make_method(method, params).fit(feature_matrix(split.train))
    t = float(owner_frozen["threshold"])
    val = far_frr(model.score(feature_matrix(split.val_owner)), model.score(feature_matrix(split.val_impostor)), t)
    if val != (owner_frozen["val_FAR"], owner_frozen["val_FRR"]):
        raise DataError(
            f"{method}/{split.owner}: refitted model gives validation FAR/FRR {val}, "
            f"frozen {(owner_frozen['val_FAR'], owner_frozen['val_FRR'])}; model is not reproduced."
        )
    return OwnerScores(
        model=model,
        threshold=t,
        test_owner=model.score(feature_matrix(split.test_owner)),
        test_impostor=model.score(feature_matrix(split.test_impostor)),
    )
