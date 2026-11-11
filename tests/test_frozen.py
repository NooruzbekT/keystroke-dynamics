import json

import numpy as np
import pytest

from keystroke.data import DataError
from keystroke.frozen import check_frozen, load_frozen, score_frozen
from keystroke.methods import MeanAbsoluteDistance
from keystroke.splits import build_splits, feature_matrix
from keystroke.thresholds import select_threshold
from test_splits import CFG


def frozen_for(split):
    model = MeanAbsoluteDistance().fit(feature_matrix(split.train))
    r = select_threshold(model.score(feature_matrix(split.val_owner)), model.score(feature_matrix(split.val_impostor)))
    return {"threshold": r.threshold, "val_FAR": r.far, "val_FRR": r.frr}


def test_load_frozen_missing_file(tmp_path):
    with pytest.raises(DataError, match="never tunes"):
        load_frozen(tmp_path / "frozen.json")


def test_score_frozen_reproduces_validation_and_scores_test(cmu_like):
    split = build_splits(cmu_like, CFG).by_owner["s002"]
    owner_frozen = frozen_for(split)
    s = score_frozen("mean_abs_distance", {}, owner_frozen, split)
    assert s.test_owner.shape == (100,) and s.test_impostor.shape == (920,)
    assert s.threshold == owner_frozen["threshold"]
    np.testing.assert_array_equal(s.test_owner, s.model.score(feature_matrix(split.test_owner)))
    np.testing.assert_array_equal(s.test_impostor, s.model.score(feature_matrix(split.test_impostor)))


def test_score_frozen_rejects_mismatched_validation(cmu_like):
    split = build_splits(cmu_like, CFG).by_owner["s002"]
    owner_frozen = {**frozen_for(split), "val_FRR": 0.999}
    with pytest.raises(DataError, match="not reproduced"):
        score_frozen("mean_abs_distance", {}, owner_frozen, split)


def test_score_frozen_requires_test_roles(cmu_like):
    split = build_splits(cmu_like, CFG, include_test=False).by_owner["s002"]
    with pytest.raises(ValueError):
        score_frozen("mean_abs_distance", {}, {"threshold": 1.0, "val_FAR": 0.0, "val_FRR": 0.0}, split)


def test_check_frozen_detects_drift(cmu_like, tmp_path):
    splits = build_splits(cmu_like, CFG)
    from keystroke.data import FEATURE_COLUMNS

    frozen = {
        "data_sha256": "a" * 64,
        "feature_columns": FEATURE_COLUMNS,
        "split": {"owners": list(splits.owners), "pool_val": list(splits.pool_val), "pool_test": list(splits.pool_test)},
    }
    check_frozen(frozen, "a" * 64, splits)
    with pytest.raises(DataError, match="data_sha256"):
        check_frozen(frozen, "b" * 64, splits)
    swapped = json.loads(json.dumps(frozen))
    swapped["split"]["pool_val"], swapped["split"]["pool_test"] = swapped["split"]["pool_test"], swapped["split"]["pool_val"]
    with pytest.raises(DataError, match="pool_val"):
        check_frozen(swapped, "a" * 64, splits)


@pytest.mark.parametrize("key", ["owners", "pool_val", "pool_test", "feature_columns"])
def test_check_frozen_each_key(cmu_like, key):
    from keystroke.data import FEATURE_COLUMNS

    splits = build_splits(cmu_like, CFG)
    frozen = {
        "data_sha256": "a" * 64,
        "feature_columns": list(FEATURE_COLUMNS),
        "split": {"owners": list(splits.owners), "pool_val": list(splits.pool_val), "pool_test": list(splits.pool_test)},
    }
    target = frozen if key == "feature_columns" else frozen["split"]
    target[key] = list(reversed(target[key]))
    with pytest.raises(DataError, match=key.split("_")[0]):
        check_frozen(frozen, "a" * 64, splits)
