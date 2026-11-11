from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from keystroke.data import FEATURE_COLUMNS, META_COLUMNS
from keystroke.splits import SplitConfig, build_splits, feature_matrix, split_table

CFG = SplitConfig(
    n_owners=5,
    train_sessions=(1, 2, 3, 4),
    val_sessions=(5, 6),
    test_sessions=(7, 8),
    impostor_reps=(1, 2, 3, 4, 5),
)


def keys(frame: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, frame[list(META_COLUMNS)].itertuples(index=False)))


def test_splits_disjoint_and_no_leakage(cmu_like):
    s = build_splits(cmu_like, CFG)
    subjects = sorted(cmu_like.subject.unique())

    assert list(s.owners) == subjects[:5]
    assert len(s.pool_val) == 23 and len(s.pool_test) == 23
    assert not set(s.pool_val) & set(s.pool_test)
    assert not set(s.owners) & (set(s.pool_val) | set(s.pool_test))
    assert set(s.owners) | set(s.pool_val) | set(s.pool_test) == set(subjects)

    for owner, o in s.by_owner.items():
        sizes = [len(o.train), len(o.val_owner), len(o.test_owner), len(o.val_impostor), len(o.test_impostor)]
        assert sizes == [200, 100, 100, 920, 920]

        parts = [o.train, o.val_owner, o.test_owner, o.val_impostor, o.test_impostor]
        for i in range(len(parts)):
            for j in range(i + 1, len(parts)):
                assert not keys(parts[i]) & keys(parts[j])

        assert set(o.train.subject) == set(o.val_owner.subject) == set(o.test_owner.subject) == {owner}
        assert set(o.train.sessionIndex) == {1, 2, 3, 4}
        assert set(o.val_owner.sessionIndex) == {5, 6}
        assert set(o.test_owner.sessionIndex) == {7, 8}
        assert set(o.val_impostor.subject) == set(s.pool_val)
        assert set(o.test_impostor.subject) == set(s.pool_test)
        assert set(o.val_impostor.rep) == set(o.test_impostor.rep) == {1, 2, 3, 4, 5}
        assert o.val_impostor.groupby("subject").size().eq(40).all()

        X = feature_matrix(o.train)
        assert X.shape == (200, len(FEATURE_COLUMNS))
        np.testing.assert_array_equal(X, o.train[FEATURE_COLUMNS].to_numpy())


def test_splits_deterministic_regardless_of_row_order(cmu_like):
    a = split_table(build_splits(cmu_like, CFG))
    shuffled = cmu_like.sample(frac=1.0, random_state=123).reset_index(drop=True)
    b = split_table(build_splits(shuffled, CFG))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 5 * 400 + 2 * 920


def test_pool_is_interleaved(cmu_like):
    s = build_splits(cmu_like, CFG)
    subjects = sorted(cmu_like.subject.unique())
    assert list(s.pool_val) == subjects[5::2]
    assert list(s.pool_test) == subjects[6::2]


def test_tuning_view_has_no_test_roles(cmu_like):
    s = build_splits(cmu_like, CFG, include_test=False)
    for o in s.by_owner.values():
        assert o.test_owner is None and o.test_impostor is None
        assert len(o.train) == 200 and len(o.val_owner) == 100 and len(o.val_impostor) == 920
    with pytest.raises(ValueError):
        split_table(s)


@pytest.mark.parametrize(
    "change",
    [
        {"val_sessions": (4, 5)},
        {"val_sessions": (9,)},
        {"impostor_reps": (51,)},
        {"impostor_reps": ()},
        {"n_owners": 0},
    ],
)
def test_invalid_split_config_rejected(cmu_like, change):
    with pytest.raises(ValueError):
        build_splits(cmu_like, replace(CFG, **change))
