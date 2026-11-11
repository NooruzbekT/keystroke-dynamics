import numpy as np
import pytest

from keystroke.thresholds import far_frr, select_threshold


def test_accept_is_score_le_threshold():
    far, frr = far_frr(owner_scores=np.array([1.0, 2.0]), impostor_scores=np.array([2.0, 3.0]), t=2.0)
    assert far == 0.5
    assert frr == 0.0


def test_tie_on_avg_error_prefers_lower_far():
    # t=2: FAR 0, FRR .5; t=3: FAR .25, FRR .25; t=4: FAR .5, FRR 0 -> all avg .25.
    owner = np.array([1.0, 2.0, 3.0, 4.0])
    impostor = np.array([2.5, 3.5, 6.0, 7.0])
    r = select_threshold(owner, impostor)
    assert r.threshold == 2.0
    assert (r.far, r.frr, r.avg_error) == (0.0, 0.5, 0.25)


def test_perfect_separation():
    r = select_threshold(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.9]))
    assert r.threshold == 0.3
    assert (r.far, r.frr) == (0.0, 0.0)


def test_reject_all_candidate_wins_when_impostors_score_lower():
    r = select_threshold(np.array([5.0, 6.0]), np.array([1.0, 2.0]))
    assert r.threshold < 1.0
    assert (r.far, r.frr) == (0.0, 1.0)


def test_selected_metrics_match_far_frr():
    rng = np.random.default_rng(42)
    own, imp = rng.normal(0, 1, 100), rng.normal(1.5, 1, 920)
    r = select_threshold(own, imp)
    assert (r.far, r.frr) == far_frr(own, imp, r.threshold)
    grid = np.unique(np.concatenate([own, imp]))
    best = min(sum(far_frr(own, imp, t)) / 2 for t in grid)
    assert r.avg_error == pytest.approx(best)


def test_select_threshold_counts_equal_impostor_as_accepted():
    r = select_threshold(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
    assert (r.threshold, r.far, r.frr) == (1.0, 0.0, 0.5)


def test_exact_tie_not_broken_by_float_rounding():
    # t=5: 0.4+0.4; t=7: 0.7+0.1 evaluates to 0.7999999999999999 in float.
    own = np.array([1, 2, 3, 3, 4, 5, 6, 6, 7, 8], dtype=float)
    imp = np.array([0, 3, 3, 3, 6, 6, 6, 8, 9, 10], dtype=float)
    r = select_threshold(own, imp)
    assert (r.threshold, r.far, r.frr) == (5.0, 0.4, 0.4)
    assert (r.false_accepts, r.false_rejects, r.n_impostor, r.n_owner) == (4, 4, 10, 10)
