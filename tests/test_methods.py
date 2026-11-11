import numpy as np
import pytest

from keystroke.methods import IsolationForestScorer, MeanAbsoluteDistance, StandardScaler, make_method


def test_zero_variance_feature_gives_finite_scores():
    rng = np.random.default_rng(0)
    X = rng.normal(0.1, 0.02, size=(50, 4))
    X[:, 2] = 0.08
    m = MeanAbsoluteDistance().fit(X)
    assert m.scaler_.zero_variance_.tolist() == [2]
    assert m.scaler_.std_[2] == 1.0

    probe = rng.normal(0.1, 0.02, size=(10, 4))
    probe[:, 2] = 0.5
    s = m.score(probe)
    assert np.isfinite(s).all()
    assert np.isfinite(m.score(X)).all()


def test_scaler_uses_train_statistics_only():
    train = np.array([[0.0, 10.0], [2.0, 30.0]])
    other = np.array([[100.0, 1000.0]])
    sc = StandardScaler().fit(train)
    np.testing.assert_allclose(sc.mean_, [1.0, 20.0])
    np.testing.assert_allclose(sc.std_, [1.0, 10.0])
    np.testing.assert_allclose(sc.transform(other), [[99.0, 98.0]])


def test_mean_abs_distance_outlier_scores_higher():
    rng = np.random.default_rng(1)
    X = rng.normal(0.1, 0.02, size=(200, 31))
    m = MeanAbsoluteDistance().fit(X)
    inlier = X.mean(axis=0, keepdims=True)
    outlier = inlier + 0.2
    assert m.score(outlier)[0] > m.score(inlier)[0]
    assert (m.score(X) >= 0).all()


def test_rejects_wrong_dimension_and_nan():
    m = MeanAbsoluteDistance().fit(np.random.default_rng(2).normal(size=(20, 5)))
    with pytest.raises(ValueError):
        m.score(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        m.score(np.full((1, 5), np.nan))


def test_isolation_forest_outlier_scores_higher():
    rng = np.random.default_rng(3)
    X = rng.normal(0.1, 0.02, size=(200, 31))
    m = IsolationForestScorer(n_estimators=100, random_state=42).fit(X)
    inlier = X.mean(axis=0, keepdims=True)
    outlier = inlier + 0.3
    s_in, s_out = m.score(inlier)[0], m.score(outlier)[0]
    assert s_out > s_in
    assert m.score(X).mean() < s_out


def test_isolation_forest_is_deterministic_with_seed():
    X = np.random.default_rng(4).normal(size=(100, 6))
    a = IsolationForestScorer(max_samples=64, max_features=0.5, random_state=42).fit(X).score(X)
    b = IsolationForestScorer(max_samples=64, max_features=0.5, random_state=42).fit(X).score(X)
    np.testing.assert_array_equal(a, b)


def test_make_method_and_profile_mean():
    X = np.random.default_rng(5).normal(1.0, 0.1, size=(50, 3))
    m = make_method("mean_abs_distance").fit(X)
    np.testing.assert_allclose(m.profile_mean(), X.mean(axis=0))
    assert make_method("isolation_forest", {"n_estimators": 10}).n_estimators == 10
    with pytest.raises(ValueError):
        make_method("unknown")


def test_mad_known_answer():
    m = MeanAbsoluteDistance().fit(np.array([[0.0, 10.0], [2.0, 30.0]]))
    s = m.score(np.array([[1.0, 50.0], [0.0, 20.0], [1.0, 20.0]]))
    np.testing.assert_allclose(s, [1.5, 0.5, 0.0])


def test_isolation_forest_score_is_negated_score_samples():
    X = np.random.default_rng(3).normal(0.1, 0.02, size=(200, 31))
    m = IsolationForestScorer().fit(X)
    s = m.score(X)
    np.testing.assert_allclose(s, -m.forest_.score_samples(m.scaler_.transform(X)))
    assert ((s > 0) & (s <= 1)).all()


def test_isolation_forest_invariant_to_affine_rescale():
    X = np.random.default_rng(6).normal(0.1, 0.02, size=(100, 6))
    probe = np.random.default_rng(7).normal(0.1, 0.05, size=(20, 6))
    a = IsolationForestScorer(random_state=42).fit(X).score(probe)
    b = IsolationForestScorer(random_state=42).fit(X * 1000 + 5).score(probe * 1000 + 5)
    np.testing.assert_allclose(a, b)


def test_isolation_forest_params_reach_forest():
    X = np.random.default_rng(8).normal(size=(100, 6))
    f = IsolationForestScorer(n_estimators=30, max_samples=64, max_features=0.5).fit(X).forest_
    assert (f.n_estimators, f.max_samples, f.max_features) == (30, 64, 0.5)
