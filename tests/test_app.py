import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from keystroke.events import validate_attempt
from keystroke.features import compute_features, feature_names
from keystroke.experiment_log import read_records
from keystroke.profile_store import PROFILE_FILE, Profile, ProfileError, ProfileStore
from keystroke.server import BodyLimitMiddleware, create_app
from test_features import PHRASE, sequential_events

N_TRAIN, N_CALIB = 30, 10


def attempt(rng, hold=80.0, gap=120.0):
    events, t = [], 1000.0
    for ch in PHRASE:
        code = "Space" if ch == " " else f"Key{ch.upper()}"
        h = hold + rng.normal(0, 8)
        events += [{"type": "down", "key": ch, "code": code, "t": t}, {"type": "up", "key": ch, "code": code, "t": t + h}]
        t += h + gap + rng.normal(0, 15)
    return {"events": events}


def to_x(payload):
    from keystroke.events import RawEvent

    raw = [RawEvent(e["type"], e["key"], e["code"], e["t"]) for e in payload["events"]]
    return compute_features(validate_attempt(raw, PHRASE, 30))


def filled_store(tmp_path, rng):
    store = ProfileStore(tmp_path / "profiles", PHRASE, N_TRAIN, N_CALIB, 95)
    for _ in range(N_TRAIN + N_CALIB):
        store.add_enrollment(to_x(attempt(rng)))
    return store


def test_feature_schema_consistency(tmp_path):
    names = feature_names(PHRASE)
    assert len(names) == 37 and len(set(names)) == 37
    assert names[:4] == ["H.0_t", "DD.0_t.1_y", "UD.0_t.1_y", "H.1_y"]
    assert names[-1] == "H.12_m" and "H.6_space" in names
    assert compute_features(validate_attempt(sequential_events(), PHRASE, 30)).shape == (len(names),)

    store = filled_store(tmp_path, np.random.default_rng(0))
    store.build("mean_abs_distance", {})
    meta_path = store.active_dir / PROFILE_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["names"] == names and meta["units"] == "seconds" and len(meta["profile_mean"]) == 37

    meta["names"] = list(reversed(meta["names"]))
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    fresh = ProfileStore(tmp_path / "profiles", PHRASE, N_TRAIN, N_CALIB, 95)
    with pytest.raises(ProfileError):
        fresh.load_active()


@pytest.mark.parametrize("method, params", [("mean_abs_distance", {}), ("isolation_forest", {"n_estimators": 100, "max_samples": "auto", "max_features": 0.5, "random_state": 42})])
def test_profile_roundtrip(tmp_path, method, params):
    rng = np.random.default_rng(1)
    store = filled_store(tmp_path, rng)
    built = store.build(method, params)
    probes = [to_x(attempt(rng)) for _ in range(5)] + [to_x(attempt(rng, hold=160, gap=60))]
    before = [built.verify(x) for x in probes]

    loaded = ProfileStore(tmp_path / "profiles", PHRASE, N_TRAIN, N_CALIB, 95).load_active()
    after = [loaded.verify(x) for x in probes]
    assert [a for a, _ in before] == [a for a, _ in after]
    np.testing.assert_allclose([s for _, s in before], [s for _, s in after])
    calib = np.asarray(loaded.meta["calib_scores"])
    assert loaded.threshold == pytest.approx(np.percentile(calib, 95, method="linear"))


def test_tampered_model_file_is_not_loaded(tmp_path):
    store = filled_store(tmp_path, np.random.default_rng(2))
    store.build("mean_abs_distance", {})
    with (store.active_dir / "model.joblib").open("ab") as f:
        f.write(b"\0")
    with pytest.raises(ProfileError):
        ProfileStore(tmp_path / "profiles", PHRASE, N_TRAIN, N_CALIB, 95).load_active()


@pytest.fixture
def client(tmp_path):
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"app_method": {"name": "mean_abs_distance", "params": {}}}), encoding="utf-8")
    app = create_app(profiles_dir=tmp_path / "profiles", logs_dir=tmp_path / "logs", frozen_path=frozen)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c, tmp_path


def test_api_full_cycle(client):
    c, tmp_path = client
    rng = np.random.default_rng(3)

    assert c.get("/api/profile").json()["status"] == "none"
    r = c.post("/api/verify", json=attempt(rng))
    assert r.status_code == 404 and r.json()["error_code"] == "no_profile"

    for i in range(N_TRAIN + N_CALIB - 1):
        assert c.post("/api/enroll/attempt", json=attempt(rng)).status_code == 200
    r = c.post("/api/profile/build", json={"confirm": True})
    assert r.json()["error_code"] == "enrollment_incomplete"
    last = c.post("/api/enroll/attempt", json=attempt(rng)).json()
    assert (last["role"], last["n_train"], last["n_calib"]) == ("calib", 30, 10)
    assert c.post("/api/enroll/attempt", json=attempt(rng)).json()["error_code"] == "enrollment_complete"

    status = c.post("/api/profile/build", json={"confirm": True}).json()
    assert status["status"] == "ready" and status["method"] == "mean_abs_distance"

    v = c.post("/api/verify", json=attempt(rng)).json()
    assert set(v) >= {"accepted", "score", "threshold", "features", "profile_mean", "duration"}
    assert len(v["features"]["values"]) == len(v["profile_mean"]) == 37
    assert v["accepted"] == (v["score"] <= v["threshold"])

    e = c.post("/api/experiment/attempt", json={**attempt(rng, gap=400), "condition": "slow"}).json()
    assert e["counts"]["slow"] == 1
    assert (tmp_path / "logs" / "own_experiment.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert c.get("/api/profile").json()["threshold"] == status["threshold"]

    assert c.post("/api/profile/reset", json={}).json()["error_code"] == "invalid_attempt"
    reset = c.post("/api/profile/reset", json={"confirm": True}).json()
    assert reset["status"] == "none" and reset["archived"]
    assert (tmp_path / "profiles" / "archive" / reset["archived"] / PROFILE_FILE).is_file()


def test_api_restart_loads_profile(client):
    c, tmp_path = client
    rng = np.random.default_rng(4)
    for _ in range(N_TRAIN + N_CALIB):
        c.post("/api/enroll/attempt", json=attempt(rng))
    c.post("/api/profile/build", json={"confirm": True})
    probe = attempt(rng)
    first = c.post("/api/verify", json=probe).json()

    app2 = create_app(profiles_dir=tmp_path / "profiles", logs_dir=tmp_path / "logs", frozen_path=tmp_path / "frozen.json")
    with TestClient(app2, base_url="http://127.0.0.1") as c2:
        assert c2.get("/api/profile").json()["status"] == "ready"
        second = c2.post("/api/verify", json=probe).json()
    assert (first["accepted"], first["score"]) == (second["accepted"], second["score"])


def test_api_rejects_bad_input(client):
    c, _ = client
    rng = np.random.default_rng(5)
    bad = attempt(rng)
    bad["events"] = bad["events"][:-4]
    r = c.post("/api/enroll/attempt", json=bad)
    assert r.status_code == 422 and r.json()["error_code"] == "wrong_phrase"
    assert c.post("/api/enroll/attempt", json={"events": []}).json()["error_code"] == "invalid_attempt"
    assert c.post("/api/enroll/attempt", json={"events": [{"type": "down", "key": "t", "code": "KeyT", "t": "nan"}]}).json()["error_code"] == "invalid_attempt"
    assert c.post("/api/enroll/attempt", content=b"x" * 70000, headers={"content-type": "application/json"}).status_code == 413
    assert c.get("/api/profile", headers={"host": "evil.example"}).status_code == 400
    assert c.get("/api/profile").json()["n_train"] == 0


def build_ready(c, rng):
    for _ in range(N_TRAIN + N_CALIB):
        c.post("/api/enroll/attempt", json=attempt(rng))
    return c.post("/api/profile/build", json={"confirm": True}).json()


def test_build_requires_confirmation(client):
    c, _ = client
    rng = np.random.default_rng(6)
    for _ in range(N_TRAIN + N_CALIB):
        c.post("/api/enroll/attempt", json=attempt(rng))
    assert c.post("/api/profile/build").json()["error_code"] == "invalid_attempt"
    assert c.post("/api/profile/build", content=b"confirm=true", headers={"content-type": "text/plain"}).json()["error_code"] == "invalid_attempt"
    assert c.get("/api/profile").json()["status"] == "collecting"


def test_verify_during_collection_reports_incomplete(client):
    c, _ = client
    rng = np.random.default_rng(7)
    c.post("/api/enroll/attempt", json=attempt(rng))
    r = c.post("/api/verify", json=attempt(rng))
    assert r.status_code == 409 and r.json()["error_code"] == "enrollment_incomplete"


def test_chunked_body_over_limit_rejected(client):
    c, _ = client
    r = c.post("/api/enroll/attempt", content=iter([b"x" * 40000, b"x" * 40000]), headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_experiment_series_capped_and_counted_per_profile(client):
    c, tmp_path = client
    rng = np.random.default_rng(8)
    build_ready(c, rng)
    for i in range(10):
        r = c.post("/api/experiment/attempt", json={**attempt(rng), "condition": "normal"})
        assert r.status_code == 200 and r.json()["counts"]["normal"] == i + 1
    r = c.post("/api/experiment/attempt", json={**attempt(rng), "condition": "normal"})
    assert r.status_code == 409 and r.json()["error_code"] == "series_complete"
    log = tmp_path / "logs" / "own_experiment.jsonl"
    assert log.read_text(encoding="utf-8").count("\n") == 10

    c.post("/api/profile/reset", json={"confirm": True})
    build_ready(c, rng)
    assert c.get("/api/experiment").json()["counts"] == {"normal": 0, "slow": 0, "pause": 0}


def test_torn_log_line_does_not_break_experiment(client):
    c, tmp_path = client
    rng = np.random.default_rng(9)
    build_ready(c, rng)
    c.post("/api/experiment/attempt", json={**attempt(rng), "condition": "slow"})
    log = tmp_path / "logs" / "own_experiment.jsonl"
    with log.open("a", encoding="utf-8") as f:
        f.write('{"condition": "slow", "sco')
    r = c.post("/api/experiment/attempt", json={**attempt(rng), "condition": "slow"})
    assert r.status_code == 200 and r.json()["counts"]["slow"] == 2
    assert c.get("/api/experiment").json()["counts"]["slow"] == 2


def test_build_uses_disjoint_train_and_calib(tmp_path):
    store = ProfileStore(tmp_path / "profiles", PHRASE, N_TRAIN, N_CALIB, 95)
    for _ in range(N_TRAIN):
        store.add_enrollment(np.full(37, 0.1))
    for _ in range(N_CALIB):
        store.add_enrollment(np.full(37, 0.3))
    p = store.build("mean_abs_distance", {})
    np.testing.assert_allclose(p.meta["profile_mean"], 0.1)
    np.testing.assert_allclose(p.meta["calib_scores"], 0.2)


def test_pending_schema_mismatch_rejected(tmp_path):
    store = ProfileStore(tmp_path / "profiles", PHRASE, N_TRAIN, N_CALIB, 95)
    store.pending_path.parent.mkdir(parents=True)
    store.pending_path.write_text(json.dumps({"schema": ["x"], "train": [], "calib": []}), encoding="utf-8")
    with pytest.raises(ProfileError):
        store.status()


def test_second_build_rejected(client):
    c, tmp_path = client
    rng = np.random.default_rng(10)
    build_ready(c, rng)
    r = c.post("/api/profile/build", json={"confirm": True})
    assert r.status_code == 409 and r.json()["error_code"] == "profile_exists"
    r = c.post("/api/enroll/attempt", json=attempt(rng))
    assert r.status_code == 409 and r.json()["error_code"] == "profile_exists"


def test_verify_accepts_score_equal_to_threshold():
    class Stub:
        def score(self, X):
            return np.array([1.0])

    assert Profile(Stub(), {"threshold": 1.0}).verify(np.zeros(3)) == (True, 1.0)
    assert Profile(Stub(), {"threshold": 0.999}).verify(np.zeros(3)) == (False, 1.0)


@pytest.mark.parametrize("body", [{"confirm": False}, {"confirm": "yes"}, {"confirm": 1}, {}])
def test_build_and_reset_need_literal_true(client, body):
    c, _ = client
    rng = np.random.default_rng(11)
    for _ in range(N_TRAIN + N_CALIB):
        c.post("/api/enroll/attempt", json=attempt(rng))
    assert c.post("/api/profile/build", json=body).json()["error_code"] == "invalid_attempt"
    assert c.post("/api/profile/reset", json=body).json()["error_code"] == "invalid_attempt"
    assert c.get("/api/profile").json()["status"] == "collecting"


def test_verify_when_all_collected_but_not_built(client):
    c, _ = client
    rng = np.random.default_rng(12)
    for _ in range(N_TRAIN + N_CALIB):
        c.post("/api/enroll/attempt", json=attempt(rng))
    r = c.post("/api/verify", json=attempt(rng)).json()
    assert r["error_code"] == "enrollment_incomplete" and "Создать профиль" in r["message_ru"]


def test_negative_time_rejected(client):
    c, _ = client
    payload = attempt(np.random.default_rng(13))
    payload["events"][0]["t"] = -1
    assert c.post("/api/enroll/attempt", json=payload).json()["error_code"] == "invalid_attempt"


def test_experiment_log_record_content(client):
    c, tmp_path = client
    rng = np.random.default_rng(14)
    status = build_ready(c, rng)
    r = c.post("/api/experiment/attempt", json={**attempt(rng), "condition": "pause"}).json()
    rec = json.loads((tmp_path / "logs" / "own_experiment.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["condition"] == "pause" and rec["score"] == r["score"] and rec["threshold"] == r["threshold"]
    assert len(rec["features"]) == 37 and rec["profile_created_utc"] == status["created_utc"]


def test_read_records_skips_invalid_lines(tmp_path):
    good = {"condition": "normal", "profile_created_utc": "p", "score": 0.1}
    log = tmp_path / "log.jsonl"
    log.write_bytes(
        (json.dumps(good) + "\n").encode()
        + b'{"condition": "fast", "profile_created_utc": "p"}\n'
        + b'{"condition": "normal"}\n'
        + b"[" * 100000 + b"\n"
        + "{\"condition\": \"обычный\"}\n".encode("cp1251")
    )
    records, skipped = read_records(log)
    assert records == [good] and skipped == 4


def _asgi_call(app, headers, chunks):
    sent, calls = [], {"n": 0}

    async def receive():
        calls["n"] += 1
        if not chunks:
            return {"type": "http.disconnect"}
        body = chunks.pop(0)
        return {"type": "http.request", "body": body, "more_body": bool(chunks)}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "headers": headers, "path": "/", "query_string": b""}
    asyncio.run(app(scope, receive, send))
    return sent, calls["n"]


def test_body_limit_middleware_asgi():
    seen = {}

    async def inner(scope, receive, send):
        seen["body"] = (await receive())["body"]
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = BodyLimitMiddleware(inner, max_bytes=10)
    sent, n = _asgi_call(mw, [(b"content-length", b"11")], [b"x" * 11])
    assert sent[0]["status"] == 413 and n == 0
    sent, n = _asgi_call(mw, [(b"content-length", b"1e3")], [b"x"])
    assert sent[0]["status"] == 413 and n == 0
    sent, _ = _asgi_call(mw, [], [b"x" * 5, b"x" * 5, b"x" * 5])
    assert sent[0]["status"] == 413 and "body" not in seen
    sent, _ = _asgi_call(mw, [], [b"ab", b"cd", b"ef"])
    assert sent[0]["status"] == 200 and seen["body"] == b"abcdef"
