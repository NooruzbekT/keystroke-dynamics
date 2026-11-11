import numpy as np
import pytest

from keystroke.events import FORBIDDEN_KEYS, MAX_EVENTS, InvalidAttempt, RawEvent, attempt_duration, validate_attempt
from keystroke.features import compute_features, feature_names

PHRASE = "typing rhythm"


def sequential_events(phrase=PHRASE, hold=80.0, gap=120.0, start=1000.0):
    """Key i goes down at start + i*(hold+gap) ms and up `hold` ms later; no overlap."""
    events = []
    for i, ch in enumerate(phrase):
        down = start + i * (hold + gap)
        code = "Space" if ch == " " else f"Key{ch.upper()}"
        events += [RawEvent("down", ch, code, down), RawEvent("up", ch, code, down + hold)]
    return events


def test_features_known_answer():
    strokes = validate_attempt(sequential_events(), PHRASE, max_seconds=30)
    x = compute_features(strokes)
    assert x.shape == (37,)
    names = feature_names(PHRASE)
    # Seconds: H = 0.08, DD = 0.2, UD = 0.12 for every key / pair.
    for name, value in zip(names, x):
        expected = {"H": 0.08, "DD": 0.2, "UD": 0.12}[name.split(".")[0]]
        assert value == pytest.approx(expected)
    assert attempt_duration(strokes) == pytest.approx((12 * 200 + 80) / 1000)


def test_features_known_answer_irregular():
    ev = [
        RawEvent("down", "t", "KeyT", 0), RawEvent("up", "t", "KeyT", 100),
        RawEvent("down", "y", "KeyY", 250), RawEvent("up", "y", "KeyY", 300),
    ]
    strokes = validate_attempt(ev, "ty", max_seconds=30)
    np.testing.assert_allclose(compute_features(strokes), [0.1, 0.25, 0.15, 0.05])
    assert feature_names("ty") == ["H.0_t", "DD.0_t.1_y", "UD.0_t.1_y", "H.1_y"]


def test_negative_ud():
    # "y" goes down before "t" is released: rollover.
    ev = [
        RawEvent("down", "t", "KeyT", 0), RawEvent("down", "y", "KeyY", 60),
        RawEvent("up", "t", "KeyT", 100), RawEvent("up", "y", "KeyY", 150),
    ]
    x = compute_features(validate_attempt(ev, "ty", max_seconds=30))
    np.testing.assert_allclose(x, [0.1, 0.06, -0.04, 0.09])
    assert x[2] < 0


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda ev: ev[:-4], "wrong_phrase"),
        (lambda ev: ev + [RawEvent("down", "x", "KeyX", 1e6), RawEvent("up", "x", "KeyX", 1e6 + 50)], "wrong_phrase"),
        (lambda ev: ev[:4] + [RawEvent("down", "Backspace", "Backspace", ev[3].t_ms + 1)] + ev[4:], "invalid_attempt"),
        (lambda ev: ev[:1] + [RawEvent("down", "t", "KeyT", ev[0].t_ms + 30)] + ev[1:], "invalid_attempt"),
        (lambda ev: [RawEvent("up", "q", "KeyQ", 0)] + ev, "invalid_attempt"),
        (lambda ev: ev[:2] + [RawEvent("down", "u", "KeyU", ev[1].t_ms + 10)] + ev[2:], "wrong_phrase"),
        (lambda ev: ev[:-1], "invalid_attempt"),
        (lambda ev: ev[:2] + [RawEvent("down", "Shift", "ShiftLeft", ev[1].t_ms + 5)] + ev[2:], "invalid_attempt"),
    ],
    ids=["incomplete", "extra_char", "backspace", "autorepeat", "keyup_without_keydown", "wrong_char", "not_released", "shift"],
)
def test_reject_incomplete_and_wrong_text(mutate, code):
    with pytest.raises(InvalidAttempt) as exc:
        validate_attempt(mutate(sequential_events()), PHRASE, max_seconds=30)
    assert exc.value.error_code == code
    assert exc.value.message_ru


def test_reject_too_long_and_unordered():
    with pytest.raises(InvalidAttempt):
        validate_attempt(sequential_events(gap=3000.0), PHRASE, max_seconds=30)
    ev = sequential_events()
    ev[2], ev[3] = ev[3], ev[2]
    with pytest.raises(InvalidAttempt):
        validate_attempt(ev, PHRASE, max_seconds=30)


def test_enter_is_ignored():
    ev = sequential_events()
    ev = ev + [RawEvent("down", "Enter", "Enter", ev[-1].t_ms + 10), RawEvent("up", "Enter", "Enter", ev[-1].t_ms + 60)]
    assert compute_features(validate_attempt(ev, PHRASE, max_seconds=30)).shape == (37,)


@pytest.mark.parametrize("key", sorted(FORBIDDEN_KEYS))
def test_every_forbidden_key_rejected(key):
    ev = sequential_events()
    ev = ev[:2] + [RawEvent("down", key, key, ev[1].t_ms + 5)] + ev[2:]
    with pytest.raises(InvalidAttempt) as exc:
        validate_attempt(ev, PHRASE, max_seconds=30)
    assert exc.value.error_code == "invalid_attempt"


def test_duration_uses_latest_release():
    ev = [
        RawEvent("down", "t", "KeyT", 0), RawEvent("down", "y", "KeyY", 50),
        RawEvent("up", "y", "KeyY", 100), RawEvent("up", "t", "KeyT", 200),
    ]
    assert attempt_duration(validate_attempt(ev, "ty", max_seconds=30)) == pytest.approx(0.2)


def test_decreasing_timestamp_rejected():
    ev = [RawEvent("down", "t", "KeyT", 1000), RawEvent("up", "t", "KeyT", 900)]
    with pytest.raises(InvalidAttempt):
        validate_attempt(ev, "t", max_seconds=30)


def test_equal_timestamps_accepted():
    ev = [
        RawEvent("down", "t", "KeyT", 0), RawEvent("up", "t", "KeyT", 100),
        RawEvent("down", "y", "KeyY", 100), RawEvent("up", "y", "KeyY", 150),
    ]
    np.testing.assert_allclose(compute_features(validate_attempt(ev, "ty", max_seconds=30)), [0.1, 0.1, 0.0, 0.05])


def test_duration_limit_is_inclusive():
    ok = [RawEvent("down", "t", "KeyT", 0), RawEvent("up", "t", "KeyT", 10), RawEvent("down", "y", "KeyY", 20), RawEvent("up", "y", "KeyY", 30000)]
    assert len(validate_attempt(ok, "ty", max_seconds=30)) == 2
    too_long = ok[:3] + [RawEvent("up", "y", "KeyY", 30001)]
    with pytest.raises(InvalidAttempt):
        validate_attempt(too_long, "ty", max_seconds=30)


def test_max_events_boundary():
    ev = sequential_events()
    enter = [RawEvent("down", "Enter", "Enter", ev[-1].t_ms + 1)] * (MAX_EVENTS - len(ev))
    assert len(validate_attempt(ev + enter, PHRASE, max_seconds=30)) == len(PHRASE)
    with pytest.raises(InvalidAttempt, match="Слишком много"):
        validate_attempt(ev + enter + enter[:1], PHRASE, max_seconds=30)
