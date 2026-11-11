import numpy as np

from keystroke.events import Keystroke


def key_label(ch: str) -> str:
    return "space" if ch == " " else ch


def feature_names(phrase: str) -> list[str]:
    """CMU-style interleaving H, DD, UD per key pair; position prefix keeps repeated letters unique."""
    keys = [f"{i}_{key_label(c)}" for i, c in enumerate(phrase)]
    names: list[str] = []
    for a, b in zip(keys, keys[1:]):
        names += [f"H.{a}", f"DD.{a}.{b}", f"UD.{a}.{b}"]
    names.append(f"H.{keys[-1]}")
    return names


def feature_schema(phrase: str) -> dict:
    return {"phrase": phrase, "names": feature_names(phrase), "units": "seconds"}


def compute_features(strokes: list[Keystroke]) -> np.ndarray:
    """H_i = up_i - down_i; DD = down_{i+1} - down_i; UD = down_{i+1} - up_i (negative on rollover)."""
    values: list[float] = []
    for a, b in zip(strokes, strokes[1:]):
        values += [a.up - a.down, b.down - a.down, b.down - a.up]
    values.append(strokes[-1].up - strokes[-1].down)
    return np.asarray(values, dtype=np.float64)


def group_indices(names: list[str], prefix: str) -> list[int]:
    return [i for i, n in enumerate(names) if n.startswith(prefix)]
