from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from keystroke.data import FEATURE_COLUMNS, META_COLUMNS

ROLES = ("train", "val_owner", "test_owner", "val_impostor", "test_impostor")
SHARED_OWNER = "*"


@dataclass(frozen=True)
class SplitConfig:
    n_owners: int
    train_sessions: tuple[int, ...]
    val_sessions: tuple[int, ...]
    test_sessions: tuple[int, ...]
    impostor_reps: tuple[int, ...]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitConfig":
        return cls(
            n_owners=int(d["n_owners"]),
            train_sessions=tuple(d["train_sessions"]),
            val_sessions=tuple(d["val_sessions"]),
            test_sessions=tuple(d["test_sessions"]),
            impostor_reps=tuple(d["impostor_reps"]),
        )


@dataclass(frozen=True)
class OwnerSplit:
    owner: str
    train: pd.DataFrame
    val_owner: pd.DataFrame
    test_owner: pd.DataFrame | None
    val_impostor: pd.DataFrame
    test_impostor: pd.DataFrame | None


@dataclass(frozen=True)
class Splits:
    owners: tuple[str, ...]
    pool_val: tuple[str, ...]
    pool_test: tuple[str, ...]
    by_owner: dict[str, OwnerSplit]


def assign_subjects(subjects: list[str], n_owners: int) -> tuple[list[str], list[str], list[str]]:
    # String sort equals numeric sort for zero-padded sNNN ids.
    ordered = sorted(set(subjects))
    if n_owners < 1:
        raise ValueError(f"n_owners must be >= 1, got {n_owners}")
    if len(ordered) <= n_owners + 1:
        raise ValueError(f"need more than {n_owners + 1} subjects, got {len(ordered)}")
    owners, pool = ordered[:n_owners], ordered[n_owners:]
    return owners, pool[0::2], pool[1::2]


def _rows(df: pd.DataFrame, subjects: list[str], sessions: tuple[int, ...], reps=None) -> pd.DataFrame:
    mask = df["subject"].isin(subjects) & df["sessionIndex"].isin(sessions)
    if reps is not None:
        mask &= df["rep"].isin(reps)
    return df.loc[mask].sort_values(list(META_COLUMNS)).reset_index(drop=True)


def _check_config(df: pd.DataFrame, cfg: SplitConfig) -> tuple[int, ...]:
    all_sessions = cfg.train_sessions + cfg.val_sessions + cfg.test_sessions
    if len(set(all_sessions)) != len(all_sessions):
        raise ValueError("train/val/test sessions overlap")
    for name, values in (("train", cfg.train_sessions), ("val", cfg.val_sessions), ("test", cfg.test_sessions)):
        if not values:
            raise ValueError(f"{name}_sessions is empty")
    missing_sessions = set(all_sessions) - set(df["sessionIndex"].unique().tolist())
    missing_reps = set(cfg.impostor_reps) - set(df["rep"].unique().tolist())
    if missing_sessions or missing_reps or not cfg.impostor_reps:
        raise ValueError(
            f"split config refers to absent data: sessions {sorted(missing_sessions)}, "
            f"impostor_reps {sorted(missing_reps) or ('empty' if not cfg.impostor_reps else [])}"
        )
    return all_sessions


def build_splits(df: pd.DataFrame, cfg: SplitConfig, include_test: bool = True) -> Splits:
    """include_test=False leaves test roles as None so tuning code cannot reach them."""
    all_sessions = _check_config(df, cfg)
    owners, pool_val, pool_test = assign_subjects(df["subject"].unique().tolist(), cfg.n_owners)

    # Impostors: cfg.impostor_reps of every configured session; one shared set per role for all owners.
    val_imp = _rows(df, pool_val, all_sessions, cfg.impostor_reps)
    test_imp = _rows(df, pool_test, all_sessions, cfg.impostor_reps) if include_test else None

    by_owner = {
        o: OwnerSplit(
            owner=o,
            train=_rows(df, [o], cfg.train_sessions),
            val_owner=_rows(df, [o], cfg.val_sessions),
            test_owner=_rows(df, [o], cfg.test_sessions) if include_test else None,
            val_impostor=val_imp,
            test_impostor=test_imp,
        )
        for o in owners
    }
    for o, s in by_owner.items():
        for role in ROLES:
            frame = getattr(s, role)
            if frame is not None and frame.empty:
                raise ValueError(f"empty split role {role!r} for owner {o}")
    return Splits(tuple(owners), tuple(pool_val), tuple(pool_test), by_owner)


def split_table(splits: Splits) -> pd.DataFrame:
    """One row per used record: subject, sessionIndex, rep, role, owner ('*' = shared impostor set)."""
    first = splits.by_owner[splits.owners[0]]
    if first.test_owner is None:
        raise ValueError("split_table needs splits built with include_test=True")
    parts = []
    for o, s in splits.by_owner.items():
        for role in ("train", "val_owner", "test_owner"):
            parts.append(getattr(s, role)[list(META_COLUMNS)].assign(role=role, owner=o))
    for role in ("val_impostor", "test_impostor"):
        parts.append(getattr(first, role)[list(META_COLUMNS)].assign(role=role, owner=SHARED_OWNER))
    return pd.concat(parts, ignore_index=True)


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Model input: timing columns only, fixed order; identifiers never pass through."""
    return frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
