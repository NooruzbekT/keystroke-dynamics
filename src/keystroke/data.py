import hashlib
from pathlib import Path

import pandas as pd

META_COLUMNS = ("subject", "sessionIndex", "rep")

# Key sequence of the CMU phrase ".tie5Roanl" + Return.
CMU_KEYS = ("period", "t", "i", "e", "five", "Shift.r", "o", "a", "n", "l", "Return")

EXPECTED_ROWS = 20_400
EXPECTED_SUBJECTS = 51
SESSIONS = tuple(range(1, 9))
REPS = tuple(range(1, 51))


class DataError(RuntimeError):
    pass


def build_feature_columns(keys: tuple[str, ...] = CMU_KEYS) -> list[str]:
    cols: list[str] = []
    for prev, nxt in zip(keys, keys[1:]):
        cols += [f"H.{prev}", f"DD.{prev}.{nxt}", f"UD.{prev}.{nxt}"]
    cols.append(f"H.{keys[-1]}")
    return cols


FEATURE_COLUMNS: list[str] = build_feature_columns()


def key_pairs(keys: tuple[str, ...] = CMU_KEYS) -> list[tuple[str, str]]:
    return list(zip(keys, keys[1:]))


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise DataError(f"Dataset not found: {path}. Run `python scripts/download_data.py` first.")


def read_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    try:
        return pd.read_csv(path, dtype={"subject": str})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as exc:
        raise DataError(f"{path} is not a readable CSV: {exc}") from exc


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Raise DataError listing every deviation from the CMU layout; return feature columns in file order."""
    errors: list[str] = []

    cols = list(df.columns)
    if tuple(cols[:3]) != META_COLUMNS:
        errors.append(f"first columns must be {META_COLUMNS}, got {tuple(cols[:3])}")
    features = [c for c in cols if c not in META_COLUMNS]
    if features != FEATURE_COLUMNS:
        errors.append("feature columns differ from the expected CMU order")

    if len(df) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
    if df.isna().any().any():
        errors.append(f"missing values: {int(df.isna().sum().sum())}")

    if set(META_COLUMNS) <= set(cols):
        n_subjects = df["subject"].nunique()
        if n_subjects != EXPECTED_SUBJECTS:
            errors.append(f"expected {EXPECTED_SUBJECTS} subjects, got {n_subjects}")
        if not df["subject"].str.fullmatch(r"s\d{3}").all():
            errors.append("subject ids must match sNNN")
        if set(df["sessionIndex"].unique()) != set(SESSIONS):
            errors.append("sessionIndex must cover 1..8")
        if set(df["rep"].unique()) != set(REPS):
            errors.append("rep must cover 1..50")
        cell = df.groupby(["subject", "sessionIndex"]).size()
        if not (cell == len(REPS)).all():
            errors.append("each (subject, session) must have exactly 50 reps")
        if df.duplicated(list(META_COLUMNS)).any():
            errors.append("duplicate (subject, sessionIndex, rep) rows")

    non_numeric = [c for c in features if not pd.api.types.is_float_dtype(df[c])]
    if non_numeric:
        errors.append(f"non-float feature columns: {non_numeric}")

    if errors:
        raise DataError("CMU schema check failed:\n- " + "\n- ".join(errors))
    return features


def verify_sha256(path: Path, expected: str | None) -> str:
    require_file(path)
    if not expected:
        raise DataError(
            "data.sha256 is not recorded in config/experiment.yaml. "
            "Run `python scripts/download_data.py` first."
        )
    actual = sha256_file(path)
    if actual != expected:
        raise DataError(
            f"SHA-256 mismatch for {path}:\n  expected {expected}\n  actual   {actual}\n"
            "The file differs from the one the experiment was recorded with."
        )
    return actual


def load_dataset(path: Path, expected_sha256: str | None) -> pd.DataFrame:
    verify_sha256(path, expected_sha256)
    df = read_csv(path)
    validate_schema(df)
    return df
