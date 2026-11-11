import numpy as np
import pytest

from keystroke.data import FEATURE_COLUMNS, DataError, read_csv, sha256_file, validate_schema, verify_sha256


def test_feature_columns_follow_cmu_order():
    assert len(FEATURE_COLUMNS) == 31
    assert FEATURE_COLUMNS[:4] == ["H.period", "DD.period.t", "UD.period.t", "H.t"]
    assert FEATURE_COLUMNS[-1] == "H.Return"
    assert sum(c.startswith("H.") for c in FEATURE_COLUMNS) == 11
    assert sum(c.startswith("DD.") for c in FEATURE_COLUMNS) == 10
    assert sum(c.startswith("UD.") for c in FEATURE_COLUMNS) == 10


def test_validate_schema_accepts_layout_and_rejects_deviations(cmu_like):
    df = cmu_like.sort_values(["subject", "sessionIndex", "rep"]).reset_index(drop=True)
    assert validate_schema(df) == FEATURE_COLUMNS

    cols = list(df.columns)
    cols[3], cols[4] = cols[4], cols[3]
    with pytest.raises(DataError, match="order"):
        validate_schema(df[cols])

    with_nan = df.copy()
    with_nan.iloc[0, 5] = np.nan
    with pytest.raises(DataError, match="missing"):
        validate_schema(with_nan)

    with pytest.raises(DataError, match="rows"):
        validate_schema(df.iloc[:-1])


def test_verify_sha256(tmp_path):
    p = tmp_path / "x.csv"
    p.write_bytes(b"abc")
    assert verify_sha256(p, sha256_file(p)) == sha256_file(p)
    with pytest.raises(DataError, match="mismatch"):
        verify_sha256(p, "0" * 64)
    with pytest.raises(DataError, match="not recorded"):
        verify_sha256(p, None)
    with pytest.raises(DataError, match="not found"):
        verify_sha256(tmp_path / "missing.csv", "0" * 64)


def test_read_csv_wraps_parse_errors(tmp_path):
    p = tmp_path / "page.csv"
    p.write_bytes(b"\xff\xfe<html>\x00\x00")
    with pytest.raises(DataError):
        read_csv(p)
