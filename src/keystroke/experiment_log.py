import json
from pathlib import Path
from typing import Any

CONDITIONS = ("normal", "slow", "pause")


def read_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Return (records, skipped); a non-blank line is skipped unless it is a JSON object with a known condition and profile_created_utc."""
    if not path.is_file():
        return [], 0
    records, skipped = [], 0
    for line in path.read_bytes().decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except (ValueError, RecursionError):
            skipped += 1
            continue
        if not isinstance(rec, dict) or rec.get("condition") not in CONDITIONS or "profile_created_utc" not in rec:
            skipped += 1
            continue
        records.append(rec)
    return records, skipped


def counts_for_profile(records: list[dict[str, Any]], profile_created_utc: str) -> dict[str, int]:
    counts = dict.fromkeys(CONDITIONS, 0)
    for rec in records:
        if rec["profile_created_utc"] == profile_created_utc:
            counts[rec["condition"]] += 1
    return counts


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A torn last line must not swallow the next record.
    needs_newline = path.is_file() and path.stat().st_size > 0 and path.read_bytes()[-1:] != b"\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        if needs_newline:
            f.write("\n")
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
