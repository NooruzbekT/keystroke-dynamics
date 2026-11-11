"""Aggregate logs/own_experiment.jsonl into reports/own_experiment.csv (per condition, current profile only)."""

import sys

import pandas as pd

from keystroke.config import load_app_config, load_experiment_config, resolve, utf8_stdio
from keystroke.experiment_log import CONDITIONS, read_records


def summarize(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    rows = []
    for cond in CONDITIONS:
        part = df[df.condition == cond] if not df.empty else df
        n = len(part)
        rows.append(
            {
                "condition": cond,
                "n": n,
                "mean_duration_s": float(part.duration.mean()) if n else float("nan"),
                "mean_score": float(part.score.mean()) if n else float("nan"),
                "threshold": float(part.threshold.iloc[0]) if n else float("nan"),
                "accepted": int(part.accepted.astype(bool).sum()) if n else 0,
                "rejected": int((~part.accepted.astype(bool)).sum()) if n else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    utf8_stdio()
    app_cfg = load_app_config()
    log_path = resolve(app_cfg["paths"]["logs"]) / "own_experiment.jsonl"
    records, skipped = read_records(log_path)
    if skipped:
        print(f"skipped {skipped} unreadable log lines")
    if not records:
        print(f"no experiment records in {log_path}: run the experiment mode in the app first.", file=sys.stderr)
        return 1

    # A profile reset mid-experiment would mix two profiles; keep only the latest one.
    latest = max(r["profile_created_utc"] for r in records)
    current = [r for r in records if r["profile_created_utc"] == latest]
    if len(current) != len(records):
        print(f"skipped {len(records) - len(current)} records made with an older profile")

    table = summarize(current)
    out = resolve(load_experiment_config()["paths"]["reports"]) / "own_experiment.csv"
    table.to_csv(out, index=False, lineterminator="\n")
    print(table.to_string(index=False))
    expected = int(app_cfg["experiment_per_condition"])
    for row in table.itertuples():
        if row.n != expected:
            print(f"warning: condition {row.condition} has {row.n} attempts, expected {expected}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
