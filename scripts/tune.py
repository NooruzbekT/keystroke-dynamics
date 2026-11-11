"""Select thresholds, the IF config and the app method on validation; test rows are never built.

Writes reports/validation_metrics.csv, then config/frozen.json.
"""

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction

import pandas as pd

from keystroke.config import git_state, library_versions, load_experiment_config, resolve, utf8_stdio
from keystroke.data import FEATURE_COLUMNS, DataError, load_dataset
from keystroke.methods import IsolationForestScorer, MeanAbsoluteDistance
from keystroke.splits import SplitConfig, build_splits, feature_matrix
from keystroke.thresholds import ThresholdResult, select_threshold

MAD = MeanAbsoluteDistance.name
IF = IsolationForestScorer.name
COUNT_COLS = ["n_owner_train", "n_owner_val", "n_impostor_val"]


def if_grid(cfg: dict) -> list[dict]:
    grid = cfg["isolation_forest"]["grid"]
    keys = ("n_estimators", "max_samples", "max_features")
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def tune_owner(model, split) -> tuple[ThresholdResult, int]:
    model.fit(feature_matrix(split.train))
    own = model.score(feature_matrix(split.val_owner))
    imp = model.score(feature_matrix(split.val_impostor))
    return select_threshold(own, imp), len(split.train)


def exact_means(results: list[ThresholdResult]) -> tuple[Fraction, Fraction]:
    """Mean over owners of (FAR+FRR)/2 and of FAR as exact fractions, so ties are real ties."""
    k = len(results)
    return sum(r.exact_avg_error() for r in results) / k, sum(r.exact_far() for r in results) / k


def main() -> int:
    utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen.json")
    ap.add_argument("--allow-dirty", action="store_true", help="allow running without a clean committed revision")
    args = ap.parse_args()

    cfg = load_experiment_config()
    frozen_path = resolve(cfg["paths"]["frozen"])
    if frozen_path.exists() and not args.force:
        print(
            f"{frozen_path} already exists. Thresholds are frozen; re-tuning after looking at test "
            "results would leak test information. Use --force only if test was never evaluated.",
            file=sys.stderr,
        )
        return 1

    git = git_state()
    if (git["git_commit"] is None or git["git_dirty"]) and not args.allow_dirty:
        print(
            "Refusing to freeze thresholds: no commit or uncommitted changes in tracked files "
            f"(git_commit={git['git_commit']}, git_dirty={git['git_dirty']}). "
            "Commit the code first, or pass --allow-dirty for a dry run.",
            file=sys.stderr,
        )
        return 1

    try:
        df = load_dataset(resolve(cfg["data"]["path"]), cfg["data"]["sha256"])
    except DataError as exc:
        print(exc, file=sys.stderr)
        return 1

    seed = int(cfg["seed"])
    contamination = cfg["isolation_forest"]["contamination"]
    splits = build_splits(df, SplitConfig.from_dict(cfg["split"]), include_test=False)
    grid = if_grid(cfg)

    configs: list[tuple[str, int, dict]] = [(MAD, 0, {})] + [(IF, i, p) for i, p in enumerate(grid)]
    results: dict[tuple[str, int], dict[str, ThresholdResult]] = {(m, i): {} for m, i, _ in configs}
    rows = []
    for owner in splits.owners:
        s = splits.by_owner[owner]
        for method, cid, params in configs:
            model = (
                MeanAbsoluteDistance()
                if method == MAD
                else IsolationForestScorer(**params, random_state=seed, contamination=contamination)
            )
            r, n_train = tune_owner(model, s)
            results[(method, cid)][owner] = r
            rows.append(
                {
                    "method": method,
                    "config_id": cid,
                    "params": json.dumps(params),
                    "owner": owner,
                    "threshold": r.threshold,
                    "n_owner_train": n_train,
                    "n_owner_val": r.n_owner,
                    "n_impostor_val": r.n_impostor,
                    "FAR": r.far,
                    "FRR": r.frr,
                    "avg_error": r.avg_error,
                }
            )
        print(f"{owner}: done")

    means = {key: exact_means(list(by_owner.values())) for key, by_owner in results.items()}
    # One IF config for all owners: min mean val avg_error, then lower mean FAR, then grid order.
    best_if = min(range(len(grid)), key=lambda i: (*means[(IF, i)], i))
    best_if_params = grid[best_if]
    # App method by validation only; full tie keeps the simpler baseline.
    app_method = min([(MAD, 0), (IF, best_if)], key=lambda k: (*means[k], k[0] != MAD))[0]

    per_owner = pd.DataFrame(rows)
    mean_rows = (
        per_owner.groupby(["method", "config_id"], sort=False)
        .agg(
            params=("params", "first"),
            **{c: (c, "first") for c in COUNT_COLS},
            FAR=("FAR", "mean"),
            FRR=("FRR", "mean"),
            avg_error=("avg_error", "mean"),
        )
        .reset_index()
        .assign(owner="MEAN")
    )
    out = pd.concat([per_owner, mean_rows], ignore_index=True)
    out[COUNT_COLS] = out[COUNT_COLS].astype(int)
    out["selected"] = (out.method == MAD) | ((out.method == IF) & (out.config_id == best_if))
    cols = ["method", "config_id", "params", "owner", "threshold", *COUNT_COLS, "FAR", "FRR", "avg_error", "selected"]
    metrics_path = resolve(cfg["paths"]["reports"]) / "validation_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(metrics_path, index=False, lineterminator="\n")

    def method_block(key: tuple[str, int]) -> dict:
        by_owner = results[key]
        return {
            "owners": {
                o: {"threshold": r.threshold, "val_FAR": r.far, "val_FRR": r.frr, "val_avg_error": r.avg_error}
                for o, r in by_owner.items()
            },
            "val_mean": {
                "FAR": float(sum(r.far for r in by_owner.values()) / len(by_owner)),
                "FRR": float(sum(r.frr for r in by_owner.values()) / len(by_owner)),
                "avg_error": float(means[key][0]),
            },
        }

    if_params = {**best_if_params, "contamination": contamination, "random_state": seed}
    frozen = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "data_sha256": cfg["data"]["sha256"],
        **git,
        "versions": library_versions(),
        "split": {
            "owners": list(splits.owners),
            "pool_val": list(splits.pool_val),
            "pool_test": list(splits.pool_test),
            **{k: list(v) for k, v in cfg["split"].items() if k != "n_owners"},
        },
        "feature_columns": FEATURE_COLUMNS,
        "threshold_rule": "accept if score <= t; t minimises (FAR+FRR)/2 on validation over all validation "
        "scores plus min-1e-9; ties: lower FAR, then lower t",
        "methods": {
            MAD: {"params": {}, **method_block((MAD, 0))},
            IF: {
                "params": if_params,
                "config_id": best_if,
                "selection_rule": "one config for all owners: min mean validation (FAR+FRR)/2, "
                "then lower mean FAR, then first in grid order",
                **method_block((IF, best_if)),
            },
        },
        "app_method": {
            "name": app_method,
            "params": {} if app_method == MAD else if_params,
            "selection_rule": "lower mean validation (FAR+FRR)/2, then lower mean FAR, then the baseline",
        },
    }
    frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    summary = mean_rows[["method", "config_id", "params", "FAR", "FRR", "avg_error"]]
    print("\nValidation mean by config:")
    print(summary.to_string(index=False))
    print(f"\nselected IF config #{best_if}: {best_if_params}")
    print(f"app_method: {app_method}")
    print(f"wrote {metrics_path} and {frozen_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
