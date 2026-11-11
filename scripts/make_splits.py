"""Build the deterministic split and save its composition to reports/splits.csv."""

import sys

from keystroke.config import load_experiment_config, resolve, utf8_stdio
from keystroke.data import DataError, load_dataset
from keystroke.splits import SplitConfig, build_splits, split_table


def main() -> int:
    utf8_stdio()
    cfg = load_experiment_config()
    try:
        df = load_dataset(resolve(cfg["data"]["path"]), cfg["data"]["sha256"])
    except DataError as exc:
        print(exc, file=sys.stderr)
        return 1

    splits = build_splits(df, SplitConfig.from_dict(cfg["split"]))
    table = split_table(splits)
    out = resolve(cfg["paths"]["reports"]) / "splits.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")

    print(f"owners:    {', '.join(splits.owners)}")
    print(f"pool_val:  {len(splits.pool_val)} subjects  {', '.join(splits.pool_val)}")
    print(f"pool_test: {len(splits.pool_test)} subjects  {', '.join(splits.pool_test)}")
    print(table.groupby(["role", "owner"]).size().rename("n").to_string())
    print(f"wrote {out} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
