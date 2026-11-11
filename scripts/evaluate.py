"""Apply frozen models and thresholds to test; no parameter or threshold is chosen here.

Writes reports/metrics.csv and reports/figures/score_distribution_<owner>_<method>.png.
"""

import json
import sys

import numpy as np
import pandas as pd

from keystroke.config import load_experiment_config, resolve, utf8_stdio
from keystroke.data import DataError, load_dataset
from keystroke.frozen import OwnerScores, check_frozen, load_frozen, score_frozen
from keystroke.plotting import SERIES, TEXT, TEXT_MUTED, plt, style
from keystroke.splits import SplitConfig, build_splits
from keystroke.thresholds import far_frr

METHOD_LABELS = {"mean_abs_distance": "среднее абсолютное расстояние", "isolation_forest": "Isolation Forest"}
COLS = ["owner", "method", "params", "threshold", "n_owner_train", "n_owner_test", "n_impostor_test",
        "FAR", "FRR", "avg_error"]


def plot_scores(s: OwnerScores, owner: str, method: str, far: float, frr: float, path) -> None:
    lo = min(s.test_owner.min(), s.test_impostor.min())
    hi = max(s.test_owner.max(), s.test_impostor.max())
    bins = np.linspace(lo, hi, 50)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(s.test_impostor, bins=bins, density=True, color=SERIES[1], alpha=0.55,
            label=f"чужие (n={s.test_impostor.size})")
    ax.hist(s.test_owner, bins=bins, density=True, color=SERIES[0], alpha=0.55,
            label=f"владелец {owner} (n={s.test_owner.size})")
    ax.axvline(s.threshold, color=TEXT, linewidth=1.5, linestyle="--", label=f"порог t = {s.threshold:.4f}")
    ax.set_xlabel("Оценка отклонения (больше — сильнее отличие от профиля)")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Test: {METHOD_LABELS.get(method, method)}, профиль {owner}")
    ax.text(0.99, 0.80, f"FAR = {far:.3f}\nFRR = {frr:.3f}", transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TEXT_MUTED)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    utf8_stdio()
    cfg = load_experiment_config()
    try:
        frozen = load_frozen(resolve(cfg["paths"]["frozen"]))
        df = load_dataset(resolve(cfg["data"]["path"]), cfg["data"]["sha256"])
        splits = build_splits(df, SplitConfig.from_dict(cfg["split"]))
        check_frozen(frozen, cfg["data"]["sha256"], splits)
    except DataError as exc:
        print(exc, file=sys.stderr)
        return 1

    reports, figures = resolve(cfg["paths"]["reports"]), resolve(cfg["paths"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    style()
    # Fixed before looking at test: the first owner is the plotted profile.
    figure_owner = splits.owners[0]

    rows = []
    for method, block in frozen["methods"].items():
        for owner in splits.owners:
            split = splits.by_owner[owner]
            try:
                s = score_frozen(method, block["params"], block["owners"][owner], split)
            except DataError as exc:
                print(exc, file=sys.stderr)
                return 1
            far, frr = far_frr(s.test_owner, s.test_impostor, s.threshold)
            rows.append(
                {
                    "owner": owner,
                    "method": method,
                    "params": json.dumps(block["params"]),
                    "threshold": s.threshold,
                    "n_owner_train": len(split.train),
                    "n_owner_test": s.test_owner.size,
                    "n_impostor_test": s.test_impostor.size,
                    "FAR": far,
                    "FRR": frr,
                    "avg_error": (far + frr) / 2,
                }
            )
            if owner == figure_owner:
                plot_scores(s, owner, method, far, frr, figures / f"score_distribution_{owner}_{method}.png")

    per_owner = pd.DataFrame(rows)
    means = (
        per_owner.groupby("method", sort=False)
        .agg(
            params=("params", "first"),
            n_owner_train=("n_owner_train", "first"),
            n_owner_test=("n_owner_test", "first"),
            n_impostor_test=("n_impostor_test", "first"),
            FAR=("FAR", "mean"),
            FRR=("FRR", "mean"),
            avg_error=("avg_error", "mean"),
        )
        .reset_index()
        .assign(owner="MEAN")
    )
    out = pd.concat([per_owner, means], ignore_index=True)[COLS]
    out.to_csv(reports / "metrics.csv", index=False, lineterminator="\n")

    with pd.option_context("display.width", 140, "display.float_format", "{:.4f}".format):
        print(out.drop(columns="params").to_string(index=False))
    print(f"app_method (chosen on validation): {frozen['app_method']['name']}")
    print(f"wrote {reports / 'metrics.csv'} and score distributions for {figure_owner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
