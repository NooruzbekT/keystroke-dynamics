"""Test-error analysis with frozen models: reports/error_cases.csv and reports/error_analysis.md.

Per-feature deviation is z - mu in the owner's train scaling; negative = shorter interval than the profile mean.
"""

import sys

import numpy as np
import pandas as pd

from keystroke.config import load_experiment_config, resolve, utf8_stdio
from keystroke.data import FEATURE_COLUMNS, META_COLUMNS, DataError, load_dataset
from keystroke.frozen import check_frozen, load_frozen, score_frozen
from keystroke.reporting import md_table
from keystroke.splits import SplitConfig, build_splits, feature_matrix

N_PER_TYPE = 3
TOP_K = 5
GROUPS = {"H": "H.", "DD": "DD.", "UD": "UD."}


def top_features(signed: np.ndarray, k: int = TOP_K) -> str:
    order = np.argsort(-np.abs(signed), kind="stable")[:k]
    return "; ".join(f"{FEATURE_COLUMNS[j]}:{signed[j]:+.2f}" for j in order)


def records_for(method: str, block: dict, splits) -> tuple[pd.DataFrame, np.ndarray]:
    frames, signed_all = [], []
    for owner in splits.owners:
        split = splits.by_owner[owner]
        s = score_frozen(method, block["params"], block["owners"][owner], split)
        for source, frame, scores in (
            ("owner", split.test_owner, s.test_owner),
            ("impostor", split.test_impostor, s.test_impostor),
        ):
            signed_all.append(s.model.signed_deviations(feature_matrix(frame)))
            accepted = scores <= s.threshold
            error = np.where(
                (source == "impostor") & accepted, "false_accept", np.where((source == "owner") & ~accepted, "false_reject", "")
            )
            frames.append(
                frame[list(META_COLUMNS)].assign(
                    method=method, owner=owner, source=source, score=scores, threshold=s.threshold,
                    margin=scores - s.threshold, accepted=accepted, error=error,
                )
            )
    return pd.concat(frames, ignore_index=True), np.vstack(signed_all)


def select_cases(rec: pd.DataFrame) -> pd.DataFrame:
    """Up to N_PER_TYPE each: false accepts with most negative margin, false rejects with most positive margin, then remaining records closest to the threshold."""
    fa = rec[rec.error == "false_accept"].sort_values(["margin", "owner", "subject", "sessionIndex", "rep"]).head(N_PER_TYPE)
    fr = rec[rec.error == "false_reject"].sort_values(["margin", "owner", "sessionIndex", "rep"], ascending=[False, True, True, True]).head(N_PER_TYPE)
    rest = rec.drop(index=fa.index.union(fr.index))
    border = rest.assign(abs_margin=rest.margin.abs()).sort_values(["abs_margin", "owner", "subject", "sessionIndex", "rep"]).head(N_PER_TYPE)
    return pd.concat(
        [fa.assign(case="false_accept"), fr.assign(case="false_reject"), border.drop(columns="abs_margin").assign(case="borderline")]
    )


def group_means(signed: np.ndarray) -> dict[str, float]:
    return {g: float(signed[:, [j for j, c in enumerate(FEATURE_COLUMNS) if c.startswith(p)]].mean()) for g, p in GROUPS.items()}


def main() -> int:
    utf8_stdio()
    cfg = load_experiment_config()
    try:
        frozen = load_frozen(resolve(cfg["paths"]["frozen"]))
        df = load_dataset(resolve(cfg["data"]["path"]), cfg["data"]["sha256"])
        splits = build_splits(df, SplitConfig.from_dict(cfg["split"]))
        check_frozen(frozen, cfg["data"]["sha256"], splits)
        per_method = {m: records_for(m, b, splits) for m, b in frozen["methods"].items()}
    except DataError as exc:
        print(exc, file=sys.stderr)
        return 1

    cases, lines = [], ["# Анализ ошибок на test", "", "Сгенерировано `python scripts/error_analysis.py` по замороженным моделям и порогам.", ""]
    lines += [
        "Отклонение признака — `z − mu` в масштабе train владельца; знак минус — интервал короче, чем в профиле.",
        "",
    ]
    for method, (rec, signed) in per_method.items():
        chosen = select_cases(rec)
        chosen = chosen.assign(top_features=[top_features(signed[i]) for i in chosen.index])
        cases.append(chosen)

        own = rec.source == "owner"
        frr_session = (
            rec[own].assign(rejected=~rec[own].accepted)
            .groupby(["owner", "sessionIndex"]).rejected.mean().unstack("sessionIndex")
        )
        frr_session.columns = [f"FRR сессия {c}" for c in frr_session.columns]

        fa = rec[rec.error == "false_accept"]
        fa_conc = fa.groupby("subject").size().sort_values(ascending=False)
        top_share = float(fa_conc.head(3).sum() / max(len(fa), 1))

        drift_rows = []
        for owner in splits.owners:
            m_own = (rec.owner == owner) & own
            for label, mask in (("принят", m_own & rec.accepted), ("отклонён", m_own & ~rec.accepted)):
                if mask.any():
                    drift_rows.append({"владелец": owner, "вводы владельца": label, "n": int(mask.sum()), **group_means(signed[mask.to_numpy()])})

        freq = {}
        for err in ("false_accept", "false_reject"):
            idx = np.flatnonzero((rec.error == err).to_numpy())
            counts = pd.Series(0, index=FEATURE_COLUMNS)
            for i in idx:
                for j in np.argsort(-np.abs(signed[i]), kind="stable")[:TOP_K]:
                    counts.iloc[j] += 1
            freq[err] = counts.sort_values(ascending=False).head(8)

        lines += [
            f"## {method}",
            "",
            f"Ложных принятий: {len(fa)} из {int((~own).sum())}; ложных отказов: {int((rec.error == 'false_reject').sum())} из {int(own.sum())}.",
            "",
            "### FRR по сессиям test",
            "",
            md_table(frr_session.reset_index().rename(columns={"owner": "владелец"}), "{:.2f}"),
            "",
            "### Средние отклонения z − mu по группам признаков для вводов владельца на test",
            "",
            md_table(pd.DataFrame(drift_rows), "{:+.2f}"),
            "",
            "### Концентрация ложных принятий",
            "",
            f"Разных чужих участников среди ложных принятий: {fa_conc.size} из {len(splits.pool_test)}; "
            f"на трёх самых частых приходится {top_share:.0%} ложных принятий.",
            "",
            md_table(fa_conc.head(5).rename_axis("участник").reset_index(name="ложных принятий")),
            "",
            "### Признаки, чаще всего входящие в топ-5 отклонений",
            "",
            md_table(pd.DataFrame({
                "ложные принятия": [f"{k} ({v})" for k, v in freq["false_accept"].items()],
                "ложные отказы": [f"{k} ({v})" for k, v in freq["false_reject"].items()],
            })),
            "",
        ]

    out = pd.concat(cases, ignore_index=True)
    cols = ["method", "case", "owner", "source", "subject", "sessionIndex", "rep", "score", "threshold", "margin", "accepted", "top_features"]
    reports = resolve(cfg["paths"]["reports"])
    out[cols].to_csv(reports / "error_cases.csv", index=False, lineterminator="\n")
    (reports / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(out[["method", "case", "owner", "subject", "sessionIndex", "rep", "margin", "top_features"]].to_string(index=False))
    print(f"wrote {reports / 'error_cases.csv'} ({len(out)} cases) and {reports / 'error_analysis.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
