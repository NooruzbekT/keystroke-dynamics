"""EDA of the CMU benchmark: reports/eda_summary.md and figures in reports/figures/."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from keystroke.config import load_experiment_config, resolve, utf8_stdio
from keystroke.data import FEATURE_COLUMNS, META_COLUMNS, DataError, key_pairs, load_dataset
from keystroke.plotting import SERIES, SURFACE, TEXT, TEXT_MUTED, plt, style
from keystroke.reporting import md_table
from keystroke.splits import assign_subjects

UD_TOLERANCE = 1e-3


def ud_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in key_pairs():
        diff = (df[f"UD.{a}.{b}"] - (df[f"DD.{a}.{b}"] - df[f"H.{a}"])).abs()
        rows.append(
            {
                "pair": f"{a}→{b}",
                "match_share": float((diff <= UD_TOLERANCE).mean()),
                "max_abs_diff": float(diff.max()),
            }
        )
    return pd.DataFrame(rows)


def plot_hold_times(df: pd.DataFrame, path: Path) -> None:
    h_cols = [c for c in FEATURE_COLUMNS if c.startswith("H.")]
    values = df[h_cols].to_numpy().ravel()
    cap = float(np.quantile(values, 0.995))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.4]})
    counts, _, _ = ax1.hist(values[values <= cap], bins=80, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax1.set_ylim(0, counts.max() * 1.25)
    ax1.set_title("Распределение всех H (удержание)")
    ax1.set_xlabel("Время удержания, с")
    ax1.set_ylabel("Число нажатий")
    ax1.text(
        0.98, 0.95, f"показано ≤ 99.5-го перцентиля ({cap:.3f} с)\nn = {values.size:,}".replace(",", " "),
        transform=ax1.transAxes, ha="right", va="top", fontsize=8, color=TEXT_MUTED,
    )

    ax2.boxplot(
        [df[c] for c in h_cols],
        tick_labels=[c.removeprefix("H.") for c in h_cols],
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]},
        medianprops={"color": TEXT, "linewidth": 1.5},
        whiskerprops={"color": SERIES[0]},
        capprops={"color": SERIES[0]},
    )
    ax2.set_title("Удержание по клавишам (без выбросов)")
    ax2.set_ylabel("Время удержания, с")
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_two_subjects(df: pd.DataFrame, subjects: list[str], sessions: list[int], path: Path) -> None:
    x = np.arange(len(FEATURE_COLUMNS))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for color, subj in zip(SERIES, subjects):
        part = df[(df.subject == subj) & df.sessionIndex.isin(sessions)][FEATURE_COLUMNS]
        mean, std = part.mean().to_numpy(), part.std(ddof=0).to_numpy()
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=4, label=f"{subj} (n={len(part)})")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    ax.set_xticks(x, FEATURE_COLUMNS, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Время, с")
    ax.set_title(
        f"Средний ритм двух участников (сессии {sessions[0]}–{sessions[-1]}), полоса — ±1 ст. откл."
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_negative_ud(neg: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(neg))
    ax.barh(y, neg.to_numpy() * 100, color=SERIES[0], height=0.6)
    ax.set_yticks(y, neg.index)
    ax.invert_yaxis()
    ax.set_xlabel("Доля отрицательных значений, %")
    ax.set_title("Перекрытие нажатий: доля UD < 0 по парам клавиш")
    for yi, v in zip(y, neg.to_numpy() * 100):
        ax.text(v + 0.5, yi, f"{v:.1f}", va="center", fontsize=8, color=TEXT_MUTED)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    utf8_stdio()
    cfg = load_experiment_config()
    try:
        df = load_dataset(resolve(cfg["data"]["path"]), cfg["data"]["sha256"])
    except DataError as exc:
        print(exc, file=sys.stderr)
        return 1

    reports, figures = resolve(cfg["paths"]["reports"]), resolve(cfg["paths"]["figures"])
    figures.mkdir(parents=True, exist_ok=True)
    style()

    subjects = sorted(df["subject"].unique())
    owners, _, _ = assign_subjects(subjects, int(cfg["split"]["n_owners"]))
    per_subject = df.groupby("subject").size()
    per_cell = df.groupby(["subject", "sessionIndex"]).size()
    dtypes = df.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="columns")

    ud_cols = [c for c in FEATURE_COLUMNS if c.startswith("UD.")]
    neg = (df[ud_cols] < 0).mean()
    consistency = ud_consistency(df)
    overall_match = float(consistency["match_share"].mean())

    groups = []
    for prefix, name in (("H.", "H — удержание"), ("DD.", "DD — keydown→keydown"), ("UD.", "UD — keyup→keydown")):
        v = df[[c for c in FEATURE_COLUMNS if c.startswith(prefix)]].to_numpy().ravel()
        groups.append(
            {
                "группа": name,
                "признаков": sum(c.startswith(prefix) for c in FEATURE_COLUMNS),
                "медиана, с": float(np.median(v)),
                "Q1, с": float(np.quantile(v, 0.25)),
                "Q3, с": float(np.quantile(v, 0.75)),
                "min, с": float(v.min()),
                "max, с": float(v.max()),
            }
        )

    feature_stats = (
        df[FEATURE_COLUMNS].agg(["mean", "std", "min", "median", "max"]).T.rename_axis("признак").reset_index()
    )

    two = owners[:2]
    train_sessions = cfg["split"]["train_sessions"]
    plot_hold_times(df, figures / "hold_time_distribution.png")
    plot_two_subjects(df, two, train_sessions, figures / "two_subjects_rhythm.png")
    plot_negative_ud(neg, figures / "ud_negative_share.png")

    lines = [
        "# EDA: CMU Keystroke Dynamics Benchmark",
        "",
        "Сгенерировано `python scripts/explore_data.py`. Файл проверен по SHA-256 из `config/experiment.yaml`.",
        "",
        "## Размер и типы",
        "",
        f"- Строк: **{len(df)}**, столбцов: **{df.shape[1]}** (служебных {len(META_COLUMNS)}, временных {len(FEATURE_COLUMNS)}).",
        f"- Пропусков: **{int(df.isna().sum().sum())}**.",
        f"- Дубликатов (subject, sessionIndex, rep): **{int(df.duplicated(list(META_COLUMNS)).sum())}**.",
        "",
        md_table(dtypes),
        "",
        "## Записи по участникам и сессиям",
        "",
        f"- Участников: **{len(subjects)}**; записей на участника: min {per_subject.min()}, max {per_subject.max()}.",
        f"- Сессий: {', '.join(map(str, sorted(df.sessionIndex.unique().tolist())))}; записей на (участник, сессия): min {per_cell.min()}, max {per_cell.max()}.",
        f"- Повторений в сессии: {df.rep.min()}–{df.rep.max()}.",
        f"- Первые {len(owners)} идентификаторов после сортировки (владельцы профилей): {', '.join(owners)}.",
        "",
        "## Группы признаков",
        "",
        md_table(pd.DataFrame(groups)),
        "",
        "## Отрицательные UD (перекрытие нажатий)",
        "",
        f"Доля отрицательных значений среди всех UD: **{float((df[ud_cols] < 0).to_numpy().mean()):.2%}**; "
        f"строк хотя бы с одним UD < 0: **{float((df[ud_cols] < 0).any(axis=1).mean()):.2%}**. "
        "Значения не удаляются и не обрезаются.",
        "",
        md_table(neg.rename_axis("признак").reset_index(name="доля UD < 0")),
        "",
        "## Санити-проверка UD ≈ DD − H",
        "",
        f"Для каждой пары (a, b) проверяется |UD.a.b − (DD.a.b − H.a)| ≤ {UD_TOLERANCE}. "
        f"Средняя доля совпадений по парам: **{overall_match:.2%}**.",
        "",
        md_table(consistency),
        "",
        "## Статистики по признакам (секунды)",
        "",
        md_table(feature_stats),
        "",
        "## Графики",
        "",
        "- `figures/hold_time_distribution.png` — распределение всех H и удержание по клавишам.",
        f"- `figures/two_subjects_rhythm.png` — средний ритм {two[0]} и {two[1]} по сессиям "
        f"{train_sessions[0]}–{train_sessions[-1]} (только train-часть владельцев).",
        "- `figures/ud_negative_share.png` — доля UD < 0 по парам клавиш.",
        "",
    ]
    (reports / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"rows={len(df)} subjects={len(subjects)} owners={owners}")
    print(f"negative UD share={float((df[ud_cols] < 0).to_numpy().mean()):.4f}")
    print(f"UD≈DD−H match share={overall_match:.4f}")
    print(f"wrote {reports / 'eda_summary.md'} and 3 figures in {figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
