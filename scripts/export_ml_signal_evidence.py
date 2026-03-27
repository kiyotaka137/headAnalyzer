from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MONTH_RU = {
    1: "Янв",
    2: "Фев",
    3: "Мар",
    4: "Апр",
    5: "Май",
    6: "Июн",
    7: "Июл",
    8: "Авг",
    9: "Сен",
    10: "Окт",
    11: "Ноя",
    12: "Дек",
}


def _fill_text(s: pd.Series, default: str = "не указано") -> pd.Series:
    return s.astype("string").fillna(default).replace("", default)


def _collapse_rare(s: pd.Series, *, min_rows: int, other_label: str = "Прочие") -> pd.Series:
    filled = _fill_text(s)
    counts = filled.value_counts(dropna=False)
    keep = set(counts[counts >= min_rows].index.tolist())
    return filled.where(filled.isin(keep), other_label)


def _zscore(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros(len(x), dtype=float)
    x = np.where(np.isfinite(x), x, mean)
    return (x - mean) / std


def _prepare_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="all_pairs")
    df = df.copy()
    df["Срок до отёла, дней"] = pd.to_numeric(df["days_from_conception_to_calving"], errors="coerce")
    df["DIM/возраст на зачатии"] = pd.to_numeric(df["dim_or_age_at_conception"], errors="coerce")
    df = df[df["Срок до отёла, дней"].notna()].copy()

    df["Подразделение (крупные)"] = _collapse_rare(df["conception_subdivision"], min_rows=500)
    df["Сезон зачатия"] = _fill_text(df["conception_season"])
    df["Месяц зачатия"] = pd.to_numeric(df["conception_month"], errors="coerce").map(MONTH_RU).fillna("не указано")
    df["Период лактации"] = _fill_text(df["lactation_group"])
    df["Корова / нетель"] = np.where(df["is_heifer_pregnancy"].fillna(False), "Нетель", "Корова")
    df["Тип семени"] = np.where(_fill_text(df["semen_type"]).str.lower().eq("sex"), "Сексированное", "Традиционное")
    df["Порода быка"] = _collapse_rare(df["bull_breed"], min_rows=500)
    df["Многоплодие"] = np.where(pd.to_numeric(df["is_multiple_birth"], errors="coerce").fillna(0).gt(0), "Да", "Нет")
    df["Хвост срока"] = np.where(
        df["Срок до отёла, дней"] < 270,
        "Короткий",
        np.where(df["Срок до отёла, дней"] > 320, "Длинный", "Основной"),
    )
    return df


def _effect_comment(r2_pct: float) -> str:
    if r2_pct < 0.05:
        return "Почти не влияет"
    if r2_pct < 0.20:
        return "Очень слабая связь"
    if r2_pct < 1.00:
        return "Слабая связь"
    return "Есть сигнал, но надо проверять на редкие группы"


def _categorical_effect(df: pd.DataFrame, feature: str) -> tuple[dict, pd.DataFrame]:
    work = df[[feature, "Срок до отёла, дней"]].copy()
    work[feature] = _fill_text(work[feature])
    work = work[work["Срок до отёла, дней"].notna()].copy()

    y = work["Срок до отёла, дней"].to_numpy(dtype=float)
    y_mean = float(y.mean())
    sst = float(np.sum((y - y_mean) ** 2))

    grouped = work.groupby(feature, dropna=False)["Срок до отёла, дней"]
    means = grouped.mean()
    counts = grouped.size()
    ssb = float(np.sum(((means - y_mean) ** 2) * counts))
    r2_pct = 100.0 * ssb / sst if sst > 0 else 0.0

    stats = (
        grouped.agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                feature: "Категория",
                "count": "Строк",
                "mean": "Средний срок, дней",
                "median": "Медиана, дней",
                "min": "Минимум, дней",
                "max": "Максимум, дней",
            }
        )
    )
    stats["Признак"] = feature
    stats["Доля строк, %"] = stats["Строк"] / max(len(work), 1) * 100.0
    stats = stats.sort_values("Строк", ascending=False, kind="mergesort").reset_index(drop=True)

    summary = {
        "Признак": feature,
        "Тип": "Категориальный",
        "Категорий": int(stats["Категория"].nunique()),
        "Строк": int(len(work)),
        "Объяснённый разброс, %": float(r2_pct),
        "Макс. разница средних, дней": float(means.max() - means.min()),
        "Комментарий": _effect_comment(r2_pct),
    }
    return summary, stats


def _numeric_effect(df: pd.DataFrame, feature: str) -> tuple[dict, pd.DataFrame]:
    work = df[[feature, "Срок до отёла, дней"]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work[work[feature].notna() & work["Срок до отёла, дней"].notna()].copy()

    corr = float(work[feature].corr(work["Срок до отёла, дней"]))
    r2_pct = float((corr ** 2) * 100.0)
    stats = pd.DataFrame(
        [
            {
                "Признак": feature,
                "Строк": int(len(work)),
                "Среднее значение признака": float(work[feature].mean()),
                "Корреляция с сроком": float(corr),
                "Объяснённый разброс, %": float(r2_pct),
            }
        ]
    )
    summary = {
        "Признак": feature,
        "Тип": "Числовой",
        "Категорий": np.nan,
        "Строк": int(len(work)),
        "Объяснённый разброс, %": float(r2_pct),
        "Макс. разница средних, дней": np.nan,
        "Комментарий": _effect_comment(r2_pct),
    }
    return summary, stats


def _feature_signal_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict] = []
    detail_frames: list[pd.DataFrame] = []

    categorical_features = [
        "Подразделение (крупные)",
        "Месяц зачатия",
        "Сезон зачатия",
        "Период лактации",
        "Корова / нетель",
        "Тип семени",
        "Порода быка",
        "Многоплодие",
    ]

    for feature in categorical_features:
        summary, stats = _categorical_effect(df, feature)
        summaries.append(summary)
        detail_frames.append(stats)

    summary, stats = _numeric_effect(df, "DIM/возраст на зачатии")
    summaries.append(summary)
    detail_frames.append(stats)

    summary_df = pd.DataFrame(summaries).sort_values("Объяснённый разброс, %", ascending=False, kind="mergesort").reset_index(drop=True)
    detail_df = pd.concat(detail_frames, ignore_index=True, sort=False)
    return summary_df, detail_df


def _build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    blocks: list[np.ndarray] = []
    categorical = [
        "Подразделение (крупные)",
        "Месяц зачатия",
        "Сезон зачатия",
        "Период лактации",
        "Корова / нетель",
        "Тип семени",
        "Порода быка",
        "Многоплодие",
    ]
    for feature in categorical:
        dummies = pd.get_dummies(_fill_text(df[feature]), prefix=feature)
        if dummies.shape[1] == 0:
            continue
        blocks.append(dummies.to_numpy(dtype=float))
    blocks.append(_zscore(df["DIM/возраст на зачатии"]).reshape(-1, 1))
    return np.hstack(blocks)


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(float)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * s[:2]
    explained = (s[:2] ** 2) / np.sum(s ** 2) if np.sum(s ** 2) > 0 else np.zeros(2)
    return coords, explained


def _plot_effect_sizes(summary_df: pd.DataFrame, out_path: Path) -> None:
    data = summary_df.copy()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(data["Признак"], data["Объяснённый разброс, %"], color="#4C78A8")
    ax.set_xlabel("Сколько % разброса объясняет признак")
    ax.set_ylabel("")
    ax.set_title("Сила признаков для срока от зачатия до отёла")
    for i, v in enumerate(data["Объяснённый разброс, %"]):
        ax.text(v + 0.01, i, f"{v:.3f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_categorical_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    feature: str,
    *,
    category_order: list[str] | None = None,
    max_points: int = 5000,
) -> None:
    work = df[[feature, "Срок до отёла, дней"]].copy()
    work[feature] = _fill_text(work[feature])
    if category_order is None:
        category_order = work[feature].value_counts().index.tolist()
    work = work[work[feature].isin(category_order)].copy()
    if len(work) > max_points:
        work = work.sample(n=max_points, random_state=42)

    pos = {cat: i for i, cat in enumerate(category_order)}
    x = work[feature].map(pos).astype(float).to_numpy()
    jitter = np.random.default_rng(42).uniform(-0.18, 0.18, size=len(work))
    ax.scatter(x + jitter, work["Срок до отёла, дней"], s=8, alpha=0.18, color="#4C78A8", edgecolors="none")

    means = work.groupby(feature)["Срок до отёла, дней"].mean().reindex(category_order)
    ax.scatter(range(len(category_order)), means, color="#D62728", s=28, zorder=3)
    ax.axhline(df["Срок до отёла, дней"].mean(), color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(category_order)))
    ax.set_xticklabels(category_order, rotation=35, ha="right")
    ax.set_title(feature)
    ax.set_ylabel("Дней")


def _plot_numeric_scatter(ax: plt.Axes, df: pd.DataFrame, feature: str, *, max_points: int = 6000) -> None:
    work = df[[feature, "Срок до отёла, дней"]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work[work[feature].notna()].copy()
    if len(work) > max_points:
        work = work.sample(n=max_points, random_state=42)
    ax.scatter(work[feature], work["Срок до отёла, дней"], s=8, alpha=0.18, color="#4C78A8", edgecolors="none")
    if len(work) >= 2:
        x = work[feature].to_numpy(float)
        y = work["Срок до отёла, дней"].to_numpy(float)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ys = coef[0] * xs + coef[1]
        ax.plot(xs, ys, color="#D62728", linewidth=2)
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("Дней")


def _plot_strip_panels(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    features = [
        ("Месяц зачатия", list(MONTH_RU.values())),
        ("Период лактации", ["0", "1", "2", "3", "4", "5+"]),
        ("Подразделение (крупные)", df["Подразделение (крупные)"].value_counts().index.tolist()),
        ("Корова / нетель", ["Корова", "Нетель"]),
        ("Тип семени", ["Традиционное", "Сексированное"]),
        ("Порода быка", df["Порода быка"].value_counts().index.tolist()),
    ]
    for ax, (feature, order) in zip(axes.flatten(), features):
        _plot_categorical_scatter(ax, df, feature, category_order=[x for x in order if x in set(df[feature].astype(str))])
    fig.suptitle("Точечные графики: признак vs срок до отёла", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_pca_overlap(df: pd.DataFrame, out_path: Path) -> None:
    work = df.copy()
    if len(work) > 8000:
        work = work.sample(n=8000, random_state=42)
    x = _build_feature_matrix(work)
    coords, explained = _pca_2d(x)

    colors = {"Короткий": "#D62728", "Основной": "#4C78A8", "Длинный": "#F2A541"}
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for label in ["Короткий", "Основной", "Длинный"]:
        mask = work["Хвост срока"].eq(label).to_numpy()
        if mask.any():
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10,
                alpha=0.25,
                color=colors[label],
                edgecolors="none",
                label=label,
            )
    ax.set_title("PCA по признакам: классы срока сильно перекрываются")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% дисперсии признаков)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% дисперсии признаков)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_numeric_signal(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_numeric_scatter(ax, df, "DIM/возраст на зачатии")
    ax.set_title("DIM / возраст на зачатии почти не объясняет срок до отёла")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_conclusion(summary_df: pd.DataFrame) -> pd.DataFrame:
    top = summary_df.sort_values("Объяснённый разброс, %", ascending=False).reset_index(drop=True)
    best = top.iloc[0]
    return pd.DataFrame(
        [
            {
                "Вывод": "Самый сильный признак из устойчивых объясняет только малую долю разброса срока до отёла.",
                "Значение": f"{best['Признак']}: {best['Объяснённый разброс, %']:.3f}%",
            },
            {
                "Вывод": "Для большинства признаков разница средних сроков держится около 0–2 дней.",
                "Значение": "Это слишком мало для отдельной ML-модели по сроку отёла.",
            },
            {
                "Вывод": "Основная масса случаев сосредоточена в узком диапазоне 270–289 дней.",
                "Значение": "Из-за этого признаки почти не разделяют выборку на чёткие кластеры.",
            },
            {
                "Вывод": "Следовательно, ML здесь даст мало пользы и высокий риск переобучения на редких группах.",
                "Значение": "Сильнее влиять нужно на качество as-of, переходы групп и фактические бизнес-правила.",
            },
        ]
    )


def build_evidence(
    *,
    input_path: str | Path,
    output_xlsx: str | Path,
    plot_prefix: str | Path,
) -> dict[str, Path]:
    input_path = Path(input_path)
    output_xlsx = Path(output_xlsx)
    plot_prefix = Path(plot_prefix)
    plot_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = _prepare_frame(input_path)
    summary_df, detail_df = _feature_signal_tables(df)
    conclusion_df = _build_conclusion(summary_df)

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        conclusion_df.to_excel(writer, sheet_name="Вывод", index=False)
        summary_df.to_excel(writer, sheet_name="Сила признаков", index=False)
        detail_df.to_excel(writer, sheet_name="Детализация", index=False)

    effect_png = plot_prefix.with_name(plot_prefix.name + "_1_сила_признаков.png")
    strip_png = plot_prefix.with_name(plot_prefix.name + "_2_точечные_графики.png")
    pca_png = plot_prefix.with_name(plot_prefix.name + "_3_pca_перекрытие.png")
    num_png = plot_prefix.with_name(plot_prefix.name + "_4_dim_возраст.png")

    _plot_effect_sizes(summary_df, effect_png)
    _plot_strip_panels(df, strip_png)
    _plot_pca_overlap(df, pca_png)
    _plot_numeric_signal(df, num_png)

    return {
        "excel": output_xlsx,
        "effect_png": effect_png,
        "strip_png": strip_png,
        "pca_png": pca_png,
        "numeric_png": num_png,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="exports/conception_to_calving_pairs_enalb.xlsx",
        help="Path to conception-to-calving pairs Excel",
    )
    parser.add_argument(
        "--output",
        default="exports/доказательство_слабого_сигнала.xlsx",
        help="Path to Excel summary",
    )
    parser.add_argument(
        "--plot-prefix",
        default="exports/доказательство_слабого_сигнала",
        help="Prefix for output PNGs",
    )
    args = parser.parse_args()

    outputs = build_evidence(
        input_path=args.input,
        output_xlsx=args.output,
        plot_prefix=args.plot_prefix,
    )
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
