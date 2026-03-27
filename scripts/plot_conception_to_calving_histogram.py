from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _load_pairs(path: str | Path, subdivision: str | None = None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="all_pairs")
    df = df.copy()
    if subdivision:
        df = df[df["conception_subdivision"].astype("string") == subdivision].copy()
    df["days_from_conception_to_calving"] = pd.to_numeric(df["days_from_conception_to_calving"], errors="coerce")
    df = df[df["days_from_conception_to_calving"].notna()].copy()
    df["days_from_conception_to_calving"] = df["days_from_conception_to_calving"].astype(int)
    df["is_heifer_pregnancy"] = df["is_heifer_pregnancy"].fillna(False).astype(bool)
    return df


def _build_counts(df: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"day": range(int(df["days_from_conception_to_calving"].min()), int(df["days_from_conception_to_calving"].max()) + 1)})
    all_counts = (
        df.groupby("days_from_conception_to_calving")
        .size()
        .rename("count_all")
        .reset_index()
        .rename(columns={"days_from_conception_to_calving": "day"})
    )
    cow_counts = (
        df.loc[~df["is_heifer_pregnancy"]]
        .groupby("days_from_conception_to_calving")
        .size()
        .rename("count_cows")
        .reset_index()
        .rename(columns={"days_from_conception_to_calving": "day"})
    )
    heifer_counts = (
        df.loc[df["is_heifer_pregnancy"]]
        .groupby("days_from_conception_to_calving")
        .size()
        .rename("count_heifers")
        .reset_index()
        .rename(columns={"days_from_conception_to_calving": "day"})
    )
    out = base.merge(all_counts, on="day", how="left").merge(cow_counts, on="day", how="left").merge(heifer_counts, on="day", how="left")
    for col in ("count_all", "count_cows", "count_heifers"):
        out[col] = out[col].fillna(0).astype(int)
    return out


def _plot_counts(counts: pd.DataFrame, *, out_path: Path, title: str, highlight_day: int | None) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(counts["day"], counts["count_all"], color="#4C78A8", width=0.85, label="Все")
    ax.plot(counts["day"], counts["count_cows"], color="#F58518", linewidth=1.8, label="Коровы")
    ax.plot(counts["day"], counts["count_heifers"], color="#54A24B", linewidth=1.8, label="Нетели")

    if highlight_day is not None:
        row = counts[counts["day"] == highlight_day]
        if not row.empty:
            y = int(row["count_all"].iloc[0])
            ax.axvline(highlight_day, color="#B22222", linestyle="--", linewidth=1.2)
            ax.text(
                highlight_day,
                y + max(counts["count_all"].max() * 0.02, 5),
                f"{highlight_day} дн.: {y}",
                color="#B22222",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_title(title)
    ax.set_xlabel("Дней от зачатия до отела")
    ax.set_ylabel("Количество животных")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_histogram(
    *,
    input_path: str | Path,
    output_png: str | Path,
    output_xlsx: str | Path,
    subdivision: str | None,
    highlight_day: int | None,
) -> dict[str, object]:
    df = _load_pairs(input_path, subdivision=subdivision)
    counts = _build_counts(df)

    title = "Распределение срока от зачатия до отела"
    if subdivision:
        title += f" | {subdivision}"

    output_png = Path(output_png)
    output_xlsx = Path(output_xlsx)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    _plot_counts(counts, out_path=output_png, title=title, highlight_day=highlight_day)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        counts.to_excel(writer, sheet_name="counts_by_day", index=False)
        df.to_excel(writer, sheet_name="pairs_filtered", index=False)

    result: dict[str, object] = {
        "rows": int(len(df)),
        "png": str(output_png),
        "xlsx": str(output_xlsx),
    }
    if highlight_day is not None:
        row = counts[counts["day"] == highlight_day]
        if not row.empty:
            result["highlight_day"] = int(highlight_day)
            result["highlight_count_all"] = int(row["count_all"].iloc[0])
            result["highlight_count_cows"] = int(row["count_cows"].iloc[0])
            result["highlight_count_heifers"] = int(row["count_heifers"].iloc[0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="exports/conception_to_calving_pairs_enalb.xlsx",
        help="Path to conception-to-calving pairs Excel",
    )
    parser.add_argument(
        "--output-png",
        default="exports/conception_to_calving_pairs_histogram.png",
        help="Path to output PNG chart",
    )
    parser.add_argument(
        "--output-xlsx",
        default="exports/conception_to_calving_pairs_histogram.xlsx",
        help="Path to output Excel with counts by day",
    )
    parser.add_argument(
        "--subdivision",
        default=None,
        help="Optional conception_subdivision filter",
    )
    parser.add_argument(
        "--highlight-day",
        type=int,
        default=276,
        help="Day to highlight on chart",
    )
    args = parser.parse_args()

    result = build_histogram(
        input_path=args.input,
        output_png=args.output_png,
        output_xlsx=args.output_xlsx,
        subdivision=args.subdivision,
        highlight_day=args.highlight_day,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
