from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.insemination_success import infer_confirmed_conceptions
from etl.bulls import read_bulls_txt
from etl.calvings_births import read_calvings_excel
from etl.inseminations import clean_inseminations, read_inseminations_excel
from forecast_dynamic_normalization import classify_semen_from_bull_type_strict, norm_event_type, norm_id


SEMINATION_KEEP_COLS = [
    "reg",
    "lact",
    "dim_age",
    "event_date",
    "bull",
    "result",
    "event_type",
    "tech_id",
    "insemination_type",
    "technician",
    "__farm",
    "__subdivision",
]


def _norm_text_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.replace("\u00a0", " ", regex=False).str.strip()


def _lact_group(v: object) -> str:
    n = pd.to_numeric(v, errors="coerce")
    if pd.isna(n):
        return "unknown"
    if n <= 0:
        return "0"
    if n <= 4:
        return str(int(n))
    return "5+"


def _season_from_month(month: object) -> str:
    n = pd.to_numeric(month, errors="coerce")
    if pd.isna(n):
        return "unknown"
    m = int(n)
    if m in (12, 1, 2):
        return "winter"
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    return "autumn"


def _discover_bull_files(*, calv_path: str | Path, ins_path: str | Path) -> list[Path]:
    dirs = []
    for p in (calv_path, ins_path):
        pp = Path(p).resolve().parent
        if pp not in dirs:
            dirs.append(pp)
    files: list[Path] = []
    seen: set[Path] = set()
    for folder in dirs:
        for cand in folder.iterdir():
            if not cand.is_file():
                continue
            name = cand.name.lower()
            if "быки" not in name or not name.endswith(".txt"):
                continue
            if cand not in seen:
                seen.add(cand)
                files.append(cand)
    return sorted(files)


def _load_bull_lookup(bull_paths: list[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in bull_paths:
        p = Path(path)
        try:
            with p.open("rb") as fh:
                df = read_bulls_txt(fh)
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.copy()
        df["bull_code_s"] = df.get("bull_code", pd.Series(index=df.index, dtype=object)).apply(norm_id)
        df["bull_reg"] = df.get("reg", pd.Series(index=df.index, dtype=object)).apply(norm_id)
        df["bull_short_name"] = _norm_text_series(df.get("short_name", pd.Series(index=df.index, dtype=object)))
        df["bull_breed"] = _norm_text_series(df.get("breed", pd.Series(index=df.index, dtype=object)))
        df["bull_type_raw"] = _norm_text_series(df.get("bull_type", pd.Series(index=df.index, dtype=object)))
        df["semen_type"] = df["bull_type_raw"].map(classify_semen_from_bull_type_strict)
        df["bull_plem"] = pd.to_numeric(df.get("plem", pd.Series(index=df.index, dtype=object)), errors="coerce")
        frames.append(
            df[
                [
                    "bull_code_s",
                    "bull_reg",
                    "bull_short_name",
                    "bull_breed",
                    "bull_type_raw",
                    "semen_type",
                    "bull_plem",
                ]
            ].copy()
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "bull_code_s",
                "bull_reg",
                "bull_short_name",
                "bull_breed",
                "bull_type_raw",
                "semen_type",
                "bull_plem",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out = out[out["bull_code_s"] != ""].copy()
    out = out.sort_values(["bull_code_s", "semen_type", "bull_type_raw"], kind="mergesort")
    out = out.drop_duplicates(subset=["bull_code_s"], keep="first").reset_index(drop=True)
    return out


def _prepare_conceptions(ins_path: str | Path, *, farm: str = "", subdivision: str = "") -> pd.DataFrame:
    ins_raw = read_inseminations_excel(ins_path, include_meta=True)
    ins = clean_inseminations(ins_raw.copy())
    for col in SEMINATION_KEEP_COLS:
        if col not in ins.columns:
            ins[col] = pd.NA

    if farm:
        ins = ins[_norm_text_series(ins["__farm"]) == str(farm).strip()].copy()
    if subdivision:
        ins = ins[_norm_text_series(ins["__subdivision"]) == str(subdivision).strip()].copy()

    if ins.empty:
        return pd.DataFrame()

    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["event_date_n"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["lact_n"] = pd.to_numeric(ins["lact"], errors="coerce")
    ins["dim_age_n"] = pd.to_numeric(ins["dim_age"], errors="coerce")
    ins["bull_s"] = ins["bull"].apply(norm_id)

    conc = infer_confirmed_conceptions(ins)
    if conc.empty:
        return pd.DataFrame()

    svc_meta = ins[
        [
            "reg_s",
            "event_date_n",
            "lact_n",
            "dim_age_n",
            "bull_s",
            "__farm",
            "__subdivision",
            "event_type",
            "result",
            "tech_id",
            "insemination_type",
            "technician",
        ]
    ].copy()
    svc_meta = svc_meta.rename(
        columns={
            "event_date_n": "concept_date",
            "__farm": "conception_farm",
            "__subdivision": "conception_subdivision",
            "event_type": "service_event_type",
            "result": "service_result",
        }
    )
    svc_meta = svc_meta.sort_values(["reg_s", "concept_date"], kind="mergesort")
    svc_meta = svc_meta.drop_duplicates(subset=["reg_s", "concept_date"], keep="last")

    out = conc.merge(svc_meta, on=["reg_s", "concept_date"], how="left", suffixes=("", "_meta"))
    return out


def _prepare_calving_events(calv_path: str | Path, *, farm: str = "", subdivision: str = "") -> pd.DataFrame:
    calv = read_calvings_excel(calv_path, include_meta=True)
    if farm:
        calv = calv[_norm_text_series(calv["__farm"]) == str(farm).strip()].copy()
    if subdivision:
        calv = calv[_norm_text_series(calv["__subdivision"]) == str(subdivision).strip()].copy()

    if calv.empty:
        return pd.DataFrame()

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date_n"] = pd.to_datetime(calv["event_date"], errors="coerce").dt.normalize()
    calv["birth_date_n"] = pd.to_datetime(calv["birth_date"], errors="coerce").dt.normalize()
    calv["sex_n"] = _norm_text_series(calv.get("sex", pd.Series(index=calv.index, dtype=object))).str.upper()
    calv["reg_s"] = calv["reg"].apply(norm_id)
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
    calv["lact_n"] = pd.to_numeric(calv.get("lact"), errors="coerce")

    otel = calv.loc[
        (calv["event_type_n"] == "ОТЕЛ") & (calv["reg_s"] != "") & calv["event_date_n"].notna(),
        ["reg_s", "event_date_n", "lact_n", "__farm", "__subdivision"],
    ].copy()
    otel["calving_source"] = "ОТЕЛ"
    otel["source_priority"] = 0
    otel = otel.rename(
        columns={
            "reg_s": "cow_reg_s",
            "event_date_n": "calving_date",
            "lact_n": "calving_lact",
            "__farm": "calving_farm",
            "__subdivision": "calving_subdivision",
        }
    )

    born = calv.loc[
        (calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != ""),
        ["mother_reg_s", "birth_date_n", "event_date_n", "lact_n", "__farm", "__subdivision", "sex_n"],
    ].copy()
    born["calving_date"] = born["birth_date_n"].where(born["birth_date_n"].notna(), born["event_date_n"])
    born = born.loc[born["calving_date"].notna()].copy()
    born["calving_source"] = "РОЖДЕН"
    born["source_priority"] = 1
    born = born.rename(
        columns={
            "mother_reg_s": "cow_reg_s",
            "lact_n": "calving_lact",
            "__farm": "calving_farm",
            "__subdivision": "calving_subdivision",
        }
    )
    born = born[["cow_reg_s", "calving_date", "calving_lact", "calving_farm", "calving_subdivision", "calving_source", "source_priority"]]

    calf_stats = pd.DataFrame(
        columns=[
            "cow_reg_s",
            "calving_date",
            "calf_count_in_calving",
            "male_calves_in_calving",
            "female_calves_in_calving",
            "is_multiple_birth",
        ]
    )
    if not born.empty:
        born_stats = calv.loc[
            (calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != ""),
            ["mother_reg_s", "birth_date_n", "event_date_n", "sex_n"],
        ].copy()
        born_stats["calving_date"] = born_stats["birth_date_n"].where(born_stats["birth_date_n"].notna(), born_stats["event_date_n"])
        born_stats = born_stats.loc[born_stats["calving_date"].notna()].copy()
        if not born_stats.empty:
            born_stats["male_calf"] = (born_stats["sex_n"] == "M").astype(int)
            born_stats["female_calf"] = (born_stats["sex_n"] == "F").astype(int)
            calf_stats = (
                born_stats.groupby(["mother_reg_s", "calving_date"], as_index=False)
                .agg(
                    calf_count_in_calving=("sex_n", "size"),
                    male_calves_in_calving=("male_calf", "sum"),
                    female_calves_in_calving=("female_calf", "sum"),
                )
                .rename(columns={"mother_reg_s": "cow_reg_s"})
            )
            calf_stats["is_multiple_birth"] = calf_stats["calf_count_in_calving"] > 1

    otel = otel[["cow_reg_s", "calving_date", "calving_lact", "calving_farm", "calving_subdivision", "calving_source", "source_priority"]]
    events = pd.concat([otel, born], ignore_index=True)
    events = events[(events["cow_reg_s"] != "") & events["calving_date"].notna()].copy()
    events = events.sort_values(["cow_reg_s", "calving_date", "source_priority"], kind="mergesort")
    events = events.drop_duplicates(subset=["cow_reg_s", "calving_date"], keep="first").reset_index(drop=True)
    if not calf_stats.empty:
        events = events.merge(calf_stats, on=["cow_reg_s", "calving_date"], how="left")
    else:
        events["calf_count_in_calving"] = pd.NA
        events["male_calves_in_calving"] = pd.NA
        events["female_calves_in_calving"] = pd.NA
        events["is_multiple_birth"] = pd.NA
    return events


def build_pairs(
    calv_path: str | Path,
    ins_path: str | Path,
    *,
    farm: str = "",
    subdivision: str = "",
    bull_paths: list[str | Path] | None = None,
) -> pd.DataFrame:
    conc = _prepare_conceptions(ins_path, farm=farm, subdivision=subdivision)
    calv = _prepare_calving_events(calv_path, farm=farm, subdivision=subdivision)
    if conc.empty or calv.empty:
        return pd.DataFrame()

    left = calv.rename(columns={"cow_reg_s": "reg_s"}).sort_values(["calving_date", "reg_s"], kind="mergesort").copy()
    right = conc.sort_values(["concept_date", "reg_s"], kind="mergesort").copy()

    merged = pd.merge_asof(
        left,
        right,
        by="reg_s",
        left_on="calving_date",
        right_on="concept_date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged[merged["concept_date"].notna()].copy()
    merged["days_from_conception_to_calving"] = (
        pd.to_datetime(merged["calving_date"], errors="coerce") - pd.to_datetime(merged["concept_date"], errors="coerce")
    ).dt.days
    merged = merged[merged["days_from_conception_to_calving"].notna()].copy()
    merged["days_from_conception_to_calving"] = merged["days_from_conception_to_calving"].astype(int)
    merged = merged[merged["days_from_conception_to_calving"] > 0].copy()
    merged["plausible_gestation_200_320"] = merged["days_from_conception_to_calving"].between(200, 320, inclusive="both")

    bull_lookup = _load_bull_lookup(bull_paths or _discover_bull_files(calv_path=calv_path, ins_path=ins_path))
    if not bull_lookup.empty:
        merged = merged.merge(bull_lookup, left_on="bull_s", right_on="bull_code_s", how="left")
    else:
        merged["bull_reg"] = pd.NA
        merged["bull_short_name"] = pd.NA
        merged["bull_breed"] = pd.NA
        merged["bull_type_raw"] = pd.NA
        merged["semen_type"] = pd.NA
        merged["bull_plem"] = pd.NA

    merged["same_farm_between_conception_and_calving"] = (
        _norm_text_series(merged["conception_farm"]) == _norm_text_series(merged["calving_farm"])
    )
    merged["same_subdivision_between_conception_and_calving"] = (
        _norm_text_series(merged["conception_subdivision"]) == _norm_text_series(merged["calving_subdivision"])
    )

    merged["conception_month"] = pd.to_datetime(merged["concept_date"], errors="coerce").dt.month
    merged["conception_season"] = merged["conception_month"].map(_season_from_month)
    merged["lactation_group"] = merged["lact_n"].map(_lact_group)
    merged["is_heifer_pregnancy"] = pd.to_numeric(merged["lact_n"], errors="coerce").fillna(0).le(0)

    out = merged[
        [
            "reg_s",
            "days_from_conception_to_calving",
            "concept_date",
            "calving_date",
            "conception_month",
            "conception_season",
            "lact_n",
            "lactation_group",
            "calving_lact",
            "dim_age_n",
            "is_heifer_pregnancy",
            "conception_subdivision",
            "semen_type",
            "bull_breed",
            "bull_plem",
            "calf_count_in_calving",
            "male_calves_in_calving",
            "female_calves_in_calving",
            "is_multiple_birth",
            "plausible_gestation_200_320",
        ]
    ].copy()

    out = out.rename(
        columns={
            "reg_s": "reg",
            "concept_date": "conception_date",
            "lact_n": "lactation_at_conception",
            "calving_lact": "lactation_at_calving",
            "dim_age_n": "dim_or_age_at_conception",
        }
    )

    out = out.sort_values(["calving_date", "reg"], kind="mergesort").reset_index(drop=True)
    return out


def export_pairs_to_excel(df: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plausible = df[df["plausible_gestation_200_320"]].copy() if not df.empty else df.copy()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_pairs", index=False)
        plausible.to_excel(writer, sheet_name="plausible_200_320", index=False)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт пар 'зачатие -> отёл' в Excel.")
    parser.add_argument("--calv", required=True, help="Путь к файлу 'Отёлы + родившиеся'")
    parser.add_argument("--ins", required=True, help="Путь к файлу 'Осеменения'")
    parser.add_argument("--output", required=True, help="Путь к выходному xlsx")
    parser.add_argument("--farm", default="", help="Опциональный фильтр по хозяйству")
    parser.add_argument("--subdivision", default="", help="Опциональный фильтр по подразделению")
    parser.add_argument("--bulls", nargs="*", default=None, help="Опциональные txt-файлы таблиц быков")
    args = parser.parse_args()

    df = build_pairs(
        args.calv,
        args.ins,
        farm=args.farm,
        subdivision=args.subdivision,
        bull_paths=args.bulls,
    )
    out = export_pairs_to_excel(df, args.output)
    print(f"saved: {out}")
    print(f"rows_all: {len(df)}")
    print(f"rows_plausible_200_320: {int(df['plausible_gestation_200_320'].sum()) if not df.empty else 0}")


if __name__ == "__main__":
    main()
