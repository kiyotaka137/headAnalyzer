from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from core.date_parse import parse_mixed_date, parse_mixed_datetime
from core.insemination_success import SERVICE_RESULT_MARKERS, infer_confirmed_conceptions
from db import engine
from model_params import (
    CONCEPTION_PARAMS as DEFAULT_CONCEPTION_PARAMS,
    GESTATION_DAYS as DEFAULT_GESTATION_DAYS,
    DRY_DAYS as DEFAULT_DRY_DAYS,
    DISPOSAL_PARAMS as DEFAULT_DISPOSAL_PARAMS,
    ANNUAL_DISPOSAL_RATE as DEFAULT_ANNUAL_DISPOSAL_RATE,
    BULL_CALF_DAILY_EXIT_RATE as DEFAULT_BULL_CALF_DAILY_EXIT_RATE,
    INSEMINATION_PARAMS as DEFAULT_INSEMINATION_PARAMS,
)

import re


def norm_id(x: object) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u00a0", " ").strip()
    if s == "" or s.lower() == "nan":
        return ""
    m = re.fullmatch(r"(\d+)\.0+", s)
    if m:
        return m.group(1)
    return s


def norm_result(x: object) -> str:
    if x is None:
        return ""
    return str(x).replace("\u00a0", " ").strip().upper()


def norm_event_type(x: object) -> str:
    if x is None:
        return ""
    return str(x).replace("\u00a0", " ").strip().upper().replace("Ё", "Е")


def lact_cat_from_count(n_calvings: int) -> int:
    if n_calvings <= 1:
        return 1
    if n_calvings == 2:
        return 2
    if n_calvings == 3:
        return 3
    return 4


def _safe_to_date(s: pd.Series) -> pd.Series:
    return parse_mixed_date(s)


def _merge_asof_by_reg(left: pd.DataFrame, right: pd.DataFrame, left_on: str, right_on: str) -> pd.DataFrame:
    left = left.copy()
    right = right.copy()

    left[left_on] = parse_mixed_datetime(left[left_on])
    right[right_on] = parse_mixed_datetime(right[right_on])

    left = left.dropna(subset=["reg_s", left_on]).copy()
    right = right.dropna(subset=["reg_s", right_on]).copy()

    # For merge_asof pandas requires global sorting by merge key (date/time) first.
    left = left.sort_values([left_on, "reg_s"], kind="mergesort")
    right = right.sort_values([right_on, "reg_s"], kind="mergesort")

    return pd.merge_asof(
        left,
        right,
        by="reg_s",
        left_on=left_on,
        right_on=right_on,
        direction="backward",
        allow_exact_matches=True,
    )


def _build_first_calving_by_reg(calv: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if not isinstance(calv, pd.DataFrame) or calv.empty:
        return {}

    work = calv.copy()
    work["event_type_n"] = work.get("event_type", pd.Series(dtype=object)).apply(norm_event_type)
    work["event_date_n"] = parse_mixed_datetime(work.get("event_date")).dt.normalize()
    work["birth_date_n"] = parse_mixed_datetime(work.get("birth_date")).dt.normalize()
    work["reg_s"] = work.get("reg", pd.Series(dtype=object)).apply(norm_id)
    work["mother_reg_s"] = work.get("mother_reg", pd.Series(dtype=object)).apply(norm_id)

    parts: list[pd.DataFrame] = []
    otel = work.loc[
        (work["event_type_n"] == "ОТЕЛ") & (work["reg_s"] != "") & work["event_date_n"].notna(),
        ["reg_s", "event_date_n"],
    ].rename(columns={"reg_s": "reg_s", "event_date_n": "calving_dt"})
    born = work.loc[
        (work["event_type_n"] == "РОЖДЕН") & (work["mother_reg_s"] != ""),
        ["mother_reg_s", "birth_date_n", "event_date_n"],
    ].copy()
    if not born.empty:
        born["calving_dt"] = born["birth_date_n"].where(born["birth_date_n"].notna(), born["event_date_n"])
        born = born.loc[born["calving_dt"].notna(), ["mother_reg_s", "calving_dt"]].rename(columns={"mother_reg_s": "reg_s"})
    if not otel.empty:
        parts.append(otel)
    if not born.empty:
        parts.append(born)
    if not parts:
        return {}

    all_calv = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["reg_s", "calving_dt"], keep="first")
        .sort_values(["reg_s", "calving_dt"], kind="mergesort")
    )
    return (
        all_calv.drop_duplicates(subset=["reg_s"], keep="first")
        .set_index("reg_s")["calving_dt"]
        .to_dict()
    )


def _split_pre_first_calving(
    df: pd.DataFrame,
    *,
    date_col: str,
    first_calv_by_reg: dict[str, pd.Timestamp],
    lact_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=getattr(df, "columns", [])), pd.DataFrame(columns=getattr(df, "columns", []))

    work = df.copy()
    work["reg_s"] = work.get("reg_s", work.get("reg", pd.Series(index=work.index, dtype=object)).apply(norm_id))
    work[date_col] = parse_mixed_datetime(work.get(date_col)).dt.normalize()
    work = work[(work["reg_s"] != "") & work[date_col].notna()].copy()
    if work.empty:
        return work.copy(), work.copy()

    first_dt = pd.to_datetime(work["reg_s"].map(first_calv_by_reg), errors="coerce").dt.normalize()
    has_first = first_dt.notna()
    pre_first = has_first & (work[date_col] < first_dt)
    post_first = has_first & (work[date_col] >= first_dt)

    if lact_col is not None and lact_col in work.columns:
        lact_n = pd.to_numeric(work[lact_col], errors="coerce")
        pre_first = pre_first | (~has_first & (lact_n <= 0))
        post_first = post_first | (~has_first & (lact_n > 0))

    return work.loc[pre_first].copy(), work.loc[post_first].copy()


def _trusted_first_parity_regs(ins: pd.DataFrame, first_calv_by_reg: dict[str, pd.Timestamp]) -> set[str]:
    if not isinstance(ins, pd.DataFrame) or ins.empty or not first_calv_by_reg:
        return set()

    work = ins.copy()
    work["reg_s"] = work.get("reg_s", work.get("reg", pd.Series(index=work.index, dtype=object)).apply(norm_id))
    work["event_date"] = parse_mixed_datetime(work.get("event_date")).dt.normalize()
    work["lact_n"] = pd.to_numeric(work.get("lact", pd.Series(index=work.index, dtype=object)), errors="coerce")
    work = work[(work["reg_s"] != "") & work["event_date"].notna()].copy()
    if work.empty:
        return set()

    first_dt = pd.to_datetime(work["reg_s"].map(first_calv_by_reg), errors="coerce").dt.normalize()
    post = work[first_dt.notna() & (work["event_date"] >= first_dt)].copy()
    if post.empty:
        return set()

    first_post = post.sort_values(["reg_s", "event_date"], kind="mergesort").groupby("reg_s", sort=False).head(1)
    trusted = first_post.loc[(first_post["lact_n"] > 0) & (first_post["lact_n"] <= 1), "reg_s"]
    return set(trusted.astype(str).tolist())


@dataclass(frozen=True)
class RuntimeParams:
    conception_params: object
    gestation_days: float
    dry_days: int
    disposal_params: dict
    annual_disposal_rate: float
    heifer_precalving_annual_disposal_rate: float
    bull_calf_daily_exit_rate: float
    insemination_params: object
    meta: dict


def _compute_conception_params(ins: pd.DataFrame, calv: pd.DataFrame | None = None):
    ins = ins.copy()
    ins["event_date"] = parse_mixed_datetime(ins["event_date"])
    first_calv_by_reg = _build_first_calving_by_reg(calv.copy() if isinstance(calv, pd.DataFrame) else pd.DataFrame())
    trusted_first_regs = _trusted_first_parity_regs(ins, first_calv_by_reg)
    conc = infer_confirmed_conceptions(ins)
    if conc.empty:
        return DEFAULT_CONCEPTION_PARAMS

    conc["concept_date"] = parse_mixed_datetime(conc.get("concept_date")).dt.normalize()
    conc["reg_s"] = conc.get("reg_s", pd.Series(index=conc.index, dtype=object)).apply(norm_id)
    conc["lact_n"] = pd.to_numeric(conc["lact_n"], errors="coerce").fillna(0).astype(int)
    conc["dim_age_n"] = pd.to_numeric(conc["dim_age_n"], errors="coerce")
    conc = conc[conc["dim_age_n"].notna()].copy()
    if conc.empty:
        return DEFAULT_CONCEPTION_PARAMS

    first_dt = pd.to_datetime(conc["reg_s"].map(first_calv_by_reg), errors="coerce").dt.normalize()
    pre_first_trusted = (
        first_dt.notna()
        & (conc["concept_date"] < first_dt)
        & conc["reg_s"].isin(trusted_first_regs)
    )
    heifers = conc[pre_first_trusted].copy()
    cows = conc[(conc["lact_n"] > 0) & (~pre_first_trusted)].copy()
    cows["lact_cat"] = cows["lact_n"].clip(lower=1, upper=4)

    avg_by = (
        cows.groupby("lact_cat")["dim_age_n"]
        .mean()
        .to_dict()
    )
    avg_by = {int(k): float(v) for k, v in avg_by.items()}

    global_mean = float(cows["dim_age_n"].mean()) if not cows.empty else float(conc["dim_age_n"].mean())

    if not heifers.empty:
        heifer_n = int(len(heifers))
        heifer_raw = float(heifers["dim_age_n"].mean())
        heifer_w = min(1.0, heifer_n / 80.0)
        heifer_mean = float(
            DEFAULT_CONCEPTION_PARAMS.avg_heifer_age_days
            + heifer_w * (heifer_raw - float(DEFAULT_CONCEPTION_PARAMS.avg_heifer_age_days))
        )
    else:
        heifer_mean = float(DEFAULT_CONCEPTION_PARAMS.avg_heifer_age_days)

    from model_params.defaults import ConceptionParams
    return ConceptionParams(
        avg_cow_dim_by_lact={
            1: float(avg_by.get(1, global_mean)),
            2: float(avg_by.get(2, global_mean)),
            3: float(avg_by.get(3, global_mean)),
            4: float(avg_by.get(4, global_mean)),
        },
        avg_cow_dim_global=float(global_mean),
        avg_heifer_age_days=float(heifer_mean),
    )


def _compute_gestation_days(calv: pd.DataFrame, ins: pd.DataFrame) -> Tuple[float, dict]:
    calv = calv.copy()
    ins = ins.copy()

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = parse_mixed_datetime(calv["event_date"])
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)

    births = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    if births.empty:
        return float(DEFAULT_GESTATION_DAYS), {"n": 0}

    births = births[["mother_reg_s", "event_date"]].drop_duplicates().rename(
        columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"}
    )

    conc = infer_confirmed_conceptions(ins)
    if conc.empty:
        return float(DEFAULT_GESTATION_DAYS), {"n": 0}

    p = conc[["reg_s", "concept_date"]].rename(columns={"concept_date": "p_dt"})
    merged = _merge_asof_by_reg(
        births.rename(columns={"calving_dt": "left_dt"}),
        p.rename(columns={"p_dt": "right_dt"}),
        "left_dt",
        "right_dt",
    )

    merged["gest_days"] = (merged["left_dt"] - merged["right_dt"]).dt.days
    merged = merged[merged["gest_days"].between(200, 310, inclusive="both")].copy()

    if merged.empty:
        return float(DEFAULT_GESTATION_DAYS), {"n": 0}

    mean = float(merged["gest_days"].mean())
    meta = {
        "n": int(len(merged)),
        "min": int(merged["gest_days"].min()),
        "median": float(merged["gest_days"].median()),
        "mean": float(mean),
        "max": int(merged["gest_days"].max()),
    }
    return mean, meta


def _compute_dry_days(calv: pd.DataFrame, dry: pd.DataFrame) -> Tuple[int, dict]:
    calv = calv.copy()
    dry = dry.copy()

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = parse_mixed_datetime(calv["event_date"])
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)

    births = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    if births.empty:
        return int(DEFAULT_DRY_DAYS), {"n": 0}

    births = births[["mother_reg_s", "event_date"]].drop_duplicates().rename(
        columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"}
    )

    dry["event_date"] = parse_mixed_datetime(dry["event_date"])
    dry["reg_s"] = dry["reg"].apply(norm_id)
    dry = dry[(dry["reg_s"] != "") & (dry["event_date"].notna())].copy()
    if dry.empty:
        return int(DEFAULT_DRY_DAYS), {"n": 0}

    dry = dry[["reg_s", "event_date"]].rename(columns={"event_date": "dry_dt"})

    merged = _merge_asof_by_reg(
        births.rename(columns={"calving_dt": "left_dt"}),
        dry.rename(columns={"dry_dt": "right_dt"}),
        "left_dt",
        "right_dt",
    )

    merged["dry_days"] = (merged["left_dt"] - merged["right_dt"]).dt.days
    merged = merged[merged["dry_days"].between(10, 200, inclusive="both")].copy()

    if merged.empty:
        return int(DEFAULT_DRY_DAYS), {"n": 0}

    mean = float(merged["dry_days"].mean())
    median = float(merged["dry_days"].median())
    meta = {
        "n": int(len(merged)),
        "min": int(merged["dry_days"].min()),
        "median": median,
        "mean": mean,
        "max": int(merged["dry_days"].max()),
    }
    return int(round(mean)), meta


def _compute_disposal_params(calv: pd.DataFrame, disp: pd.DataFrame) -> Tuple[dict, float, dict]:
    calv = calv.copy()
    disp = disp.copy()

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = parse_mixed_datetime(calv["event_date"])
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)

    calv_events = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    calv_events = calv_events.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"})[["reg_s", "calving_dt"]]
    if calv_events.empty:
        return DEFAULT_DISPOSAL_PARAMS, float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0}

    calv_counts = calv_events.groupby("reg_s")["calving_dt"].count().to_dict()
    calv_events = calv_events.drop_duplicates()

    disp["event_date"] = parse_mixed_datetime(disp["event_date"])
    disp["reg_s"] = disp["reg"].apply(norm_id)
    disp["reason"] = disp.get("disposal_reason", "").astype(str).str.lower().str.replace("ё", "е")
    disp = disp[(disp["reg_s"] != "") & (disp["event_date"].notna())].copy()

    if disp.empty:
        return DEFAULT_DISPOSAL_PARAMS, float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0}

    disp = disp[~disp["reason"].str.contains("переезд", na=False)].copy()

    merged = _merge_asof_by_reg(
        disp.rename(columns={"event_date": "left_dt"})[["reg_s", "left_dt"]],
        calv_events.rename(columns={"calving_dt": "right_dt"})[["reg_s", "right_dt"]],
        "left_dt",
        "right_dt",
    )
    merged["dim"] = (merged["left_dt"] - merged["right_dt"]).dt.days
    merged = merged[merged["dim"].between(0, 500, inclusive="both")].copy()
    if merged.empty:
        return DEFAULT_DISPOSAL_PARAMS, float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0}

    merged["lact_cat"] = merged["reg_s"].map(lambda r: lact_cat_from_count(int(calv_counts.get(r, 1))))
    merged["lact_cat"] = merged["lact_cat"].astype(int)

    by_lact = {}
    for l in (1, 2, 3, 4):
        x = merged[merged["lact_cat"] == l]["dim"]
        if x.empty:
            by_lact[l] = {"n": 0, "mean_dim": 0.0, "median_dim": 0.0}
        else:
            by_lact[l] = {
                "n": int(len(x)),
                "mean_dim": float(x.mean()),
                "median_dim": float(x.median()),
            }

    overall = {
        "n": int(len(merged)),
        "mean_dim": float(merged["dim"].mean()),
        "median_dim": float(merged["dim"].median()),
    }

    disposal_params = {"by_lact": by_lact, "overall": overall}

                                        
    dmin = merged["left_dt"].min()
    dmax = merged["left_dt"].max()
    years = (dmax - dmin).days / 365.25 if pd.notna(dmin) and pd.notna(dmax) else 0.0
    herd_proxy = float(len(calv_counts)) if len(calv_counts) > 0 else 0.0
    if years >= 0.25 and herd_proxy > 0:
        annual_rate = float(overall["n"] / (herd_proxy * years))
        annual_rate = float(max(0.0, min(0.5, annual_rate)))
    else:
        annual_rate = float(DEFAULT_ANNUAL_DISPOSAL_RATE)

    meta = {"n": int(len(merged)), "years": float(years), "herd_proxy": float(herd_proxy)}
    return disposal_params, annual_rate, meta


def _compute_heifer_precalving_disposal_rate(calv: pd.DataFrame, disp: pd.DataFrame, ins: pd.DataFrame) -> tuple[float, dict]:
    calv = calv.copy()
    disp = disp.copy()
    ins = ins.copy()

    if disp.empty or ins.empty:
        return float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0, "exposure_regs": 0, "years": 0.0}

    first_calv_by_reg = _build_first_calving_by_reg(calv)
    trusted_first_regs = _trusted_first_parity_regs(ins, first_calv_by_reg)
    if not trusted_first_regs:
        return float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0, "exposure_regs": 0, "years": 0.0}

    disp["event_date"] = parse_mixed_datetime(disp.get("event_date")).dt.normalize()
    disp["reg_s"] = disp.get("reg", pd.Series(index=disp.index, dtype=object)).apply(norm_id)
    disp["reason"] = disp.get("disposal_reason", "").astype(str).str.lower().str.replace("ё", "е")
    disp = disp[(disp["reg_s"] != "") & disp["event_date"].notna()].copy()
    disp = disp[~disp["reason"].str.contains("переезд", na=False)].copy()
    if disp.empty:
        return float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0, "exposure_regs": 0, "years": 0.0}

    first_dt_disp = pd.to_datetime(disp["reg_s"].map(first_calv_by_reg), errors="coerce").dt.normalize()
    disp_pre = disp[
        first_dt_disp.notna()
        & (disp["event_date"] < first_dt_disp)
        & disp["reg_s"].isin(trusted_first_regs)
    ].copy()

    ins["event_date"] = parse_mixed_datetime(ins.get("event_date")).dt.normalize()
    ins["reg_s"] = ins.get("reg", pd.Series(index=ins.index, dtype=object)).apply(norm_id)
    ins = ins[(ins["reg_s"] != "") & ins["event_date"].notna()].copy()
    first_dt_ins = pd.to_datetime(ins["reg_s"].map(first_calv_by_reg), errors="coerce").dt.normalize()
    ins_pre = ins[
        first_dt_ins.notna()
        & (ins["event_date"] < first_dt_ins)
        & ins["reg_s"].isin(trusted_first_regs)
    ].copy()

    exposure_regs = set(ins_pre["reg_s"].astype(str).tolist()) | set(disp_pre["reg_s"].astype(str).tolist())
    exposure_regs |= {str(reg) for reg in trusted_first_regs if reg in first_calv_by_reg}
    if not exposure_regs:
        return float(DEFAULT_ANNUAL_DISPOSAL_RATE), {"n": 0, "exposure_regs": 0, "years": 0.0}

    date_parts: list[pd.Series] = []
    if not ins_pre.empty:
        date_parts.append(ins_pre["event_date"])
    if not disp_pre.empty:
        date_parts.append(disp_pre["event_date"])
    first_calv_dates = pd.to_datetime(pd.Series([first_calv_by_reg.get(reg) for reg in exposure_regs]), errors="coerce").dropna()
    if not first_calv_dates.empty:
        date_parts.append(first_calv_dates)
    if not date_parts:
        return float(DEFAULT_ANNUAL_DISPOSAL_RATE), {
            "n": int(len(disp_pre)),
            "exposure_regs": int(len(exposure_regs)),
            "years": 0.0,
        }

    all_dates = pd.concat(date_parts, ignore_index=True)
    dmin = pd.to_datetime(all_dates, errors="coerce").min()
    dmax = pd.to_datetime(all_dates, errors="coerce").max()
    years = (dmax - dmin).days / 365.25 if pd.notna(dmin) and pd.notna(dmax) else 0.0
    if years < 0.25:
        return float(DEFAULT_ANNUAL_DISPOSAL_RATE), {
            "n": int(len(disp_pre)),
            "exposure_regs": int(len(exposure_regs)),
            "years": float(years),
            "trusted_regs": int(len(trusted_first_regs)),
        }

    raw_rate = float(len(disp_pre) / (max(1, len(exposure_regs)) * years))
    rate = float(max(0.0, min(0.5, raw_rate)))
    return rate, {
        "n": int(len(disp_pre)),
        "exposure_regs": int(len(exposure_regs)),
        "years": float(years),
        "trusted_regs": int(len(trusted_first_regs)),
    }


def _compute_bull_calf_daily_exit_rate(calv: pd.DataFrame, disp: pd.DataFrame) -> tuple[float, dict]:
    calv = calv.copy()
    disp = disp.copy()

    if calv.empty or disp.empty:
        return float(DEFAULT_BULL_CALF_DAILY_EXIT_RATE), {"n": 0, "mean_exit_days": None}

    calv["event_type_n"] = calv.get("event_type", pd.Series(dtype=object)).apply(norm_event_type)
    calv["sex_n"] = calv.get("sex", pd.Series(dtype=object)).astype("string").str.upper().str.strip()
    calv["reg_s"] = calv.get("reg", pd.Series(dtype=object)).apply(norm_id)
    calv["birth_date_n"] = parse_mixed_datetime(calv.get("birth_date")).dt.normalize()
    calv["event_date_n"] = parse_mixed_datetime(calv.get("event_date")).dt.normalize()

    born = calv[
        (calv["event_type_n"] == "РОЖДЕН")
        & calv["sex_n"].isin(["M", "М"])
        & (calv["reg_s"] != "")
    ].copy()
    if born.empty:
        return float(DEFAULT_BULL_CALF_DAILY_EXIT_RATE), {"n": 0, "mean_exit_days": None}

    born["born_dt"] = born["birth_date_n"].where(born["birth_date_n"].notna(), born["event_date_n"])
    born = born[born["born_dt"].notna()].copy()
    if born.empty:
        return float(DEFAULT_BULL_CALF_DAILY_EXIT_RATE), {"n": 0, "mean_exit_days": None}

    born = born.sort_values(["reg_s", "born_dt"], kind="mergesort").drop_duplicates(subset=["reg_s"], keep="first")

    disp["event_date_n"] = parse_mixed_datetime(disp.get("event_date")).dt.normalize()
    disp["reg_s"] = disp.get("reg", pd.Series(index=disp.index, dtype=object)).apply(norm_id)
    disp["reason"] = disp.get("disposal_reason", "").astype(str).str.lower().str.replace("ё", "е")
    disp = disp[(disp["reg_s"] != "") & disp["event_date_n"].notna()].copy()
    disp = disp[~disp["reason"].str.contains("переезд", na=False)].copy()
    if disp.empty:
        return float(DEFAULT_BULL_CALF_DAILY_EXIT_RATE), {"n": 0, "mean_exit_days": None}

    first_disp = (
        disp.sort_values(["reg_s", "event_date_n"], kind="mergesort")
        .drop_duplicates(subset=["reg_s"], keep="first")[["reg_s", "event_date_n"]]
    )
    merged = born[["reg_s", "born_dt"]].merge(first_disp, on="reg_s", how="left")
    merged["age_exit_days"] = (merged["event_date_n"] - merged["born_dt"]).dt.days
    merged = merged[merged["age_exit_days"].between(0, 60, inclusive="both")].copy()
    if merged.empty:
        return float(DEFAULT_BULL_CALF_DAILY_EXIT_RATE), {"n": 0, "mean_exit_days": None}

    mean_exit_days = float(max(1.0, min(60.0, merged["age_exit_days"].mean())))
    daily_rate = float(1.0 - np.exp(-1.0 / mean_exit_days))
    daily_rate = float(max(0.0, min(0.8, daily_rate)))
    return daily_rate, {
        "n": int(len(merged)),
        "mean_exit_days": mean_exit_days,
        "median_exit_days": float(merged["age_exit_days"].median()),
        "p90_exit_days": float(merged["age_exit_days"].quantile(0.9)),
    }


def _compute_insemination_params(ins: pd.DataFrame, calv: pd.DataFrame):
    ins = ins.copy()
    calv = calv.copy()

    ins["event_date"] = parse_mixed_datetime(ins["event_date"])
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")
    ins["reg_s"] = ins["reg"].apply(norm_id)
    first_calv_by_reg = _build_first_calving_by_reg(calv)
    trusted_first_regs = _trusted_first_parity_regs(ins, first_calv_by_reg)
    first_dt = pd.to_datetime(ins["reg_s"].map(first_calv_by_reg), errors="coerce").dt.normalize()
    pre_first_trusted = first_dt.notna() & (ins["event_date"] < first_dt) & ins["reg_s"].isin(trusted_first_regs)
    ins_heif_all = ins[pre_first_trusted].copy()
    ins_cow_all = ins[((ins["lact"] > 0) & (~pre_first_trusted))].copy()

    def _shrink_to_default(x: float, default: float, n: int, full_weight_n: float) -> float:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return float(default)
        w = min(1.0, max(0.0, float(n) / float(full_weight_n)))
        return float(default + w * (float(x) - float(default)))

    season_by_month = {
        12: "winter", 1: "winter", 2: "winter",
        3: "spring", 4: "spring", 5: "spring",
        6: "summer", 7: "summer", 8: "summer",
        9: "autumn", 10: "autumn", 11: "autumn",
    }
    season_order = ("winter", "spring", "summer", "autumn")

    def _month_factors(df: pd.DataFrame) -> dict[int, float]:
        work = df[(df["event_date"].notna()) & (df["reg_s"] != "")].copy()
        if work.empty:
            return {m: 1.0 for m in range(1, 13)}
        work["month"] = work["event_date"].dt.month.astype("Int64")
        work["season"] = work["month"].map(lambda m: season_by_month.get(int(m), ""))
        work["is_p"] = (work["result_norm"] == "P").astype(float)
        overall_rate = float(work["is_p"].mean())
        if not (overall_rate > 1e-9):
            return {m: 1.0 for m in range(1, 13)}

        season_factors: dict[str, float] = {}
        for season in season_order:
            part = work.loc[work["season"] == season]
            n = int(len(part))
            if n <= 0:
                season_factors[season] = 1.0
                continue
            season_rate = float(part["is_p"].mean())
            raw = season_rate / overall_rate if overall_rate > 1e-9 else 1.0
            shrink = min(1.0, n / 120.0)
            season_factors[season] = float(max(0.80, min(1.20, 1.0 + shrink * (raw - 1.0))))
        return {
            month: float(season_factors.get(season_by_month[month], 1.0))
            for month in range(1, 13)
        }

    def _pregnancy_loss_rate(df: pd.DataFrame, default: float, full_weight_n: float) -> float:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return float(default)
        work = df[df["result_norm"].isin(["P", "A"])].copy()
        if work.empty:
            return float(default)
        n = int(len(work))
        raw = float((work["result_norm"] == "A").mean())
        raw = float(max(0.0, min(0.5, raw)))
        w = min(1.0, float(n) / float(full_weight_n))
        return float(max(0.0, min(0.5, float(default) + w * (raw - float(default)))))

               
    def mean_interval(df: pd.DataFrame) -> float:
        df = df[(df["reg_s"] != "") & (df["event_date"].notna())].copy()
        df = df[df["result_norm"].isin(SERVICE_RESULT_MARKERS)].copy()
        if df.empty:
            return float("nan")
        df = df.sort_values(["reg_s", "event_date"], kind="mergesort")
        d = df.groupby("reg_s")["event_date"].diff().dt.days
        d = d[(d.notna()) & (d > 0) & (d <= 365)]
        return float(d.mean()) if not d.empty else float("nan")

    cow_ai_interval = mean_interval(ins_cow_all)
    heifer_ai_interval_raw = mean_interval(ins_heif_all)
    heifer_ai_interval = _shrink_to_default(
        heifer_ai_interval_raw,
        DEFAULT_INSEMINATION_PARAMS.heifer_ai_interval_days,
        int(len(ins_heif_all)),
        120.0,
    )
    cow_preg_loss = _pregnancy_loss_rate(
        ins_cow_all,
        DEFAULT_INSEMINATION_PARAMS.cow_pregnancy_loss_rate,
        120.0,
    )
    heifer_preg_loss = _pregnancy_loss_rate(
        ins_heif_all,
        DEFAULT_INSEMINATION_PARAMS.heifer_pregnancy_loss_rate,
        80.0,
    )
    cow_conception_month_factors = _month_factors(ins_cow_all)
    heifer_conception_month_factors = _month_factors(ins_heif_all)

    calv["event_type_n"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = parse_mixed_datetime(calv["event_date"])
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
    births = calv[(calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & (calv["event_date"].notna())].copy()
    births = births.rename(columns={"mother_reg_s": "reg_s", "event_date": "calving_dt"})[["reg_s", "calving_dt"]]
    births = births.drop_duplicates().sort_values(["reg_s", "calving_dt"], kind="mergesort")

    cow_first_ai_by_lact = {1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan}
    cow_spc = float("nan")

    if not births.empty:
        ins_cow = ins_cow_all[(ins_cow_all["reg_s"] != "") & (ins_cow_all["event_date"].notna())].copy()
        ins_cow_svc = ins_cow[ins_cow["result_norm"].isin(SERVICE_RESULT_MARKERS)].copy()
        if not ins_cow_svc.empty:
            left = ins_cow_svc[["reg_s", "event_date", "lact", "result_norm", "dim_age"]].rename(columns={"event_date": "left_dt"})
            right = births.rename(columns={"calving_dt": "right_dt"})[["reg_s", "right_dt"]]
            merged = _merge_asof_by_reg(left, right, "left_dt", "right_dt")
            merged = merged[merged["left_dt"] >= merged["right_dt"]].copy()

            merged = merged.sort_values(["reg_s", "right_dt", "left_dt"], kind="mergesort")
            first_ai = merged.groupby(["reg_s", "right_dt"], sort=False).head(1).copy()
            first_ai["lact_cat"] = first_ai["lact"].clip(lower=1, upper=4)
            if first_ai["dim_age"].notna().any():
                agg = first_ai.groupby("lact_cat")["dim_age"].mean().to_dict()
                for k, v in agg.items():
                    cow_first_ai_by_lact[int(k)] = float(v)
            merged["service_no"] = merged.groupby(["reg_s", "right_dt"], sort=False).cumcount() + 1
            conc_cow = infer_confirmed_conceptions(ins_cow)
            if not conc_cow.empty:
                conc_left = conc_cow[["reg_s", "concept_date"]].rename(columns={"concept_date": "left_dt"})
                conc_cycle = _merge_asof_by_reg(conc_left, right, "left_dt", "right_dt")
                conc_cycle = conc_cycle[conc_cycle["left_dt"] >= conc_cycle["right_dt"]].copy()
                if not conc_cycle.empty:
                    conc_cycle = conc_cycle.merge(
                        merged[["reg_s", "right_dt", "left_dt", "service_no"]].drop_duplicates(),
                        on=["reg_s", "right_dt", "left_dt"],
                        how="left",
                    )
                    vals = conc_cycle["service_no"].dropna().astype(float).tolist()
                    if vals:
                        cow_spc = float(np.mean(vals))

    ins_h = ins_heif_all[(ins_heif_all["reg_s"] != "") & (ins_heif_all["event_date"].notna())].copy()
    heifer_first_ai_age = float("nan")
    heifer_spc = float("nan")
    if not ins_h.empty:
        ins_h_svc = ins_h[ins_h["result_norm"].isin(SERVICE_RESULT_MARKERS)].copy()
        ins_h_for_first = ins_h_svc if not ins_h_svc.empty else ins_h
        conc_h = infer_confirmed_conceptions(ins_h)
        if not conc_h.empty:
            conc_h["concept_date"] = parse_mixed_datetime(conc_h.get("concept_date")).dt.normalize()
            conc_h["lact_n"] = pd.to_numeric(conc_h.get("lact_n"), errors="coerce").fillna(0).astype(int)
            first_dt_h = pd.to_datetime(conc_h["reg_s"].map(first_calv_by_reg), errors="coerce").dt.normalize()
            pre_first_conc_trusted = (
                first_dt_h.notna()
                & (conc_h["concept_date"] < first_dt_h)
                & conc_h["reg_s"].isin(trusted_first_regs)
            )
            conc_h = conc_h[(conc_h["lact_n"] <= 0) | pre_first_conc_trusted].copy()
        ins_h = ins_h.sort_values(["reg_s", "event_date"], kind="mergesort")
        first = ins_h_for_first.sort_values(["reg_s", "event_date"], kind="mergesort").groupby("reg_s", sort=False).head(1)
        if first["dim_age"].notna().any():
            heifer_first_ai_age = _shrink_to_default(
                float(first["dim_age"].mean()),
                DEFAULT_INSEMINATION_PARAMS.heifer_first_ai_age_days,
                int(len(first)),
                120.0,
            )

        if not conc_h.empty and not ins_h_svc.empty:
            left = conc_h[["reg_s", "concept_date"]].rename(columns={"concept_date": "left_dt"})
            right = ins_h_svc[["reg_s", "event_date"]].rename(columns={"event_date": "right_dt"})
            merged = _merge_asof_by_reg(left, right, "left_dt", "right_dt")
            merged = merged[(merged["right_dt"].notna()) & (merged["left_dt"] >= merged["right_dt"])].copy()
            if not merged.empty:
                svc_seq = (
                    ins_h_svc.sort_values(["reg_s", "event_date"], kind="mergesort")
                    .groupby("reg_s", sort=False)
                    .cumcount()
                    .add(1)
                )
                ins_h_svc = ins_h_svc.sort_values(["reg_s", "event_date"], kind="mergesort").copy()
                ins_h_svc["service_no"] = svc_seq
                merged = merged.merge(
                    ins_h_svc[["reg_s", "event_date", "service_no"]].rename(columns={"event_date": "right_dt"}),
                    on=["reg_s", "right_dt"],
                    how="left",
                )
                vals = merged["service_no"].dropna().astype(float).tolist()
                if vals:
                    heifer_spc = _shrink_to_default(
                        float(np.mean(vals)),
                        DEFAULT_INSEMINATION_PARAMS.heifer_services_per_conception,
                        int(len(vals)),
                        80.0,
                    )

    def _fallback(x: float, fb: float) -> float:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return float(fb)
        return float(x)

    from model_params.defaults import InseminationParams

    cow_first_ai_by_lact = {
        1: _fallback(cow_first_ai_by_lact.get(1), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(1)),
        2: _fallback(cow_first_ai_by_lact.get(2), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(2)),
        3: _fallback(cow_first_ai_by_lact.get(3), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(3)),
        4: _fallback(cow_first_ai_by_lact.get(4), DEFAULT_INSEMINATION_PARAMS.cow_first_ai_dim_by_lact.get(4)),
    }

    return InseminationParams(
        cow_first_ai_dim_by_lact=cow_first_ai_by_lact,
        cow_ai_interval_days=_fallback(cow_ai_interval, DEFAULT_INSEMINATION_PARAMS.cow_ai_interval_days),
        cow_services_per_conception=_fallback(cow_spc, DEFAULT_INSEMINATION_PARAMS.cow_services_per_conception),
        cow_pregnancy_loss_rate=_fallback(cow_preg_loss, DEFAULT_INSEMINATION_PARAMS.cow_pregnancy_loss_rate),
        heifer_first_ai_age_days=_fallback(heifer_first_ai_age, DEFAULT_INSEMINATION_PARAMS.heifer_first_ai_age_days),
        heifer_ai_interval_days=_fallback(heifer_ai_interval, DEFAULT_INSEMINATION_PARAMS.heifer_ai_interval_days),
        heifer_services_per_conception=_fallback(heifer_spc, DEFAULT_INSEMINATION_PARAMS.heifer_services_per_conception),
        heifer_pregnancy_loss_rate=_fallback(heifer_preg_loss, DEFAULT_INSEMINATION_PARAMS.heifer_pregnancy_loss_rate),
        cow_conception_month_factors=cow_conception_month_factors,
        heifer_conception_month_factors=heifer_conception_month_factors,
    )


def compute_params_from_db() -> RuntimeParams:
    calv = pd.read_sql("SELECT reg, mother_reg, birth_date, sex, event_type, event_date FROM calvings_births_raw", con=engine)
    ins = pd.read_sql("SELECT reg, lact, dim_age, event_date, bull, result FROM inseminations_raw", con=engine)
    dry = pd.read_sql("SELECT reg, dim, event_date FROM dryoff_raw", con=engine)
    disp = pd.read_sql("SELECT reg, event_date, disposal_reason FROM disposals_raw", con=engine)

    conception_params = _compute_conception_params(ins, calv)
    gest, gest_meta = _compute_gestation_days(calv, ins)
    dry_days, dry_meta = _compute_dry_days(calv, dry)
    disposal_params, annual_rate, disp_meta = _compute_disposal_params(calv, disp)
    heifer_precalving_rate, heifer_disp_meta = _compute_heifer_precalving_disposal_rate(calv, disp, ins)
    bull_calf_exit_rate, bull_exit_meta = _compute_bull_calf_daily_exit_rate(calv, disp)
    insemination_params = _compute_insemination_params(ins, calv)

    meta = {
        "gestation": gest_meta,
        "dry": dry_meta,
        "disposal": disp_meta,
        "heifer_precalving_disposal": heifer_disp_meta,
        "bull_calf_exit": bull_exit_meta,
    }

    return RuntimeParams(
        conception_params=conception_params,
        gestation_days=float(gest),
        dry_days=int(dry_days),
        disposal_params=disposal_params,
        annual_disposal_rate=float(annual_rate),
        heifer_precalving_annual_disposal_rate=float(heifer_precalving_rate),
        bull_calf_daily_exit_rate=float(bull_calf_exit_rate),
        insemination_params=insemination_params,
        meta=meta,
    )

from dataclasses import dataclass
from datetime import timedelta
from collections import Counter
from calendar import monthrange


@dataclass
class PendingCalvings:
    """Ожидаемые отёлы, уже 'заложенные' до даты старта прогноза."""
    cows: Counter
    heifers: Counter
    meta: dict


def _compute_pending_calvings_from_history(
    ins: pd.DataFrame,
    start_date: date,
    gestation_days: int,
) -> PendingCalvings:
    """
    Берём P-осеменения в окне [start_date - gestation_days, start_date],
    дедуп по животному (берём последнее P), считаем due_date = ai_date + gestation_days.
    """
    if ins is None or ins.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    df = infer_confirmed_conceptions(ins)
    if df.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    window_start = start_date - timedelta(days=int(gestation_days))
    df["concept_date"] = parse_mixed_date(df["concept_date"])
    df["lact_n"] = pd.to_numeric(df.get("lact_n", 0), errors="coerce").fillna(0).astype(int)
    df = df[(df["concept_date"] >= window_start) & (df["concept_date"] <= start_date)].copy()
    if df.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    df = df.sort_values(["reg_s", "concept_date"], kind="mergesort")
    df = df.groupby("reg_s", sort=False).tail(1)

    df["due_date"] = df["concept_date"].apply(lambda d: d + timedelta(days=int(gestation_days)))
                                            
    df = df[df["due_date"] >= start_date].copy()
    if df.empty:
        return PendingCalvings(Counter(), Counter(), {"n_total": 0, "n_cows": 0, "n_heifers": 0})

    cows_due = df[df["lact_n"] > 0]["due_date"].tolist()
    heifers_due = df[df["lact_n"] <= 0]["due_date"].tolist()

    cows = Counter(cows_due)
    heifers = Counter(heifers_due)
    meta = {"n_total": int(len(df)), "n_cows": int(len(cows_due)), "n_heifers": int(len(heifers_due))}
    return PendingCalvings(cows=cows, heifers=heifers, meta=meta)


def compute_pending_calvings_from_db(
    start_date: date,
    gestation_days: int | None = None,
) -> PendingCalvings:
    """
    Публичная функция: посчитать pending calvings из БД на дату старта прогноза.
    """
    g = int(round(float(gestation_days if gestation_days is not None else DEFAULT_GESTATION_DAYS)))
    ins = pd.read_sql("SELECT reg, lact, event_date, result FROM inseminations_raw", con=engine)
    return _compute_pending_calvings_from_history(ins, start_date=start_date, gestation_days=g)
