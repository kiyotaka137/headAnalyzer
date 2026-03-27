from __future__ import annotations

"""
forecast_dynamic.py

Динамическая (дневная) симуляция поголовья на основе:
- фактических таблиц (inseminations_raw, calvings_births_raw, dryoff_raw, disposals_raw, bulls_raw)
- параметров модели (model_params/defaults.py)

Ключевая идея:
1) На дату "as_of" собираем агрегированное состояние стада (HerdState).
2) Дальше каждый день "прокручиваем" состояние:
   - возраст/ДИМ сдвигаются
   - беременности "отсчитываются" к отёлу
   - часть open животных осеменяется и часть из них становится стельной
   - часть животных выбывает по hazard-формам
   - в конце месяца применяем ограничения по вместимости (реализация)
3) Для каждой целевой даты (конец месяца) считаем срезы по группам.

ВАЖНО ПРО "НУЛИ В ТЁЛКАХ":
Исторически нули в "Тёлки 3–8 мес" и "Тёлки ≥9 мес" возникают, когда initial state
собирался только из "телят с reg" (строки телят), а в данных таких строк нет или мало.
В этом файле initial state всегда «подсекается» по событиям отёла матери:
- если в calvings_births_raw есть строки телят — используем их;
- если строк телят нет — восстанавливаем рождение телёнка из события отёла коровы (по матери и дате).

При этом:
- НЕ удваиваем: если по отёлу есть строки телят — НЕ синтезируем телёнка "вдобавок";
- если у телёнка не указан пол — распределяем по долям bull/heifer для semen типа (trad/sex),
  определённого по последнему P-осеменению перед отёлом.
"""

from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

from core.insemination_success import SERVICE_RESULT_MARKERS, infer_confirmed_conceptions
from db import engine
from forecast_dynamic_normalization import (
    SemenSexRatio,
    classify_semen_from_bull_type,
    classify_semen_from_bull_type_strict,
    is_transfer_disposal_reason,
    norm_event_type,
    norm_gender,
    norm_id,
    norm_result,
    norm_sex,
    to_semen_ratio as _to_semen_ratio,
)
from model_params import (
    GESTATION_DAYS,
    DRY_DAYS,
    CONCEPTION_PARAMS,
    DISPOSAL_PARAMS,
    ANNUAL_DISPOSAL_RATE,
    HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE,
    BULL_CALF_DAILY_EXIT_RATE,
    SEMEN_USAGE_PROBS,
    SEMEN_SEX_RATIOS,
    INSEMINATION_PARAMS,
    HERD_CAPACITY,
)

import re
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


def _prepare_calving_rows(calv: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(calv, pd.DataFrame) or calv.empty:
        return pd.DataFrame(
            columns=[
                "reg_s",
                "mother_reg_s",
                "event_type_n",
                "event_date_n",
                "birth_date_n",
                "calving_dt_n",
                "sex_norm",
            ]
        )

    c = calv.copy()
    c["event_type_n"] = c.get("event_type", pd.Series(dtype=object)).apply(norm_event_type)
    c["event_date_n"] = pd.to_datetime(c.get("event_date"), errors="coerce").dt.normalize()
    c["birth_date_n"] = pd.to_datetime(c.get("birth_date"), errors="coerce").dt.normalize()
    c["reg_s"] = c.get("reg", pd.Series(dtype=object)).apply(norm_id)
    c["mother_reg_s"] = c.get("mother_reg", pd.Series(dtype=object)).apply(norm_id)
    c["sex_norm"] = c.get("sex", pd.Series(dtype=object)).apply(norm_sex)
    c["calving_dt_n"] = c["event_date_n"]
    born_mask = c["event_type_n"] == "РОЖДЕН"
    if bool(born_mask.any()):
        c.loc[born_mask, "calving_dt_n"] = c.loc[born_mask, "birth_date_n"].where(
            c.loc[born_mask, "birth_date_n"].notna(),
            c.loc[born_mask, "event_date_n"],
        )
    return c


def _extract_calf_births(calv: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Возвращает уникальные рождения телят (reg телёнка) с датой рождения и полом.
    Берём ТОЛЬКО event_type="РОЖДЕН", чтобы не сломаться на "последних" строках.
    """
    if calv.empty:
        return pd.DataFrame(columns=["reg_s", "birth_dt", "sex_norm"])

    c = _prepare_calving_rows(calv)

    born = c[
        (c["event_type_n"] == "РОЖДЕН")
        & (c["reg_s"].notna()) & (c["reg_s"] != "")
        & (c["sex_norm"].isin(["F", "M"]))
    ].copy()

    if born.empty:
        return pd.DataFrame(columns=["reg_s", "birth_dt", "sex_norm"])

    born["birth_dt"] = born["birth_date_n"]
    m = born["birth_dt"].isna() & born["event_date_n"].notna()
    born.loc[m, "birth_dt"] = born.loc[m, "event_date_n"]

    born = born[born["birth_dt"].notna() & (born["birth_dt"] <= as_of_ts)].copy()

                                                                
    born = (
        born.sort_values(["reg_s", "birth_dt"], kind="mergesort")
            .groupby("reg_s", sort=False, as_index=False)
            .first()[["reg_s", "birth_dt", "sex_norm"]]
    )
    return born
MAX_DIM = 500
MAX_AGE_DAYS = 730
BULL_AGE_MAX = 90                                             
OVERDUE_CLAMP_DAYS = 14

BIRTH_OUTPUT_KEYS = (
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
)


def age_months(d: int) -> int:
    return int(d // 30)


def end_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    first_next = date(d.year, d.month + 1, 1)
    return first_next - timedelta(days=1)


def _month_end_shift(d_end: date, months_delta: int) -> date:
    ts = pd.Timestamp(d_end) + pd.DateOffset(months=months_delta)
    return end_of_month(date(int(ts.year), int(ts.month), 1))


def _months_between_eom(start_eom: date, end_eom: date) -> int:
    return (int(end_eom.year) - int(start_eom.year)) * 12 + (int(end_eom.month) - int(start_eom.month))


def shift_right(a: np.ndarray) -> np.ndarray:
    """Возраст +1 день: index i -> i+1. То, что было на хвосте, «вылетает»."""
    out = np.zeros_like(a)
    out[1:] = a[:-1]
    return out


def shift_left(a: np.ndarray) -> np.ndarray:
    """Countdown -1 день: index i -> i-1. То, что было в 0, «вылетает» (событие наступило)."""
    out = np.zeros_like(a)
    out[:-1] = a[1:]
    return out

def _effective_ai_interval_days(interval_raw: float, mean_target: float, first: float, spc: float) -> float:
    """
    Стабилизация интервала между осеменениями.

    interval_raw из данных часто шумный/завышенный.
    Мы хотим, чтобы средняя "точка зачатия" не уезжала:
        first + (spc-1)*interval ≈ mean_target

    Поэтому берём derived-интервал из этой формулы и миксуем с raw.
    """
    interval_raw = _clamp(float(interval_raw), 14.0, 90.0)
    spc = float(spc)

    if spc <= 1.01:
        return interval_raw

    derived = (float(mean_target) - float(first)) / max(1e-9, (spc - 1.0))
    derived = _clamp(derived, 14.0, 60.0)

    return 0.30 * interval_raw + 0.70 * derived

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _normalize_month_factor_map(raw: Any) -> dict[int, float]:
    out = {m: 1.0 for m in range(1, 13)}
    if not isinstance(raw, dict):
        return out

    for k, v in raw.items():
        try:
            month = int(k)
            value = float(v)
        except Exception:
            continue
        if month < 1 or month > 12 or not np.isfinite(value):
            continue
        out[month] = _clamp(value, 0.75, 1.25)
    return out


_SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


def _apply_season_factors_to_month_map(raw: Any, base_map: dict[int, float]) -> dict[int, float]:
    out = dict(base_map or {m: 1.0 for m in range(1, 13)})
    if not isinstance(raw, dict):
        return out

    for season, months in _SEASON_MONTHS.items():
        if season not in raw:
            continue
        try:
            value = float(raw.get(season))
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        value = _clamp(value, 0.75, 1.25)
        for month in months:
            out[int(month)] = value
    return out


def _normalize_semen_usage_shares(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None

    cow_sex = raw.get("cow_sex")
    cow_trad = raw.get("cow_trad")
    heifer_sex = raw.get("heifer_sex")
    heifer_trad = raw.get("heifer_trad")

    if cow_sex is None and cow_trad is None and heifer_sex is None and heifer_trad is None:
        return None

    def _pair(a_raw: Any, b_raw: Any, a_fb: float, b_fb: float) -> tuple[float, float]:
        a = None if a_raw is None else _clamp(float(a_raw), 0.0, 1.0)
        b = None if b_raw is None else _clamp(float(b_raw), 0.0, 1.0)
        if a is None and b is None:
            a, b = float(a_fb), float(b_fb)
        elif a is None:
            a = 1.0 - float(b)
        elif b is None:
            b = 1.0 - float(a)
        s = max(1e-9, float(a) + float(b))
        return float(a) / s, float(b) / s

    cow_trad_n, cow_sex_n = _pair(
        cow_trad,
        cow_sex,
        float(SEMEN_USAGE_PROBS.cow_trad),
        float(SEMEN_USAGE_PROBS.cow_sex),
    )
    heifer_trad_n, heifer_sex_n = _pair(
        heifer_trad,
        heifer_sex,
        float(SEMEN_USAGE_PROBS.heifer_trad),
        float(SEMEN_USAGE_PROBS.heifer_sex),
    )
    return {
        "cow_trad": cow_trad_n,
        "cow_sex": cow_sex_n,
        "heifer_trad": heifer_trad_n,
        "heifer_sex": heifer_sex_n,
    }


def _month_factor_value(month_factors: dict[int, float], dt_like: Any) -> float:
    try:
        month = int(pd.Timestamp(dt_like).month)
    except Exception:
        return 1.0
    return float(month_factors.get(month, 1.0))


                                                              
                                                              

def _merge_asof_safe(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    by: str | None = None,
    direction: str = "backward",
    allow_exact_matches: bool = True,
    suffixes: tuple[str, str] = ("", "_r"),
) -> pd.DataFrame:
    if direction != "backward":
        raise ValueError("Only direction='backward' implemented in fallback version")

    l = left.copy()
    r = right.copy()
    l["_row_id"] = np.arange(len(l), dtype=np.int64)

    l[left_on] = pd.to_datetime(l[left_on], errors="coerce")
    r[right_on] = pd.to_datetime(r[right_on], errors="coerce")

    if by is None:
        l = l[l[left_on].notna()].copy()
        r = r[r[right_on].notna()].copy()
        l = l.sort_values([left_on], kind="mergesort").reset_index(drop=True)
        r = r.sort_values([right_on], kind="mergesort").reset_index(drop=True)

        out = pd.merge_asof(
            l,
            r,
            left_on=left_on,
            right_on=right_on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )
        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

    l[by] = l[by].astype("string").fillna("").str.strip()
    r[by] = r[by].astype("string").fillna("").str.strip()

    l = l[(l[by] != "") & l[left_on].notna()].copy()
    r = r[(r[by] != "") & r[right_on].notna()].copy()

    l = l.sort_values([left_on, by], kind="mergesort").reset_index(drop=True)
    r = r.sort_values([right_on, by], kind="mergesort").reset_index(drop=True)

    try:
        out = pd.merge_asof(
            l,
            r,
            by=by,
            left_on=left_on,
            right_on=right_on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )
        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)

    except ValueError as e:
        if "keys must be sorted" not in str(e).lower():
            raise

        out = l.copy()
        for c in r.columns:
            if c not in out.columns:
                out[c] = pd.NA

        r_groups: dict[str, pd.DataFrame] = {}
        for key, grp in r.groupby(by, sort=False):
            r_groups[str(key)] = grp.sort_values(right_on, kind="mergesort")

        for key, lg in out.groupby(by, sort=False):
            rg = r_groups.get(str(key))
            if rg is None or rg.empty:
                continue
            lt = lg[left_on].values.astype("datetime64[ns]").astype("int64")
            rt = rg[right_on].values.astype("datetime64[ns]").astype("int64")
            pos = np.searchsorted(rt, lt, side="right") - 1
            ok = pos >= 0
            if not np.any(ok):
                continue
            out_idx = lg.index.values[ok]
            take = rg.iloc[pos[ok]]
            for c in rg.columns:
                if c == by:
                    continue
                if c in out.columns and c == left_on:
                    continue
                out.loc[out_idx, c] = take[c].values

        return out.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"]).reset_index(drop=True)


                                                              
                                                              

def _resolve_runtime_params(overrides: dict | None) -> dict:
    ov = overrides or {}

    def _as_bool(v: object, default: bool) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
        return default

    gest_default = float(GESTATION_DAYS)
    dry_default = int(DRY_DAYS)

    cp = ov.get("CONCEPTION_PARAMS") or {
        "avg_cow_dim_by_lact": dict(CONCEPTION_PARAMS.avg_cow_dim_by_lact),
        "avg_cow_dim_global": float(CONCEPTION_PARAMS.avg_cow_dim_global),
        "avg_heifer_age_days": float(CONCEPTION_PARAMS.avg_heifer_age_days),
    }

    disp = ov.get("DISPOSAL_PARAMS") or deepcopy(DISPOSAL_PARAMS)
    annual_disp = float(ov.get("ANNUAL_DISPOSAL_RATE", ANNUAL_DISPOSAL_RATE))
    heifer_precalving_disp = float(
        ov.get(
            "HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE",
            ov.get("heifer_precalving_annual_disposal_rate", HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE),
        )
    )
    bull_calf_exit = float(
        ov.get(
            "BULL_CALF_DAILY_EXIT_RATE",
            ov.get("bull_calf_daily_exit_rate", BULL_CALF_DAILY_EXIT_RATE),
        )
    )

    ins = ov.get("INSEMINATION_PARAMS") or {
        "cow_services_per_conception": float(INSEMINATION_PARAMS.cow_services_per_conception),
        "cow_ai_interval_days": float(INSEMINATION_PARAMS.cow_ai_interval_days),
        "cow_pregnancy_loss_rate": float(INSEMINATION_PARAMS.cow_pregnancy_loss_rate),
        "cow_first_ai_dim_by_lact": dict(INSEMINATION_PARAMS.cow_first_ai_dim_by_lact),
        "cow_conception_month_factors": dict(INSEMINATION_PARAMS.cow_conception_month_factors),
        "heifer_services_per_conception": float(INSEMINATION_PARAMS.heifer_services_per_conception),
        "heifer_ai_interval_days": float(INSEMINATION_PARAMS.heifer_ai_interval_days),
        "heifer_pregnancy_loss_rate": float(INSEMINATION_PARAMS.heifer_pregnancy_loss_rate),
        "heifer_first_ai_age_days": float(INSEMINATION_PARAMS.heifer_first_ai_age_days),
        "heifer_conception_month_factors": dict(INSEMINATION_PARAMS.heifer_conception_month_factors),
    }
    ins["cow_conception_month_factors"] = _apply_season_factors_to_month_map(
        ins.get("cow_conception_season_factors"),
        _normalize_month_factor_map(
            ins.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
        ),
    )
    ins["heifer_conception_month_factors"] = _apply_season_factors_to_month_map(
        ins.get("heifer_conception_season_factors"),
        _normalize_month_factor_map(
            ins.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
        ),
    )

    semen_usage = _normalize_semen_usage_shares(ov.get("SEMEN_USAGE_SHARES"))
    if semen_usage is None:
        semen_usage = _normalize_semen_usage_shares(ov.get("semen_usage"))

    cap_norm = dict(_CAP_NORM)
    cap_ov = ov.get("HERD_CAPACITY")
    if cap_ov is None:
        cap_ov = ov.get("herd_capacity")
    if isinstance(cap_ov, dict):
        for k, v in cap_ov.items():
            try:
                iv = int(round(float(v)))
            except Exception:
                continue
            cap_norm[_norm_key(str(k))] = max(0, iv)

    gest_days = int(round(float(ov.get("GESTATION_DAYS", gest_default))))
    gest_days = max(200, min(310, gest_days))

    dry_days = int(round(float(ov.get("DRY_DAYS", dry_default))))
    dry_days = max(20, min(120, dry_days))

    annual_disp = float(max(0.0, min(0.5, annual_disp)))
    heifer_precalving_disp = float(max(0.0, min(0.5, heifer_precalving_disp)))
    bull_calf_exit = float(max(0.0, min(0.8, bull_calf_exit)))
    apply_capacity = _as_bool(ov.get("APPLY_CAPACITY"), True)
    if _as_bool(ov.get("DISABLE_CAPACITY"), False):
        apply_capacity = False

    return {
        "GESTATION_DAYS": gest_days,
        "DRY_DAYS": dry_days,
        "CONCEPTION_PARAMS": cp,
        "DISPOSAL_PARAMS": disp,
        "ANNUAL_DISPOSAL_RATE": annual_disp,
        "HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE": heifer_precalving_disp,
        "BULL_CALF_DAILY_EXIT_RATE": bull_calf_exit,
        "INSEMINATION_PARAMS": ins,
        "SEMEN_USAGE_SHARES": semen_usage,
        "HERD_CAPACITY_NORM": cap_norm,
        "APPLY_CAPACITY": apply_capacity,
    }


                                                              
                                                              

@dataclass
class HerdState:
            
    open_dim: Dict[int, np.ndarray]
    preg_lact: Dict[Tuple[int, str], np.ndarray]
    preg_dry:  Dict[Tuple[int, str], np.ndarray]
                    
    heifer_age: np.ndarray
    heifer_preg: Dict[str, np.ndarray]
    bull_age: np.ndarray


def init_empty_state(gest_days: int) -> HerdState:
    open_dim = {l: np.zeros(MAX_DIM + 1, dtype=float) for l in (1, 2, 3, 4)}
    preg_lact = {(l, s): np.zeros(gest_days + 1, dtype=float) for l in (1, 2, 3, 4) for s in ("trad", "sex")}
    preg_dry  = {(l, s): np.zeros(gest_days + 1, dtype=float) for l in (1, 2, 3, 4) for s in ("trad", "sex")}
    heifer_age = np.zeros(MAX_AGE_DAYS + 1, dtype=float)
    heifer_preg = {s: np.zeros(gest_days + 1, dtype=float) for s in ("trad", "sex")}
    bull_age = np.zeros(BULL_AGE_MAX + 1, dtype=float)
    return HerdState(open_dim, preg_lact, preg_dry, heifer_age, heifer_preg, bull_age)


def _copy_state(s: HerdState) -> HerdState:
    return HerdState(
        open_dim={k: v.copy() for k, v in s.open_dim.items()},
        preg_lact={k: v.copy() for k, v in s.preg_lact.items()},
        preg_dry={k: v.copy() for k, v in s.preg_dry.items()},
        heifer_age=s.heifer_age.copy(),
        heifer_preg={k: v.copy() for k, v in s.heifer_preg.items()},
        bull_age=s.bull_age.copy(),
    )


def _state_group_snapshot(state: HerdState) -> Dict[str, float]:
    cows_open = sum(float(state.open_dim[l].sum()) for l in (1, 2, 3, 4))
    cows_preg_lact = sum(float(state.preg_lact[(l, s)].sum()) for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(float(state.preg_dry[(l, s)].sum()) for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    return {
        "Дойные коровы": float(cows_open + cows_preg_lact),
        "Сухостойные коровы": float(cows_preg_dry),
        "Тёлки 0–3 мес": float(state.heifer_age[:90].sum()),
        "Бычки 0–2 мес": float(state.bull_age[:61].sum()),
        "Тёлки 3–8 мес": float(state.heifer_age[90:270].sum()),
        "Тёлки ≥9 мес": float(state.heifer_age[270:].sum()),
        "Нетели": float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum()),
    }


                                                              
                                                              

def load_tables() -> Dict[str, pd.DataFrame]:
    calv = pd.read_sql(
        "SELECT reg, mother_reg, birth_date, sex, event_type, event_date FROM calvings_births_raw",
        con=engine,
    )
    ins  = pd.read_sql(
        "SELECT reg, lact, dim_age, event_date, bull, result FROM inseminations_raw",
        con=engine,
    )
    dry  = pd.read_sql("SELECT reg, dim, event_date FROM dryoff_raw", con=engine)
    disp = pd.read_sql("SELECT reg, event_date, disposal_reason FROM disposals_raw", con=engine)
    bulls = pd.read_sql("SELECT bull_code, bull_type FROM bulls_raw", con=engine)
    return {"calv": calv, "ins": ins, "dry": dry, "disp": disp, "bulls": bulls}


def latest_data_date(tables: Dict[str, pd.DataFrame]) -> date:
    mx = None
    for key in ("calv", "ins", "dry", "disp"):
        df = tables[key]
        if "event_date" in df.columns and not df.empty:
            d = pd.to_datetime(df["event_date"], errors="coerce").max()
            if pd.notna(d):
                dx = d.date()
                mx = dx if mx is None else max(mx, dx)
    return mx or date.today()


                                                              
                                                              

def compute_semen_usage_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    ins = tables["ins"].copy()
    bulls = tables["bulls"].copy()

    fallback = {
        "cow_trad": float(SEMEN_USAGE_PROBS.cow_trad),
        "cow_sex": float(SEMEN_USAGE_PROBS.cow_sex),
        "heifer_trad": float(SEMEN_USAGE_PROBS.heifer_trad),
        "heifer_sex": float(SEMEN_USAGE_PROBS.heifer_sex),
    }

    if ins.empty or bulls.empty:
        return fallback

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    svc = ins[(ins["event_date"].notna()) & (ins["reg_s"] != "") & (ins["bull_s"] != "")].copy()
    if svc.empty:
        return fallback

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type_strict)
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    svc["semen"] = svc["bull_s"].map(semen_by_bull)
    svc["semen_known"] = svc["semen"].isin(["trad", "sex"])

                                                                     
    cows = svc[svc["lact"] > 0].copy()
    heif = svc[svc["lact"] <= 0].copy()

    if not cows.empty:
        cows = cows.sort_values(["reg_s", "lact", "event_date"], kind="mergesort")
        cows["service_no"] = cows.groupby(["reg_s", "lact"], sort=False).cumcount() + 1
        cows["policy_sex_allowed"] = cows["lact"].isin([1, 2]) & (cows["service_no"] <= 2)

    if not heif.empty:
        heif = heif.sort_values(["reg_s", "event_date"], kind="mergesort")
        heif["service_no"] = heif.groupby(["reg_s"], sort=False).cumcount() + 1
        heif["policy_sex_allowed"] = heif["service_no"] <= 3

    s = pd.concat([cows, heif], axis=0, ignore_index=True)
    if s.empty:
        return fallback

    max_dt = s["event_date"].max()
    s_365 = s[s["event_date"] >= (max_dt - pd.Timedelta(days=365))].copy() if pd.notna(max_dt) else s.copy()

    def _bayes_smooth_share(obs: float, n: int, prior: float, prior_w: float) -> float:
        return float((obs * n + prior * prior_w) / max(1e-9, n + prior_w))

    def _shares(df: pd.DataFrame) -> Tuple[float | None, float, int, int, float]:
        if df.empty:
            return None, 0.0, 0, 0, 0.0
        total_n = int(len(df))
        policy_sex = float(df["policy_sex_allowed"].mean()) if "policy_sex_allowed" in df.columns else 0.0
        known = df[df["semen_known"]]
        if known.empty:
            return None, policy_sex, total_n, 0, 0.0
        known_n = int(len(known))
        sex_obs = float((known["semen"] == "sex").mean())
        known_rate = float(known_n) / float(total_n)
        return sex_obs, policy_sex, total_n, known_n, known_rate

    def _mix_group(df_all: pd.DataFrame, df_365: pd.DataFrame, prior_sex: float) -> Tuple[float, dict]:
                                                                  
        use_recent = len(df_365) >= 120
        src = df_365 if use_recent else df_all
        window = "last_year" if use_recent else "all"

        sex_obs, policy_sex, total_n, known_n, known_rate = _shares(src)
        if total_n == 0:
            return prior_sex, {"window": window, "n_total": 0, "n_known": 0, "known_rate": 0.0, "policy_sex": 0.0, "obs_sex": None}

                                                               
        if sex_obs is None:
            blended = policy_sex
        else:
            blended = known_rate * sex_obs + (1.0 - known_rate) * policy_sex

        sex_final = _bayes_smooth_share(
            obs=float(max(0.0, min(1.0, blended))),
            n=total_n,
            prior=float(max(0.0, min(1.0, prior_sex))),
            prior_w=300.0,
        )
        sex_final = float(max(0.0, min(1.0, sex_final)))
        meta = {
            "window": window,
            "n_total": int(total_n),
            "n_known": int(known_n),
            "known_rate": float(known_rate),
            "policy_sex": float(policy_sex),
            "obs_sex": None if sex_obs is None else float(sex_obs),
        }
        return sex_final, meta

    cows_all = s[s["lact"] > 0].copy()
    cows_365 = s_365[s_365["lact"] > 0].copy()
    heif_all = s[s["lact"] <= 0].copy()
    heif_365 = s_365[s_365["lact"] <= 0].copy()

    cow_sex, meta_cow = _mix_group(cows_all, cows_365, fallback["cow_sex"])
    hef_sex, meta_heif = _mix_group(heif_all, heif_365, fallback["heifer_sex"])
    cow_trad = 1.0 - cow_sex
    hef_trad = 1.0 - hef_sex

    def _norm2(a: float, b: float) -> Tuple[float, float]:
        s = max(1e-9, a + b)
        return a / s, b / s

    cow_trad, cow_sex = _norm2(cow_trad, cow_sex)
    hef_trad, hef_sex = _norm2(hef_trad, hef_sex)

    return {
        "cow_trad": float(cow_trad),
        "cow_sex": float(cow_sex),
        "heifer_trad": float(hef_trad),
        "heifer_sex": float(hef_sex),
        "meta": {
            "method": "services_with_policy_blend",
            "cow": meta_cow,
            "heifer": meta_heif,
        },
    }


                                                              
                                                              

def compute_semen_sex_ratios_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, SemenSexRatio]:
    calv = tables["calv"].copy()
    ins = tables["ins"].copy()
    bulls = tables["bulls"].copy()

    fallback = {
        "trad": _to_semen_ratio(SEMEN_SEX_RATIOS["trad"]),
        "sex": _to_semen_ratio(SEMEN_SEX_RATIOS["sex"]),
    }

    if calv.empty or ins.empty or bulls.empty:
        return fallback

    calv["event_type"] = calv["event_type"].apply(norm_event_type)
    calv["event_date"] = pd.to_datetime(calv["event_date"], errors="coerce").dt.normalize()
    calv["mother_reg_s"] = calv["mother_reg"].apply(norm_id)
    calv["sex_norm"] = calv["sex"].apply(norm_sex)

    born = calv[
    (calv["event_type"] == "РОЖДЕН")
    & (calv["event_date"].notna())
    & (calv["mother_reg_s"] != "")
    ][["mother_reg_s", "event_date", "sex_norm"]].copy()

                                                   
    born = born[born["sex_norm"].isin(["M", "F"])].copy()

    if born.empty:
        return fallback

    born["calving_dt"] = born["event_date"]
    born["male"] = (born["sex_norm"] == "M").astype(int)
    born["female"] = (born["sex_norm"] == "F").astype(int)

    calv_ev = (
        born.groupby(["mother_reg_s", "calving_dt"], sort=False)[["male", "female"]]
        .sum()
        .reset_index()
        .rename(columns={"mother_reg_s": "reg_s"})
    )

    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    conc = infer_confirmed_conceptions(ins)
    if conc.empty:
        return fallback

    p = conc[
        (conc["reg_s"] != "")
        & (conc["bull_s"].fillna("") != "")
    ][["reg_s", "concept_date", "bull_s"]].copy()
    if p.empty:
        return fallback

    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type_strict)
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    p["semen"] = p["bull_s"].map(semen_by_bull)
    p = p[p["semen"].isin(["trad", "sex"])].copy()
    if p.empty:
        return fallback

    p = p.rename(columns={"concept_date": "ins_dt"})

    m = _merge_asof_safe(
        calv_ev.sort_values(["reg_s", "calving_dt"], kind="mergesort"),
        p[["reg_s", "ins_dt", "semen"]].sort_values(["reg_s", "ins_dt"], kind="mergesort"),
        by="reg_s",
        left_on="calving_dt",
        right_on="ins_dt",
        direction="backward",
        allow_exact_matches=True,
    )

    m = m[m["ins_dt"].notna()].copy()
    if m.empty:
        return fallback

    m["gest_days"] = (m["calving_dt"] - m["ins_dt"]).dt.days
    m = m[(m["gest_days"] >= 200) & (m["gest_days"] <= 310)].copy()
    if m.empty:
        return fallback

    out = dict(fallback)
    for semen in ("trad", "sex"):
        sub = m[m["semen"] == semen]
        total = int(sub["male"].sum() + sub["female"].sum())
        if total < 300:
            continue

        bull_share = float(sub["male"].sum()) / float(total)
        bull_share = max(0.0, min(1.0, bull_share))

                                                                                       
        if bull_share < 0.05 or bull_share > 0.95:
            continue

        bull_share = max(0.10, min(0.90, bull_share))
        out[semen] = SemenSexRatio(bull_share=bull_share, heifer_share=1.0 - bull_share)


    return out


                                                              
                                                              

def report_semen_and_calf_sex_params_from_db(tables: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Возвращает удобный словарь для UI:
    - доли использования semen (trad/sex)
    - доли пола телят для trad/sex
    - покрытие матчей по быкам (на P-осеменениях)
    """
    semen_shares = compute_semen_usage_from_db(tables)
    semen_sex_ratios = compute_semen_sex_ratios_from_db(tables)

    return {
        "semen_shares": semen_shares,
        "semen_sex_ratios": semen_sex_ratios,
    }


                                                              
                                                              

def hazard_from_pdf(pdf: np.ndarray, *, vwp: int = 0) -> np.ndarray:
    p = pdf.copy().astype(float)
    p[:vwp] = 0.0
    s = p.sum()
    if s <= 0:
        return np.zeros_like(p)
    p /= s

    hz = np.zeros_like(p)
    surv = 1.0
    for i in range(len(p)):
        if i < vwp:
            hz[i] = 0.0
            continue
        pi = float(p[i])
        if surv <= 1e-12:
            hz[i] = 0.0
        else:
            hz[i] = pi / surv
            hz[i] = max(0.0, min(1.0, hz[i]))
            surv *= (1.0 - hz[i])
    return hz


def lognormal_hazard_by_dim(dim_max: int, mean: float, median: float) -> np.ndarray:
    mean = float(mean) if mean and mean > 1 else 1.0
    median = float(median) if median and median > 1 else max(1.0, mean * 0.8)
    if mean < median:
        mean = median * 1.05

    sigma2 = 2.0 * np.log(max(1e-9, mean / median))
    sigma = float(np.sqrt(max(1e-9, sigma2)))
    mu = float(np.log(max(1e-9, median)))

    x = np.arange(dim_max + 1, dtype=float)
    pdf = np.zeros_like(x)
    xx = x[1:]
    pdf[1:] = (1.0 / (xx * sigma * np.sqrt(2.0 * np.pi))) * np.exp(-((np.log(xx) - mu) ** 2) / (2.0 * sigma2))
    return hazard_from_pdf(pdf, vwp=0)


def build_disposal_shape(disposal_params: dict) -> Dict[int, np.ndarray]:
    shape = {}
    by_lact = disposal_params.get("by_lact", {})
    for lact_cat in (1, 2, 3, 4):
        s = by_lact.get(lact_cat, {})
        m = float(s.get("mean_dim", 150.0) or 150.0)
        md = float(s.get("median_dim", 120.0) or 120.0)
        hz = lognormal_hazard_by_dim(MAX_DIM, m, md)
        nz = hz[hz > 0]
        sh = np.ones(MAX_DIM + 1, dtype=float) if nz.size == 0 else hz / (float(nz.mean()) if float(nz.mean()) > 0 else 1.0)
        shape[lact_cat] = np.clip(sh, 0.1, 5.0)
    return shape


                                                              
                                                              

import re

def _norm_key(s: str) -> str:
    s = (s or "").replace("\u00a0", " ").strip()
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = s.replace("Ё", "Е").replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s.upper()

_CAP_NORM = {_norm_key(k): int(v) for k, v in HERD_CAPACITY.items()}


def _cap(name: str, cap_norm: dict[str, int] | None = None) -> int | None:
    cap = cap_norm if isinstance(cap_norm, dict) else _CAP_NORM
    return cap.get(_norm_key(name))



def _take_from_array(arr: np.ndarray, idx_iter: Iterable[int], need: float) -> float:
    taken = 0.0
    for i in idx_iter:
        if need <= 1e-9:
            break
        v = float(arr[i])
        if v <= 0:
            continue
        x = v if v < need else need
        arr[i] = v - x
        taken += x
        need -= x
    return taken


def _sell_cows_from_doy(state: HerdState, need: float, gest_days: int) -> float:
    sold = 0.0
    for l in (4, 3, 2, 1):
        sold += _take_from_array(state.open_dim[l], range(MAX_DIM, -1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold

    for l in (4, 3, 2, 1):
        for semen in ("trad", "sex"):
            sold += _take_from_array(state.preg_lact[(l, semen)], range(gest_days, -1, -1), need - sold)
            if sold >= need - 1e-9:
                return sold
    return sold


def _sell_cows_from_dry(state: HerdState, need: float, gest_days: int, dry_days: int) -> float:
    sold = 0.0
    hi = min(dry_days, gest_days)
    for l in (4, 3, 2, 1):
        for semen in ("trad", "sex"):
            sold += _take_from_array(state.preg_dry[(l, semen)], range(hi, -1, -1), need - sold)
            if sold >= need - 1e-9:
                return sold
    return sold


def _sell_heifers_by_age(state: HerdState, need: float, age_lo: int, age_hi: int) -> float:
    lo = max(0, int(age_lo))
    hi = min(int(age_hi), len(state.heifer_age) - 1)
    return _take_from_array(state.heifer_age, range(hi, lo - 1, -1), need)


def _sell_neteli_4_6_months(state: HerdState, need: float, gest_days: int) -> float:
    sold = 0.0
    pref_lo = max(0, min(gest_days, 100))
    pref_hi = max(0, min(gest_days, 160))

    pref_range = list(range(pref_hi, pref_lo - 1, -1))
    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], pref_range, need - sold)
        if sold >= need - 1e-9:
            return sold

    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], range(gest_days, pref_hi + 1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold

    for semen in ("trad", "sex"):
        sold += _take_from_array(state.heifer_preg[semen], range(pref_lo - 1, -1, -1), need - sold)
        if sold >= need - 1e-9:
            return sold
    return sold


def _apply_capacity_month_end(
    state: HerdState,
    *,
    gest_days: int,
    dry_days: int,
    cap_norm: dict[str, int] | None = None,
) -> dict:
    out = {
        "over_doy": 0.0,
        "over_dry": 0.0,
        "over_h0": 0.0,
        "over_h38": 0.0,
        "over_h9": 0.0,
        "over_neteli": 0.0,
        "sell_cows": 0.0,
        "sell_cows_doy": 0.0,
        "sell_cows_dry": 0.0,
        "sell_heifers": 0.0,
        "sell_heifers_h0": 0.0,
        "sell_heifers_h38": 0.0,
        "sell_heifers_h9": 0.0,
        "sell_neteli": 0.0,
    }

    cap_doy = _cap("Дойные коровы", cap_norm)
    cap_dry = _cap("Сухостойные коровы", cap_norm)
    cap_h0 = _cap("Тёлки 0–3 мес", cap_norm)
    cap_h38 = _cap("Тёлки 3–8 мес", cap_norm)
    cap_h924 = _cap("Тёлки 9–24 мес", cap_norm)                                            
    cap_neteli = _cap("Нетели", cap_norm)                                                                        

                                                   
    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)

    h0 = float(state.heifer_age[:90].sum())                     
    h38 = float(state.heifer_age[90:270].sum())               
    h9 = float(state.heifer_age[270:].sum())                  
    neteli = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())

                             
    if cap_doy is not None and doy > cap_doy + 1e-9:
        need = doy - cap_doy
        sold = _sell_cows_from_doy(state, need, gest_days)
        out["over_doy"] += sold
        out["sell_cows"] += sold
        out["sell_cows_doy"] += sold

    cows_open = sum(state.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)

    if cap_dry is not None and dry > cap_dry + 1e-9:
        need = dry - cap_dry
        sold = _sell_cows_from_dry(state, need, gest_days, dry_days)
        out["over_dry"] += sold
        out["sell_cows"] += sold
        out["sell_cows_dry"] += sold

                                 
    if cap_h0 is not None and h0 > cap_h0 + 1e-9:
        need = h0 - cap_h0
        sold = _sell_heifers_by_age(state, need, 0, 89)
        out["over_h0"] += sold
        out["sell_heifers"] += sold
        out["sell_heifers_h0"] += sold

    if cap_h38 is not None and h38 > cap_h38 + 1e-9:
        need = h38 - cap_h38
        sold = _sell_heifers_by_age(state, need, 90, 269)
        out["over_h38"] += sold
        out["sell_heifers"] += sold
        out["sell_heifers_h38"] += sold

    if cap_h924 is not None:
        h9 = float(state.heifer_age[270:].sum())
        neteli = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())

        total_9plus = float(h9 + neteli)
        if total_9plus > cap_h924 + 1e-9:
            need = total_9plus - cap_h924

            sold_h9 = 0.0
            sold_n = 0.0

                                                                           
            if h9 > 1e-9 and need > 1e-9:
                take_h9 = min(h9, need)
                sold_h9 = _sell_heifers_by_age(state, take_h9, 270, MAX_AGE_DAYS)
                need = max(0.0, need - sold_h9)

                                      
            if need > 1e-9:
                sold_n = _sell_neteli_4_6_months(state, need, gest_days)

            out["over_h9"] += float(sold_h9)
            out["over_neteli"] += float(sold_n)
            out["sell_heifers"] += float(sold_h9)
            out["sell_heifers_h9"] += float(sold_h9)
            out["sell_neteli"] += float(sold_n)

                                                                                               
    if cap_neteli is not None:
        neteli2 = float(state.heifer_preg["trad"].sum() + state.heifer_preg["sex"].sum())
        if neteli2 > cap_neteli + 1e-9:
            need = neteli2 - cap_neteli
            sold = _sell_neteli_4_6_months(state, need, gest_days)
            out["over_neteli"] += float(sold)
            out["sell_neteli"] += float(sold)

    return out


                                                              
                                                              

def lact_cat_from_count(n_calvings: int) -> int:
    if n_calvings <= 1:
        return 1
    if n_calvings == 2:
        return 2
    if n_calvings == 3:
        return 3
    return 4


                                                              
                                                              

def _build_cow_like_regs(
    *,
    calv: pd.DataFrame,
    ins: pd.DataFrame,
    dry: pd.DataFrame,
    cows_regs: set[str],
    first_calv_by_reg: dict[str, pd.Timestamp] | None = None,
    as_of_ts: pd.Timestamp | None = None,
) -> set[str]:
    """
    Список регов, которые С БОЛЬШОЙ вероятностью коровы, даже если lact в inseminations пустой/0.
    Это критично, чтобы не записывать коров в "нетели/тёлки" и не раздувать молодняк.
    """
    out = set(str(x) for x in cows_regs if str(x))

    if not ins.empty:
        tmp = ins.copy()
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        tmp["lact_i"] = pd.to_numeric(tmp["lact"], errors="coerce").fillna(0).astype(int)
        cand = tmp.loc[tmp["lact_i"] > 0, "reg_s"].astype(str)
        if first_calv_by_reg and as_of_ts is not None and not cand.empty:
            first_dt = cand.map(first_calv_by_reg)
            keep = pd.to_datetime(first_dt, errors="coerce").isna() | (pd.to_datetime(first_dt, errors="coerce") <= as_of_ts)
            cand = cand.loc[keep]
        out |= set(cand)

    if not dry.empty:
        tmp = dry.copy()
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        out |= set(tmp["reg_s"].astype(str))

    if not calv.empty:
        tmp = calv.copy()
        tmp["event_type_n"] = tmp["event_type"].apply(norm_event_type)
        tmp["reg_s"] = tmp["reg"].apply(norm_id)
        tmp["mother_reg_s"] = tmp["mother_reg"].apply(norm_id)
        out |= set(tmp.loc[tmp["mother_reg_s"] != "", "mother_reg_s"].astype(str))
        out |= set(tmp.loc[(tmp["event_type_n"] == "ОТЕЛ") & (tmp["reg_s"] != ""), "reg_s"].astype(str))

    out.discard("")
    return out


def _estimate_active_cow_regs_at_asof(
    *,
    calv: pd.DataFrame,
    ins: pd.DataFrame,
    dry: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    lookback_days: int = 540,
    first_calv_by_reg: dict[str, pd.Timestamp] | None = None,
) -> set[str]:
    """
    Оценка "фактически присутствующих" коров на дату старта прогноза.
    Используем события за последние ~18 месяцев:
    - осеменения lact>0,
    - отёлы (ОТЕЛ по reg),
    - матери в строках РОЖДЕН (mother_reg),
    - запуски (dryoff).
    """
    lo = as_of_ts - pd.Timedelta(days=int(max(120, lookback_days)))
    out: set[str] = set()

    if isinstance(ins, pd.DataFrame) and not ins.empty:
        d = ins.copy()
        d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
        d["lact_n"] = pd.to_numeric(d.get("lact"), errors="coerce")
        d["reg_s"] = d.get("reg", pd.Series(dtype=object)).apply(norm_id)
        m = (
            d["event_date_n"].notna()
            & (d["event_date_n"] >= lo)
            & (d["event_date_n"] <= as_of_ts)
            & (d["lact_n"] > 0)
            & (d["reg_s"] != "")
        )
        if first_calv_by_reg:
            first_dt = pd.to_datetime(d["reg_s"].map(first_calv_by_reg), errors="coerce")
            m &= first_dt.isna() | (first_dt <= as_of_ts)
        out |= set(d.loc[m, "reg_s"].astype(str))

    if isinstance(calv, pd.DataFrame) and not calv.empty:
        d = calv.copy()
        d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
        d["event_type_n"] = d.get("event_type", pd.Series(dtype=object)).apply(norm_event_type)
        d["reg_s"] = d.get("reg", pd.Series(dtype=object)).apply(norm_id)
        d["mother_reg_s"] = d.get("mother_reg", pd.Series(dtype=object)).apply(norm_id)
        base = d["event_date_n"].notna() & (d["event_date_n"] >= lo) & (d["event_date_n"] <= as_of_ts)
        m1 = base & (d["event_type_n"] == "ОТЕЛ") & (d["reg_s"] != "")
        m2 = base & (d["event_type_n"] == "РОЖДЕН") & (d["mother_reg_s"] != "")
        out |= set(d.loc[m1, "reg_s"].astype(str))
        out |= set(d.loc[m2, "mother_reg_s"].astype(str))

    if isinstance(dry, pd.DataFrame) and not dry.empty:
        d = dry.copy()
        d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
        d["reg_s"] = d.get("reg", pd.Series(dtype=object)).apply(norm_id)
        m = (
            d["event_date_n"].notna()
            & (d["event_date_n"] >= lo)
            & (d["event_date_n"] <= as_of_ts)
            & (d["reg_s"] != "")
        )
        out |= set(d.loc[m, "reg_s"].astype(str))

    out.discard("")
    return out


def _infer_semen_for_calvings(
    calv_ev: pd.DataFrame,
    *,
    ins: pd.DataFrame,
    semen_by_bull: dict[str, str],
    gest_days: int,
) -> pd.DataFrame:
    """
    Для каждого (cow_reg_s, calving_dt) находим последнее P-осеменение перед отёлом,
    проверяем окно гестации и получаем semen ('trad'/'sex'). Если не нашли — 'trad'.
    """
    if calv_ev.empty:
        calv_ev["semen"] = "trad"
        return calv_ev

    ins2 = ins.copy()
    if ins2.empty:
        calv_ev["semen"] = "trad"
        return calv_ev

    p = infer_confirmed_conceptions(ins2)
    if p.empty:
        calv_ev["semen"] = "trad"
        return calv_ev

    p["semen"] = p["bull_s"].map(semen_by_bull)
    p.loc[~p["semen"].isin(["trad", "sex"]), "semen"] = "trad"
    p = p.rename(columns={"concept_date": "ins_dt"})

    left = calv_ev.sort_values(["cow_reg_s", "calving_dt"], kind="mergesort").copy()
    right = p[["reg_s", "ins_dt", "semen"]].sort_values(["reg_s", "ins_dt"], kind="mergesort")
    left = left.rename(columns={"cow_reg_s": "reg_s"})

    m = _merge_asof_safe(
        left,
        right,
        by="reg_s",
        left_on="calving_dt",
        right_on="ins_dt",
        direction="backward",
        allow_exact_matches=True,
    )

                   
    m["gest_d"] = (m["calving_dt"] - m["ins_dt"]).dt.days
    ok = (m["ins_dt"].notna()) & (m["gest_d"] >= 200) & (m["gest_d"] <= 310)
    m.loc[~ok, "semen"] = "trad"

    m = m.drop(columns=["reg_s"])
    m = m.rename(columns={"reg_s_r": "cow_reg_s"}) if "reg_s_r" in m.columns else m
    return m


def build_initial_state(
    tables: Dict[str, pd.DataFrame],
    as_of: date,
    *,
    gest_days: int | None = None,
    dry_days: int | None = None,
    insemination_params: dict | None = None,
    warmstart_from_services: bool = True,
    semen_sex_ratios: Dict[str, SemenSexRatio] | None = None,
) -> HerdState:
    """Собираем агрегированное состояние стада на дату as_of из animal-level snapshot."""
    gest_days = int(gest_days if gest_days is not None else int(GESTATION_DAYS))
    dry_days = int(dry_days if dry_days is not None else int(DRY_DAYS))
    as_of_ts = pd.Timestamp(as_of).normalize()

    snapshot = build_animal_asof_snapshot(
        tables,
        as_of=as_of,
        gest_days=gest_days,
        dry_days=dry_days,
        insemination_params=insemination_params,
        warmstart_from_services=warmstart_from_services,
    )
    state = _state_from_animal_asof_snapshot(snapshot, gest_days)
    return state


def build_animal_asof_snapshot(
    tables: Dict[str, pd.DataFrame],
    as_of: date,
    *,
    gest_days: int | None = None,
    dry_days: int | None = None,
    insemination_params: dict | None = None,
    warmstart_from_services: bool = True,
) -> pd.DataFrame:
    gest_days = int(gest_days if gest_days is not None else int(GESTATION_DAYS))
    dry_days = int(dry_days if dry_days is not None else int(DRY_DAYS))
    as_of_ts = pd.Timestamp(as_of).normalize()

    calv = tables.get("calv", pd.DataFrame()).copy()
    ins = tables.get("ins", pd.DataFrame()).copy()
    dry = tables.get("dry", pd.DataFrame()).copy()
    disp = tables.get("disp", pd.DataFrame()).copy()
    bulls = tables.get("bulls", pd.DataFrame()).copy()

    for col in ("event_date", "result", "lact", "dim_age", "reg", "bull"):
        if col not in ins.columns:
            ins[col] = pd.NA
    ins["event_date"] = pd.to_datetime(ins["event_date"], errors="coerce").dt.normalize()
    ins["result_norm"] = ins["result"].apply(norm_result)
    ins["lact"] = pd.to_numeric(ins["lact"], errors="coerce").fillna(0).astype(int)
    ins["dim_age"] = pd.to_numeric(ins["dim_age"], errors="coerce")
    ins["reg_s"] = ins["reg"].apply(norm_id)
    ins["bull_s"] = ins["bull"].apply(norm_id)

    for col in ("event_date", "reg", "dim"):
        if col not in dry.columns:
            dry[col] = pd.NA
    dry["event_date"] = pd.to_datetime(dry["event_date"], errors="coerce").dt.normalize()
    dry["reg_s"] = dry["reg"].apply(norm_id)
    dry["dim"] = pd.to_numeric(dry.get("dim"), errors="coerce")

    for col in ("event_date", "reg", "disposal_reason"):
        if col not in disp.columns:
            disp[col] = pd.NA
    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce").dt.normalize()
    disp["reg_s"] = disp["reg"].apply(norm_id)

    if "bull_code" not in bulls.columns:
        bulls["bull_code"] = pd.NA
    if "bull_type" not in bulls.columns:
        bulls["bull_type"] = pd.NA
    bulls["bull_code_s"] = bulls["bull_code"].apply(norm_id)
    bulls["semen"] = bulls["bull_type"].apply(classify_semen_from_bull_type)
    semen_by_bull = dict(zip(bulls["bull_code_s"], bulls["semen"]))

    disp["reason_is_transfer"] = disp["disposal_reason"].apply(is_transfer_disposal_reason)
    disposed_regs = set(
        disp.loc[
            disp["event_date"].notna()
            & (disp["event_date"] <= as_of_ts)
            & (~disp["reason_is_transfer"]),
            "reg_s",
        ].astype(str)
    )
    disposed_regs.discard("")

    dry_ok = dry[(dry["event_date"].notna()) & (dry["event_date"] <= as_of_ts) & (dry["reg_s"] != "")]
    dry_last = dry_ok.groupby("reg_s", sort=False)["event_date"].max().to_dict()

    calv2 = _prepare_calving_rows(calv)

    full_first_calv_by_reg: dict[str, pd.Timestamp] = {}
    full_last_calv_by_reg: dict[str, pd.Timestamp] = {}
    full_born = calv2[
        (calv2["event_type_n"] == "РОЖДЕН")
        & (calv2["mother_reg_s"] != "")
        & (calv2["calving_dt_n"].notna())
    ][["mother_reg_s", "calving_dt_n"]].drop_duplicates()
    full_otel = calv2[
        (calv2["event_type_n"] == "ОТЕЛ")
        & (calv2["reg_s"] != "")
        & (calv2["calving_dt_n"].notna())
    ][["reg_s", "calving_dt_n"]].drop_duplicates()
    full_parts: list[pd.DataFrame] = []
    if not full_born.empty:
        full_parts.append(full_born.rename(columns={"mother_reg_s": "reg_s", "calving_dt_n": "calving_date"}))
    if not full_otel.empty:
        full_parts.append(full_otel.rename(columns={"calving_dt_n": "calving_date"}))
    if full_parts:
        full_events = (
            pd.concat(full_parts, ignore_index=True)
            .drop_duplicates(subset=["reg_s", "calving_date"], keep="last")
            .sort_values(["reg_s", "calving_date"], kind="mergesort")
        )
        full_first_calv_by_reg = (
            full_events.drop_duplicates(subset=["reg_s"], keep="first")
            .set_index("reg_s")["calving_date"]
            .to_dict()
        )
        full_last_calv_by_reg = (
            full_events.drop_duplicates(subset=["reg_s"], keep="last")
            .set_index("reg_s")["calving_date"]
            .to_dict()
        )

    def _is_pre_first_calving(reg: object) -> bool:
        reg_s = str(reg or "")
        if not reg_s:
            return False
        first_dt = full_first_calv_by_reg.get(reg_s)
        return pd.isna(first_dt) or (pd.Timestamp(first_dt) > as_of_ts)

    future_first_heifer_regs = {
        str(reg)
        for reg, first_dt in full_first_calv_by_reg.items()
        if str(reg) and pd.notna(first_dt) and pd.Timestamp(first_dt) > as_of_ts
    }

    calv_hist = calv2[(calv2["calving_dt_n"].notna()) & (calv2["calving_dt_n"] <= as_of_ts)].copy()

    calves_born = calv_hist[
        (calv_hist["event_type_n"] == "РОЖДЕН")
        & (calv_hist["mother_reg_s"] != "")
        & (calv_hist["calving_dt_n"].notna())
    ][["mother_reg_s", "calving_dt_n"]].drop_duplicates()
    calves_otel = calv_hist[
        (calv_hist["event_type_n"] == "ОТЕЛ")
        & (calv_hist["reg_s"] != "")
        & (calv_hist["calving_dt_n"].notna())
    ][["reg_s", "calving_dt_n"]].drop_duplicates()

    calving_events_parts: list[pd.DataFrame] = []
    if not calves_born.empty:
        calving_events_parts.append(
            calves_born.rename(columns={"mother_reg_s": "reg_s", "calving_dt_n": "calving_date"})
        )
    if not calves_otel.empty:
        calving_events_parts.append(
            calves_otel.rename(columns={"calving_dt_n": "calving_date"})
        )

    calv_stats = None
    if calving_events_parts:
        calving_events = (
            pd.concat(calving_events_parts, ignore_index=True)
            .drop_duplicates(subset=["reg_s", "calving_date"], keep="last")
        )
        calv_stats = (
            calving_events.groupby("reg_s", sort=False)
            .agg(
                n_calvings=("calving_date", "count"),
                last_calving=("calving_date", "max"),
            )
            .reset_index()
        )

    ins_cow_hist = ins[
        (ins["event_date"].notna())
        & (ins["event_date"] <= as_of_ts)
        & (ins["reg_s"] != "")
        & (ins["lact"] > 0)
    ].copy()
    if future_first_heifer_regs:
        ins_cow_hist = ins_cow_hist[~ins_cow_hist["reg_s"].isin(future_first_heifer_regs)].copy()

    est_stats = None
    if not ins_cow_hist.empty:
        ins_cow_hist = ins_cow_hist.sort_values(["reg_s", "event_date"], kind="mergesort")
        last_dim_row = ins_cow_hist.groupby("reg_s", sort=False).tail(1).copy()
        valid_dim = last_dim_row["dim_age"].notna() & (last_dim_row["dim_age"] >= 0)
        last_dim_row["last_calving_est"] = pd.NaT
        if bool(valid_dim.any()):
            last_dim_row.loc[valid_dim, "last_calving_est"] = (
                last_dim_row.loc[valid_dim, "event_date"]
                - pd.to_timedelta(last_dim_row.loc[valid_dim, "dim_age"], unit="D")
            )
        last_dim_row["lact_cat_est"] = last_dim_row["lact"].clip(lower=1, upper=4)
        est_stats = last_dim_row[["reg_s", "last_calving_est", "lact_cat_est", "dim_age"]].copy()

    dry_stats = None
    if not dry_ok.empty:
        dry_cow_hist = dry_ok.copy()
        if future_first_heifer_regs:
            dry_cow_hist = dry_cow_hist[~dry_cow_hist["reg_s"].isin(future_first_heifer_regs)].copy()
        if not dry_cow_hist.empty:
            dry_cow_hist = dry_cow_hist.sort_values(["reg_s", "event_date"], kind="mergesort")
            last_dry_row = dry_cow_hist.groupby("reg_s", sort=False).tail(1).copy()
            valid_dry_dim = last_dry_row["dim"].notna() & (last_dry_row["dim"] >= 0)
            last_dry_row["last_calving_est"] = pd.NaT
            if bool(valid_dry_dim.any()):
                last_dry_row.loc[valid_dry_dim, "last_calving_est"] = (
                    last_dry_row.loc[valid_dry_dim, "event_date"]
                    - pd.to_timedelta(last_dry_row.loc[valid_dry_dim, "dim"], unit="D")
                )
            last_dry_row["lact_cat_est"] = 1
            last_dry_row["dim_age"] = last_dry_row["dim"]
            dry_stats = last_dry_row[["reg_s", "last_calving_est", "lact_cat_est", "dim_age"]].copy()

    est_like_stats = None
    if est_stats is not None:
        est_like_stats = est_stats.set_index("reg_s")
    if dry_stats is not None:
        dry_like_stats = dry_stats.set_index("reg_s")
        est_like_stats = (
            est_like_stats.combine_first(dry_like_stats)
            if est_like_stats is not None
            else dry_like_stats
        )
    if est_like_stats is not None:
        est_like_stats = est_like_stats.reset_index()

    if calv_stats is None and est_like_stats is None:
        cows = pd.DataFrame(columns=["reg_s", "last_calving", "n_calvings", "last_calving_est", "lact_cat_est", "dim_age"])
    elif calv_stats is None:
        cows = est_like_stats.copy()
        cows["n_calvings"] = pd.NA
        cows["last_calving"] = pd.NaT
    elif est_like_stats is None:
        cows = calv_stats.copy()
        cows["last_calving_est"] = pd.NaT
        cows["lact_cat_est"] = pd.NA
        cows["dim_age"] = pd.NA
    else:
        cows = calv_stats.merge(est_like_stats, on="reg_s", how="outer")

    cows = cows[(cows["reg_s"].notna()) & (cows["reg_s"] != "")].copy()
    cows = cows[~cows["reg_s"].isin(disposed_regs)].copy()

    active_cow_regs = _estimate_active_cow_regs_at_asof(
        calv=calv_hist,
        ins=ins,
        dry=dry,
        as_of_ts=as_of_ts,
        lookback_days=540,
        first_calv_by_reg=full_first_calv_by_reg,
    )
    cow_last_calving_ref = pd.to_datetime(
        cows["last_calving"].where(cows["last_calving"].notna(), cows["last_calving_est"]),
        errors="coerce",
    )
    plausible_silent_cow_regs = set(
        cows.loc[
            cow_last_calving_ref.notna()
            & (((as_of_ts - cow_last_calving_ref).dt.days) >= 0)
            & (((as_of_ts - cow_last_calving_ref).dt.days) <= 720),
            "reg_s",
        ].astype(str).tolist()
    )
    keep_cow_regs = active_cow_regs | plausible_silent_cow_regs
    if keep_cow_regs:
        cows_active = cows[cows["reg_s"].isin(keep_cow_regs)].copy()
        if not cows_active.empty:
            cows = cows_active

    cows["last_calving"] = cows["last_calving"].where(cows["last_calving"].notna(), cows["last_calving_est"])

    def _lcat(row: pd.Series) -> int:
        if pd.notna(row.get("n_calvings")):
            return lact_cat_from_count(int(row["n_calvings"]))
        if pd.notna(row.get("lact_cat_est")):
            return int(row["lact_cat_est"])
        return 1

    cows["lact_cat"] = cows.apply(_lcat, axis=1)
    cows_regs = set(cows["reg_s"].astype(str).tolist())
    cow_like_regs = _build_cow_like_regs(
        calv=calv_hist,
        ins=ins,
        dry=dry,
        cows_regs=cows_regs,
        first_calv_by_reg=full_first_calv_by_reg,
        as_of_ts=as_of_ts,
    )

    ins_p = infer_confirmed_conceptions(ins)
    ins_p["concept_date"] = pd.to_datetime(ins_p.get("concept_date"), errors="coerce").dt.normalize()
    ins_p["confirm_date"] = pd.to_datetime(ins_p.get("confirm_date"), errors="coerce").dt.normalize()
    ins_p["lact_n"] = pd.to_numeric(ins_p.get("lact_n"), errors="coerce").fillna(0).astype(int)
    ins_p["bull_s"] = ins_p.get("bull_s", "").astype("string").fillna("")
    ins_p = ins_p[
        (ins_p["concept_date"].notna())
        & (ins_p["concept_date"] <= as_of_ts)
        & (ins_p["reg_s"] != "")
    ].copy()

    preg_rows_by_reg: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    if not ins_p.empty:
        ins_p = ins_p.sort_values(["reg_s", "concept_date", "confirm_date"], kind="mergesort")
        for reg, g in ins_p.groupby("reg_s", sort=False):
            preg_rows_by_reg[str(reg)] = [
                (pd.Timestamp(rr.concept_date).normalize(), str(getattr(rr, "bull_s", "") or ""))
                for rr in g.itertuples(index=False)
                if pd.notna(getattr(rr, "concept_date", pd.NaT))
            ]

    ins_params = insemination_params or {}
    cow_spc = float(ins_params.get("cow_services_per_conception", float(INSEMINATION_PARAMS.cow_services_per_conception)))
    heif_spc = float(ins_params.get("heifer_services_per_conception", float(INSEMINATION_PARAMS.heifer_services_per_conception)))
    cow_month_factors = _normalize_month_factor_map(
        ins_params.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        ins_params.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )
    p_conc_cow_base = 1.0 / max(1e-9, cow_spc)
    p_conc_heif_base = float(ins_params.get("heifer_warmstart_p", 1.0 / max(1e-9, heif_spc)))

    last_service_cow: dict[str, pd.Timestamp] = {}
    last_service_bull_cow: dict[str, str] = {}
    last_service_heif: dict[str, pd.Timestamp] = {}
    last_service_bull_heif: dict[str, str] = {}
    if warmstart_from_services:
        ins_svc = ins[
            (ins["event_date"].notna())
            & (ins["event_date"] <= as_of_ts)
            & (ins["reg_s"] != "")
            & (ins["result_norm"].isin(SERVICE_RESULT_MARKERS))
        ].copy()
        ins_svc = ins_svc[~ins_svc["reg_s"].isin(disposed_regs)]
        if not ins_svc.empty:
            ins_svc = ins_svc.sort_values(["reg_s", "event_date"], kind="mergesort")
            last = ins_svc.groupby("reg_s", sort=False).tail(1)
            pre_first_mask = last["reg_s"].map(_is_pre_first_calving)
            cow_last = last[
                (((last["lact"] > 0) | (last["reg_s"].isin(cow_like_regs))))
                & (~pre_first_mask)
            ]
            heif_last = last[pre_first_mask]
            last_service_cow = dict(zip(cow_last["reg_s"], cow_last["event_date"]))
            last_service_bull_cow = dict(zip(cow_last["reg_s"], cow_last["bull_s"]))
            last_service_heif = dict(zip(heif_last["reg_s"], heif_last["event_date"]))
            last_service_bull_heif = dict(zip(heif_last["reg_s"], heif_last["bull_s"]))

    rows: list[dict[str, Any]] = []

    for r in cows.itertuples(index=False):
        reg = str(r.reg_s)
        lact_cat = int(r.lact_cat)
        last_calv = getattr(r, "last_calving", pd.NaT)
        dim_guess = getattr(r, "dim_age", pd.NA)
        dim = int(max(0, min(MAX_DIM, (as_of_ts - pd.Timestamp(last_calv).normalize()).days))) if pd.notna(last_calv) else int(max(0, min(MAX_DIM, float(dim_guess))) if pd.notna(dim_guess) else 0)
        dry_last_dt = dry_last.get(reg, pd.NaT)
        is_dry_fact = (
            pd.notna(dry_last_dt)
            and pd.notna(last_calv)
            and (pd.Timestamp(dry_last_dt) > pd.Timestamp(last_calv))
        )

        row = {
            "reg_s": reg,
            "group_code": "Дойные коровы",
            "basis": "open",
            "lact_cat": lact_cat,
            "dim": dim,
            "age_days": pd.NA,
            "semen": "trad",
            "days_to_calv": pd.NA,
            "cow_open_weight": 1.0,
            "cow_preg_lact_weight": 0.0,
            "cow_preg_dry_weight": 0.0,
            "heifer_preg_weight": 0.0,
            "heifer_open_weight": 0.0,
            "last_calving": pd.Timestamp(last_calv).normalize() if pd.notna(last_calv) else pd.NaT,
            "last_dry": pd.Timestamp(dry_last_dt).normalize() if pd.notna(dry_last_dt) else pd.NaT,
            "first_calving": pd.Timestamp(full_first_calv_by_reg.get(reg)).normalize() if pd.notna(full_first_calv_by_reg.get(reg)) else pd.NaT,
        }

        p_date = pd.NaT
        bull = ""
        for cand_dt, cand_bull in reversed(preg_rows_by_reg.get(reg, [])):
            if pd.notna(last_calv) and cand_dt <= pd.Timestamp(last_calv).normalize():
                continue
            p_date = cand_dt
            bull = cand_bull or ""
            break
        semen = semen_by_bull.get(bull, "trad") if bull else "trad"

        if pd.notna(p_date):
            p_date = pd.Timestamp(p_date).normalize()
            days_to_calv = int(gest_days - (as_of_ts - p_date).days)
            if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                days_to_calv = 0
            if 0 <= days_to_calv <= gest_days:
                row["semen"] = semen
                row["days_to_calv"] = days_to_calv
                row["cow_open_weight"] = 0.0
                if is_dry_fact:
                    row["group_code"] = "Сухостойные коровы"
                    row["basis"] = "confirmed_p+dry"
                    row["cow_preg_dry_weight"] = 1.0
                else:
                    row["group_code"] = "Дойные коровы"
                    row["basis"] = "confirmed_p"
                    row["cow_preg_lact_weight"] = 1.0
                rows.append(row)
                continue

        if is_dry_fact:
            dry_days_to_calv = int(dry_days - (as_of_ts - pd.Timestamp(dry_last_dt).normalize()).days)
            if dry_days_to_calv < 0 and dry_days_to_calv >= -OVERDUE_CLAMP_DAYS:
                dry_days_to_calv = 0
            dry_days_to_calv = int(_clamp(float(dry_days_to_calv), 0.0, float(dry_days)))
            row["group_code"] = "Сухостойные коровы"
            row["basis"] = "factual_dry"
            row["days_to_calv"] = dry_days_to_calv
            row["semen"] = semen_by_bull.get(last_service_bull_cow.get(reg, "") or bull or "", "trad")
            row["cow_open_weight"] = 0.0
            row["cow_preg_dry_weight"] = 1.0
            rows.append(row)
            continue

        if warmstart_from_services:
            s_date = last_service_cow.get(reg, pd.NaT)
            if pd.notna(s_date):
                s_date = pd.Timestamp(s_date).normalize()
                if not (pd.notna(last_calv) and s_date <= pd.Timestamp(last_calv).normalize()):
                    bull2 = last_service_bull_cow.get(reg, "") or ""
                    semen2 = semen_by_bull.get(bull2, "trad") if bull2 else "trad"
                    days_to_calv2 = int(gest_days - (as_of_ts - s_date).days)
                    if days_to_calv2 < 0 and days_to_calv2 >= -OVERDUE_CLAMP_DAYS:
                        days_to_calv2 = 0
                    if 0 <= days_to_calv2 <= gest_days:
                        add = _clamp(p_conc_cow_base * _month_factor_value(cow_month_factors, s_date), 0.05, 0.95)
                        row["basis"] = "service_warmstart"
                        row["semen"] = semen2
                        row["days_to_calv"] = days_to_calv2
                        row["cow_open_weight"] = 1.0 - add
                        row["cow_preg_lact_weight"] = add

        rows.append(row)

    heifer_p = ins_p[
        (ins_p["concept_date"].notna())
        & (ins_p["concept_date"] <= as_of_ts)
        & (ins_p["reg_s"] != "")
        & ins_p["reg_s"].map(_is_pre_first_calving)
        & (~ins_p["reg_s"].isin(disposed_regs))
    ].copy()
    p_regs: set[str] = set()
    if not heifer_p.empty:
        heifer_p = heifer_p.sort_values(["reg_s", "concept_date", "confirm_date"], kind="mergesort")
        heifer_last = heifer_p.groupby("reg_s", sort=False).tail(1)
        p_regs = set(heifer_last["reg_s"].astype(str).tolist())
        for rr in heifer_last.itertuples(index=False):
            p_date = rr.concept_date
            if pd.isna(p_date):
                continue
            bull = getattr(rr, "bull_s", "") or ""
            semen = semen_by_bull.get(bull, "trad") if bull else "trad"
            p_date = pd.Timestamp(p_date).normalize()
            days_to_calv = int(gest_days - (as_of_ts - p_date).days)
            if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                days_to_calv = 0
            if 0 <= days_to_calv <= gest_days:
                rows.append(
                    {
                        "reg_s": str(rr.reg_s),
                        "group_code": "Нетели",
                        "basis": "confirmed_first_p",
                        "lact_cat": 0,
                        "dim": pd.NA,
                        "age_days": pd.NA,
                        "semen": semen,
                        "days_to_calv": days_to_calv,
                        "cow_open_weight": 0.0,
                        "cow_preg_lact_weight": 0.0,
                        "cow_preg_dry_weight": 0.0,
                        "heifer_preg_weight": 1.0,
                        "heifer_open_weight": 0.0,
                        "last_calving": pd.NaT,
                        "last_dry": pd.NaT,
                        "first_calving": pd.Timestamp(full_first_calv_by_reg.get(str(rr.reg_s))).normalize() if pd.notna(full_first_calv_by_reg.get(str(rr.reg_s))) else pd.NaT,
                    }
                )

    future_neteli_regs = {
        str(reg)
        for reg, first_dt in full_first_calv_by_reg.items()
        if str(reg)
        and pd.notna(first_dt)
        and (pd.Timestamp(first_dt) > as_of_ts)
        and (pd.Timestamp(first_dt) <= as_of_ts + pd.Timedelta(days=gest_days))
        and str(reg) not in disposed_regs
        and str(reg) not in p_regs
    }
    for reg in future_neteli_regs:
        first_dt = full_first_calv_by_reg.get(reg)
        if pd.isna(first_dt):
            continue
        days_to_calv = int((pd.Timestamp(first_dt).normalize() - as_of_ts).days)
        if 0 <= days_to_calv <= gest_days:
            rows.append(
                {
                    "reg_s": reg,
                    "group_code": "Нетели",
                    "basis": "future_first_calving",
                    "lact_cat": 0,
                    "dim": pd.NA,
                    "age_days": pd.NA,
                    "semen": "trad",
                    "days_to_calv": days_to_calv,
                    "cow_open_weight": 0.0,
                    "cow_preg_lact_weight": 0.0,
                    "cow_preg_dry_weight": 0.0,
                    "heifer_preg_weight": 1.0,
                    "heifer_open_weight": 0.0,
                    "last_calving": pd.NaT,
                    "last_dry": pd.NaT,
                    "first_calving": pd.Timestamp(first_dt).normalize(),
                }
            )

    if warmstart_from_services and bool(ins_params.get("enable_heifer_service_warmstart", False)) and last_service_heif:
        for reg, s_date in last_service_heif.items():
            if reg in disposed_regs or reg in p_regs or reg in future_neteli_regs:
                continue
            if (reg in cow_like_regs) and (not _is_pre_first_calving(reg)):
                continue
            if pd.isna(s_date):
                continue
            bull = last_service_bull_heif.get(reg, "") or ""
            semen = semen_by_bull.get(bull, "trad") if bull else "trad"
            s_date = pd.Timestamp(s_date).normalize()
            days_to_calv = int(gest_days - (as_of_ts - s_date).days)
            if days_to_calv < 0 and days_to_calv >= -OVERDUE_CLAMP_DAYS:
                days_to_calv = 0
            if not (0 <= days_to_calv <= gest_days):
                continue
            add = _clamp(p_conc_heif_base * _month_factor_value(heifer_month_factors, s_date), 0.03, 0.35)
            rows.append(
                {
                    "reg_s": reg,
                    "group_code": "Нетели",
                    "basis": "heifer_service_warmstart",
                    "lact_cat": 0,
                    "dim": pd.NA,
                    "age_days": pd.NA,
                    "semen": semen,
                    "days_to_calv": days_to_calv,
                    "cow_open_weight": 0.0,
                    "cow_preg_lact_weight": 0.0,
                    "cow_preg_dry_weight": 0.0,
                    "heifer_preg_weight": add,
                    "heifer_open_weight": 0.0,
                    "last_calving": pd.NaT,
                    "last_dry": pd.NaT,
                    "first_calving": pd.Timestamp(full_first_calv_by_reg.get(reg)).normalize() if pd.notna(full_first_calv_by_reg.get(reg)) else pd.NaT,
                }
            )

    calf_births = _extract_calf_births(calv, as_of_ts)
    if not calf_births.empty:
        calf_births["age_days"] = (as_of_ts - pd.to_datetime(calf_births["birth_dt"], errors="coerce")).dt.days
        calf_births = calf_births[
            calf_births["reg_s"].notna()
            & (calf_births["reg_s"] != "")
            & (~calf_births["reg_s"].isin(disposed_regs))
        ].copy()
        excluded_regs = set(cows["reg_s"].astype(str).tolist()) | set(p_regs) | set(future_neteli_regs)
        calf_births = calf_births[~calf_births["reg_s"].astype(str).isin(excluded_regs)].copy()
        female_births = calf_births[
            (calf_births["sex_norm"] == "F")
            & (calf_births["age_days"] >= 0)
            & (calf_births["age_days"] < MAX_AGE_DAYS)
        ].copy()
        if not female_births.empty:
            female_births = (
                female_births.sort_values(["reg_s", "birth_dt"], kind="mergesort")
                .drop_duplicates(subset=["reg_s"], keep="last")
            )
            for rr in female_births.itertuples(index=False):
                age_days = int(rr.age_days)
                if age_days < 90:
                    group_code = "Тёлки 0–3 мес"
                elif age_days < 270:
                    group_code = "Тёлки 3–8 мес"
                else:
                    group_code = "Тёлки ≥9 мес"
                rows.append(
                    {
                        "reg_s": str(rr.reg_s),
                        "group_code": group_code,
                        "basis": "birth_age",
                        "lact_cat": 0,
                        "dim": pd.NA,
                        "age_days": age_days,
                        "semen": "trad",
                        "days_to_calv": pd.NA,
                        "cow_open_weight": 0.0,
                        "cow_preg_lact_weight": 0.0,
                        "cow_preg_dry_weight": 0.0,
                        "heifer_preg_weight": 0.0,
                        "heifer_open_weight": 1.0,
                        "bull_weight": 0.0,
                        "last_calving": pd.NaT,
                        "last_dry": pd.NaT,
                        "first_calving": pd.Timestamp(full_first_calv_by_reg.get(str(rr.reg_s))).normalize() if pd.notna(full_first_calv_by_reg.get(str(rr.reg_s))) else pd.NaT,
                    }
                )

        male_births = calf_births[
            (calf_births["sex_norm"] == "M")
            & (calf_births["age_days"] >= 0)
            & (calf_births["age_days"] < 61)
        ].copy()
        if not male_births.empty:
            male_births = (
                male_births.sort_values(["reg_s", "birth_dt"], kind="mergesort")
                .drop_duplicates(subset=["reg_s"], keep="last")
            )
            for rr in male_births.itertuples(index=False):
                rows.append(
                    {
                        "reg_s": str(rr.reg_s),
                        "group_code": "Бычки 0–2 мес",
                        "basis": "birth_age",
                        "lact_cat": 0,
                        "dim": pd.NA,
                        "age_days": int(rr.age_days),
                        "semen": "trad",
                        "days_to_calv": pd.NA,
                        "cow_open_weight": 0.0,
                        "cow_preg_lact_weight": 0.0,
                        "cow_preg_dry_weight": 0.0,
                        "heifer_preg_weight": 0.0,
                        "heifer_open_weight": 0.0,
                        "bull_weight": 1.0,
                        "last_calving": pd.NaT,
                        "last_dry": pd.NaT,
                        "first_calving": pd.NaT,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "reg_s",
                "group_code",
                "basis",
                "lact_cat",
                "dim",
                "age_days",
                "semen",
                "days_to_calv",
                "cow_open_weight",
                "cow_preg_lact_weight",
                "cow_preg_dry_weight",
                "heifer_preg_weight",
                "heifer_open_weight",
                "bull_weight",
                "last_calving",
                "last_dry",
                "first_calving",
            ]
        )

    out["reg_s"] = out["reg_s"].astype(str)
    return out.sort_values(["group_code", "reg_s"], kind="mergesort").reset_index(drop=True)


def _state_from_animal_asof_snapshot(snapshot: pd.DataFrame, gest_days: int) -> HerdState:
    state = init_empty_state(gest_days)
    if not isinstance(snapshot, pd.DataFrame) or snapshot.empty:
        return state

    work = snapshot.copy()
    work["lact_cat"] = pd.to_numeric(work.get("lact_cat"), errors="coerce").fillna(0).astype(int)
    work["dim"] = pd.to_numeric(work.get("dim"), errors="coerce")
    work["age_days"] = pd.to_numeric(work.get("age_days"), errors="coerce")
    work["days_to_calv"] = pd.to_numeric(work.get("days_to_calv"), errors="coerce")
    work["bull_weight"] = pd.to_numeric(work.get("bull_weight"), errors="coerce").fillna(0.0)

    for rr in work.itertuples(index=False):
        lact_cat = int(getattr(rr, "lact_cat", 0) or 0)
        if lact_cat > 0:
            dim = int(_clamp(float(getattr(rr, "dim", 0) or 0), 0.0, float(MAX_DIM)))
            semen = str(getattr(rr, "semen", "trad") or "trad")
            if semen not in {"trad", "sex"}:
                semen = "trad"
            days_to_calv = int(_clamp(float(getattr(rr, "days_to_calv", 0) or 0), 0.0, float(gest_days)))
            open_w = float(getattr(rr, "cow_open_weight", 0.0) or 0.0)
            preg_l_w = float(getattr(rr, "cow_preg_lact_weight", 0.0) or 0.0)
            preg_d_w = float(getattr(rr, "cow_preg_dry_weight", 0.0) or 0.0)
            if open_w > 0:
                state.open_dim[lact_cat][dim] += open_w
            if preg_l_w > 0:
                state.preg_lact[(lact_cat, semen)][days_to_calv] += preg_l_w
            if preg_d_w > 0:
                state.preg_dry[(lact_cat, semen)][days_to_calv] += preg_d_w
            continue

        heifer_preg_w = float(getattr(rr, "heifer_preg_weight", 0.0) or 0.0)
        heifer_open_w = float(getattr(rr, "heifer_open_weight", 0.0) or 0.0)
        semen = str(getattr(rr, "semen", "trad") or "trad")
        if semen not in {"trad", "sex"}:
            semen = "trad"
        if heifer_preg_w > 0:
            days_to_calv = int(_clamp(float(getattr(rr, "days_to_calv", 0) or 0), 0.0, float(gest_days)))
            state.heifer_preg[semen][days_to_calv] += heifer_preg_w
        if heifer_open_w > 0:
            age_days = int(_clamp(float(getattr(rr, "age_days", 0) or 0), 0.0, float(MAX_AGE_DAYS)))
            state.heifer_age[age_days] += heifer_open_w
        bull_w = float(getattr(rr, "bull_weight", 0.0) or 0.0)
        if bull_w > 0:
            age_days = int(_clamp(float(getattr(rr, "age_days", 0) or 0), 0.0, float(BULL_AGE_MAX)))
            state.bull_age[age_days] += bull_w

    return state


def animal_asof_snapshot_group_totals(snapshot: pd.DataFrame) -> Dict[str, float]:
    base = {
        "Дойные коровы": 0.0,
        "Сухостойные коровы": 0.0,
        "Тёлки 0–3 мес": 0.0,
        "Бычки 0–2 мес": 0.0,
        "Тёлки 3–8 мес": 0.0,
        "Тёлки ≥9 мес": 0.0,
        "Нетели": 0.0,
    }
    if not isinstance(snapshot, pd.DataFrame) or snapshot.empty:
        return base

    work = snapshot.copy()
    work["cow_open_weight"] = pd.to_numeric(work.get("cow_open_weight"), errors="coerce").fillna(0.0)
    work["cow_preg_lact_weight"] = pd.to_numeric(work.get("cow_preg_lact_weight"), errors="coerce").fillna(0.0)
    work["cow_preg_dry_weight"] = pd.to_numeric(work.get("cow_preg_dry_weight"), errors="coerce").fillna(0.0)
    work["heifer_preg_weight"] = pd.to_numeric(work.get("heifer_preg_weight"), errors="coerce").fillna(0.0)
    work["heifer_open_weight"] = pd.to_numeric(work.get("heifer_open_weight"), errors="coerce").fillna(0.0)
    work["bull_weight"] = pd.to_numeric(work.get("bull_weight"), errors="coerce").fillna(0.0)
    work["age_days"] = pd.to_numeric(work.get("age_days"), errors="coerce")

    base["Дойные коровы"] = float((work["cow_open_weight"] + work["cow_preg_lact_weight"]).sum())
    base["Сухостойные коровы"] = float(work["cow_preg_dry_weight"].sum())
    base["Нетели"] = float(work["heifer_preg_weight"].sum())
    base["Тёлки 0–3 мес"] = float(work.loc[(work["age_days"] >= 0) & (work["age_days"] < 90), "heifer_open_weight"].sum())
    base["Бычки 0–2 мес"] = float(work.loc[(work["age_days"] >= 0) & (work["age_days"] < 61), "bull_weight"].sum())
    base["Тёлки 3–8 мес"] = float(work.loc[(work["age_days"] >= 90) & (work["age_days"] < 270), "heifer_open_weight"].sum())
    base["Тёлки ≥9 мес"] = float(work.loc[work["age_days"] >= 270, "heifer_open_weight"].sum())
    return base


def snapshot_groups_on_asof_from_tables(
    tables: Dict[str, pd.DataFrame],
    as_of: date,
    *,
    gest_days: int | None = None,
    dry_days: int | None = None,
    insemination_params: dict | None = None,
    semen_sex_ratios: Dict[str, SemenSexRatio] | None = None,
    warmstart_from_services: bool = False,
) -> Dict[str, float]:
    def _frame(name: str, columns: list[str]) -> pd.DataFrame:
        df = tables.get(name, pd.DataFrame()) if isinstance(tables, dict) else pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame()
        return df.reindex(columns=columns).copy()

    norm_tables = {
        "calv": _frame("calv", ["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date"]),
        "ins": _frame("ins", ["reg", "lact", "dim_age", "event_date", "bull", "result"]),
        "dry": _frame("dry", ["reg", "dim", "event_date"]),
        "disp": _frame("disp", ["reg", "event_date", "disposal_reason"]),
        "bulls": _frame("bulls", ["bull_code", "bull_type"]),
    }
    state = build_initial_state(
        norm_tables,
        as_of=as_of,
        gest_days=gest_days,
        dry_days=dry_days,
        insemination_params=insemination_params,
        warmstart_from_services=warmstart_from_services,
        semen_sex_ratios=semen_sex_ratios,
    )
    return _state_group_snapshot(state)

import pandas as pd
import numpy as np

def build_early_realization_plan(
    df: pd.DataFrame,
    *,
    lead_neteli_months: int = 2,
    lead_heifer9_months: int = 4,
) -> pd.DataFrame:
    cols = list(df.columns)

    def row(name: str) -> pd.Series:
        if name in df.index:
            return pd.to_numeric(df.loc[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=cols)

    over_doy = row("Переполнение: Дойные коровы")
    calv_from_neteli = row("Ожидаемый отёл, из них нетелей")

    stock_neteli = row("Нетели")
    stock_h9 = row("Тёлки ≥9 мес")

    plan_neteli = pd.Series(0.0, index=cols)
    plan_h9 = pd.Series(0.0, index=cols)
    plan_cows = pd.Series(0.0, index=cols)

    for i, m in enumerate(cols):
        need = float(over_doy[m])
        if need <= 0:
            continue

        j = i - lead_neteli_months
        if j >= 0 and need > 0:
            m_sell = cols[j]

            cap_by_flow = float(calv_from_neteli[m])
            cap_by_stock = max(0.0, float(stock_neteli[m_sell]) - float(plan_neteli[m_sell]))

            add = min(need, cap_by_flow, cap_by_stock)
            if add > 0:
                plan_neteli[m_sell] += add
                need -= add

                                                                        
        k = i - lead_heifer9_months
        if k >= 0 and need > 0:
            m_sell = cols[k]

            cap_by_stock = max(0.0, float(stock_h9[m_sell]) - float(plan_h9[m_sell]))
            add = min(need, cap_by_stock)
            if add > 0:
                plan_h9[m_sell] += add
                need -= add

        if need > 0:
            plan_cows[m] += need

    out = pd.DataFrame(index=[
        "План реализации (ранний): нетели",
        "План реализации (ранний): тёлки ≥9 мес",
        "План реализации (ранний): коровы",
    ], columns=cols)

    out.loc["План реализации (ранний): нетели"] = plan_neteli.round(1)
    out.loc["План реализации (ранний): тёлки ≥9 мес"] = plan_h9.round(1)
    out.loc["План реализации (ранний): коровы"] = plan_cows.round(1)

    return out

                                                              
                                                              

def simulate_to_target(
    state: HerdState,
    *,
    start: date,
    target: date,
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    params: dict,
) -> Tuple[HerdState, Dict[str, float]]:

    start_ts = pd.Timestamp(start).normalize()
    target_ts = pd.Timestamp(target).normalize()

    end_sim_ts = pd.Timestamp(end_of_month(target_ts.date())).normalize()

    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])
    cp = params["CONCEPTION_PARAMS"]
    disp_params = params["DISPOSAL_PARAMS"]
    annual_disp = float(params["ANNUAL_DISPOSAL_RATE"])
    heifer_precalving_disp = float(params.get("HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE", annual_disp))
    bull_calf_exit = float(params.get("BULL_CALF_DAILY_EXIT_RATE", 0.0))
    ins_p = params["INSEMINATION_PARAMS"]

    target_month = (int(target_ts.year), int(target_ts.month))
    calv_total = 0.0
    calv_cows = 0.0
    calv_heifers = 0.0
    exp_bulls = 0.0
    exp_heifers = 0.0

    meta: Dict[str, float] = {
        "cow_doses_total": 0.0, "cow_doses_sex": 0.0, "cow_doses_trad": 0.0,
        "heifer_doses_total": 0.0, "heifer_doses_sex": 0.0, "heifer_doses_trad": 0.0,
        "sell_cows": 0.0, "sell_heifers": 0.0, "sell_neteli": 0.0,
        "over_doy": 0.0, "over_dry": 0.0, "over_h0": 0.0, "over_h38": 0.0, "over_h9": 0.0, "over_neteli": 0.0,
        "start_doy": 0.0, "start_dry": 0.0, "start_h0": 0.0, "start_h38": 0.0, "start_h9": 0.0, "start_neteli": 0.0,
        "start_b0": 0.0,
        "flow_birth_female_total": 0.0, "flow_birth_male_total": 0.0,
        "flow_h0_to_h38": 0.0, "flow_h38_to_h9": 0.0, "flow_b0_to_out": 0.0,
        "flow_doy_to_dry": 0.0, "flow_dry_to_doy": 0.0, "flow_neteli_to_doy": 0.0, "flow_h9_to_neteli": 0.0,
        "disp_doy_total": 0.0, "disp_dry_total": 0.0,
        "disp_h9_total": 0.0, "disp_neteli_total": 0.0,
        "preg_loss_cows_total": 0.0, "preg_loss_heifers_total": 0.0,
        "sell_cows_total": 0.0, "sell_cows_doy_total": 0.0, "sell_cows_dry_total": 0.0,
        "sell_heifers_total": 0.0, "sell_heifers_h0_total": 0.0, "sell_heifers_h38_total": 0.0, "sell_heifers_h9_total": 0.0,
        "sell_neteli_total": 0.0,
    }
    start_snapshot = _state_group_snapshot(state)
    meta["start_doy"] = float(start_snapshot["Дойные коровы"])
    meta["start_dry"] = float(start_snapshot["Сухостойные коровы"])
    meta["start_h0"] = float(start_snapshot["Тёлки 0–3 мес"])
    meta["start_h38"] = float(start_snapshot["Тёлки 3–8 мес"])
    meta["start_h9"] = float(start_snapshot["Тёлки ≥9 мес"])
    meta["start_neteli"] = float(start_snapshot["Нетели"])
    meta["start_b0"] = float(start_snapshot["Бычки 0–2 мес"])

    def _process_bucket0_for_day(curr_day_ts: pd.Timestamp) -> None:
        nonlocal calv_total, calv_cows, calv_heifers, exp_bulls, exp_heifers

        curr_month = (int(curr_day_ts.year), int(curr_day_ts.month))

        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                born_lact = float(state.preg_lact[(l, semen)][0])
                if born_lact > 0:
                    state.preg_lact[(l, semen)][0] = 0.0

                    if curr_month == target_month:
                        calv_total += born_lact
                        calv_cows += born_lact
                        sr = semen_sex_ratios[semen]
                        exp_bulls += born_lact * float(sr.bull_share)
                        exp_heifers += born_lact * float(sr.heifer_share)
                    meta["flow_dry_to_doy"] += born_lact

                    l2 = min(4, l + 1)
                    state.open_dim[l2][0] += born_lact

                    sr = semen_sex_ratios[semen]
                    state.heifer_age[0] += born_lact * float(sr.heifer_share)
                    state.bull_age[0] += born_lact * float(sr.bull_share)
                    meta["flow_birth_female_total"] += born_lact * float(sr.heifer_share)
                    meta["flow_birth_male_total"] += born_lact * float(sr.bull_share)

                born = float(state.preg_dry[(l, semen)][0])
                if born > 0:
                    state.preg_dry[(l, semen)][0] = 0.0

                    if curr_month == target_month:
                        calv_total += born
                        calv_cows += born
                        sr = semen_sex_ratios[semen]
                        exp_bulls += born * float(sr.bull_share)
                        exp_heifers += born * float(sr.heifer_share)
                    meta["flow_dry_to_doy"] += born

                    l2 = min(4, l + 1)
                    state.open_dim[l2][0] += born

                    sr = semen_sex_ratios[semen]
                    state.heifer_age[0] += born * float(sr.heifer_share)
                    state.bull_age[0] += born * float(sr.bull_share)
                    meta["flow_birth_female_total"] += born * float(sr.heifer_share)
                    meta["flow_birth_male_total"] += born * float(sr.bull_share)

        for semen in ("trad", "sex"):
            born = float(state.heifer_preg[semen][0])
            if born > 0:
                state.heifer_preg[semen][0] = 0.0

                if curr_month == target_month:
                    calv_total += born
                    calv_heifers += born
                    sr = semen_sex_ratios[semen]
                    exp_bulls += born * float(sr.bull_share)
                    exp_heifers += born * float(sr.heifer_share)
                meta["flow_neteli_to_doy"] += born

                state.open_dim[1][0] += born
                sr = semen_sex_ratios[semen]
                state.heifer_age[0] += born * float(sr.heifer_share)
                state.bull_age[0] += born * float(sr.bull_share)
                meta["flow_birth_female_total"] += born * float(sr.heifer_share)
                meta["flow_birth_male_total"] += born * float(sr.bull_share)

    _process_bucket0_for_day(start_ts)
                                                                     
    if params.get("APPLY_CAPACITY", True) and pd.Timestamp(start).normalize() == pd.Timestamp(end_of_month(start)).normalize():
        sold0 = _apply_capacity_month_end(
            state,
            gest_days=gest_days,
            dry_days=dry_days,
            cap_norm=params.get("HERD_CAPACITY_NORM"),
        )
        if (start.year, start.month) == target_month:
            meta["sell_cows"] += float(sold0["sell_cows"])
            meta["sell_heifers"] += float(sold0["sell_heifers"])
            meta["sell_neteli"] += float(sold0["sell_neteli"])
            meta["over_doy"] += float(sold0["over_doy"])
            meta["over_dry"] += float(sold0["over_dry"])
            meta["over_h0"] += float(sold0["over_h0"])
            meta["over_h38"] += float(sold0["over_h38"])
            meta["over_h9"] += float(sold0["over_h9"])
            meta["over_neteli"] += float(sold0["over_neteli"])
        meta["sell_cows_total"] += float(sold0.get("sell_cows", 0.0))
        meta["sell_cows_doy_total"] += float(sold0.get("sell_cows_doy", 0.0))
        meta["sell_cows_dry_total"] += float(sold0.get("sell_cows_dry", 0.0))
        meta["sell_heifers_total"] += float(sold0.get("sell_heifers", 0.0))
        meta["sell_heifers_h0_total"] += float(sold0.get("sell_heifers_h0", 0.0))
        meta["sell_heifers_h38_total"] += float(sold0.get("sell_heifers_h38", 0.0))
        meta["sell_heifers_h9_total"] += float(sold0.get("sell_heifers_h9", 0.0))
        meta["sell_neteli_total"] += float(sold0.get("sell_neteli", 0.0))

    p_disp_day_base = 1.0 - (1.0 - annual_disp) ** (1.0 / 365.0)
    p_disp_day_heifer = 1.0 - (1.0 - heifer_precalving_disp) ** (1.0 / 365.0)
    cow_preg_loss_rate = float(max(0.0, min(0.5, ins_p.get("cow_pregnancy_loss_rate", 0.0))))
    heifer_preg_loss_rate = float(max(0.0, min(0.5, ins_p.get("heifer_pregnancy_loss_rate", 0.0))))
    p_preg_loss_day_cow = 1.0 - (1.0 - cow_preg_loss_rate) ** (1.0 / max(1.0, float(gest_days)))
    p_preg_loss_day_heifer = 1.0 - (1.0 - heifer_preg_loss_rate) ** (1.0 / max(1.0, float(gest_days)))
    disp_shape = build_disposal_shape(disp_params)

    by_lact = disp_params.get("by_lact", {})
    total_n = float(disp_params.get("overall", {}).get("n", 1) or 1)
    shares = {l: (float(by_lact.get(l, {}).get("n", 0) or 0) / total_n) for l in (1, 2, 3, 4)}
    avg_share = sum(shares.values()) / 4.0 if sum(shares.values()) > 0 else 1.0
    w = {l: (shares.get(l, avg_share) / avg_share) for l in (1, 2, 3, 4)}

    cow_trad_share = float(semen_shares["cow_trad"])
    cow_sex_share = float(semen_shares["cow_sex"])
    heif_trad_share = float(semen_shares["heifer_trad"])
    heif_sex_share = float(semen_shares["heifer_sex"])
    cow_month_factors = _normalize_month_factor_map(
        ins_p.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        ins_p.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    snapshot: HerdState | None = None
    if target_ts <= start_ts:
        snapshot = _copy_state(state)

    idx_dry = min(dry_days, gest_days)

    day = start_ts

    while day < end_sim_ts:
        day = (day + pd.Timedelta(days=1)).normalize()

        if len(state.heifer_age) > 89:
            meta["flow_h0_to_h38"] += float(state.heifer_age[89])
        if len(state.heifer_age) > 269:
            meta["flow_h38_to_h9"] += float(state.heifer_age[269])
        if len(state.bull_age) > 60:
            meta["flow_b0_to_out"] += float(state.bull_age[60])

        for l in (1, 2, 3, 4):
            state.open_dim[l] = shift_right(state.open_dim[l])
        state.heifer_age = shift_right(state.heifer_age)
        state.bull_age = shift_right(state.bull_age)

        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                state.preg_lact[(l, semen)] = shift_left(state.preg_lact[(l, semen)])
                state.preg_dry[(l, semen)] = shift_left(state.preg_dry[(l, semen)])
        for semen in ("trad", "sex"):
            state.heifer_preg[semen] = shift_left(state.heifer_preg[semen])

        for l in (1, 2, 3, 4):
            for semen in ("trad", "sex"):
                move = float(state.preg_lact[(l, semen)][idx_dry])
                if move > 0:
                    state.preg_lact[(l, semen)][idx_dry] = 0.0
                    state.preg_dry[(l, semen)][idx_dry] += move
                    meta["flow_doy_to_dry"] += move

        _process_bucket0_for_day(day)

        for l in (1, 2, 3, 4):
            first_ai = float(ins_p["cow_first_ai_dim_by_lact"].get(l, 70.0))
            spc = float(ins_p["cow_services_per_conception"])
            interval_raw = float(ins_p["cow_ai_interval_days"])
            mean_target = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))
            month_factor = _month_factor_value(cow_month_factors, day)

            interval = _effective_ai_interval_days(interval_raw, mean_target, first_ai, spc)
            p_service = 1.0 / max(1.0, interval)
            p_conc = _clamp((1.0 / max(1e-9, spc)) * month_factor, 0.05, 0.95)

            open_arr = state.open_dim[l]
            first_i = int(_clamp(first_ai, 0.0, float(MAX_DIM)))
            if first_i >= len(open_arr):
                continue

            eligible = open_arr.copy()
            eligible[:first_i] = 0.0

            services_by_dim = eligible * p_service
            services_total = float(services_by_dim.sum())
            if services_total <= 0:
                continue

            conceived_by_dim = services_by_dim * p_conc
            conceived_total = float(conceived_by_dim.sum())
            if conceived_total <= 0:
                continue

            state.open_dim[l] = np.maximum(0.0, open_arr - conceived_by_dim)

            state.preg_lact[(l, "sex")][gest_days] += services_total * cow_sex_share * p_conc
            state.preg_lact[(l, "trad")][gest_days] += services_total * cow_trad_share * p_conc

            if (int(day.year), int(day.month)) == target_month:
                meta["cow_doses_total"] += services_total
                meta["cow_doses_sex"] += services_total * cow_sex_share
                meta["cow_doses_trad"] += services_total * cow_trad_share

        first_ai_age = float(ins_p["heifer_first_ai_age_days"])
        spc_h = float(ins_p["heifer_services_per_conception"])
        interval_raw_h = float(ins_p["heifer_ai_interval_days"])
        mean_target_h = float(cp["avg_heifer_age_days"])
        month_factor_h = _month_factor_value(heifer_month_factors, day)

        interval_h = _effective_ai_interval_days(interval_raw_h, mean_target_h, first_ai_age, spc_h)
        p_service_h = 1.0 / max(1.0, interval_h)
        p_conc_h = _clamp((1.0 / max(1e-9, spc_h)) * month_factor_h, 0.05, 0.95)

        first_h = int(_clamp(first_ai_age, 0.0, float(MAX_AGE_DAYS)))
        if first_h < len(state.heifer_age):
            eligible_h = state.heifer_age.copy()
            eligible_h[:first_h] = 0.0
            # Do not inseminate the entire old tail of open heifers equally.
            # Keep the active window around the observed mean conception age.
            heifer_active_hi = int(
                _clamp(
                    max(float(first_ai_age) + 30.0, float(mean_target_h) + max(45.0, float(interval_h) * max(1.0, float(spc_h)))),
                    0.0,
                    float(MAX_AGE_DAYS),
                )
            )
            if heifer_active_hi + 1 < len(eligible_h):
                eligible_h[heifer_active_hi + 1 :] = 0.0

            services_by_age = eligible_h * p_service_h
            services_total_h = float(services_by_age.sum())
            if services_total_h > 0:
                conceived_by_age = services_by_age * p_conc_h
                conceived_total_h = float(conceived_by_age.sum())
                if conceived_total_h > 0:
                    state.heifer_age = np.maximum(0.0, state.heifer_age - conceived_by_age)
                    state.heifer_preg["sex"][gest_days] += services_total_h * heif_sex_share * p_conc_h
                    state.heifer_preg["trad"][gest_days] += services_total_h * heif_trad_share * p_conc_h
                    meta["flow_h9_to_neteli"] += conceived_total_h

                if (int(day.year), int(day.month)) == target_month:
                    meta["heifer_doses_total"] += services_total_h
                    meta["heifer_doses_sex"] += services_total_h * heif_sex_share
                    meta["heifer_doses_trad"] += services_total_h * heif_trad_share

        heifer_disp_base = max(0.0, min(0.02, float(p_disp_day_heifer)))
        if heifer_disp_base > 0.0:
            if len(state.heifer_age) > 270:
                h9_before = state.heifer_age[270:].copy()
                state.heifer_age[270:] = h9_before * (1.0 - heifer_disp_base)
                meta["disp_h9_total"] += float((h9_before - state.heifer_age[270:]).sum())
            for semen in ("trad", "sex"):
                neteli_before = state.heifer_preg[semen].copy()
                state.heifer_preg[semen] = neteli_before * (1.0 - heifer_disp_base)
                meta["disp_neteli_total"] += float((neteli_before - state.heifer_preg[semen]).sum())

        bull_disp_base = max(0.0, min(0.8, float(bull_calf_exit)))
        if bull_disp_base > 0.0 and len(state.bull_age) > 0:
            bull_before = state.bull_age[:61].copy()
            state.bull_age[:61] = bull_before * (1.0 - bull_disp_base)
            meta["flow_b0_to_out"] += float((bull_before - state.bull_age[:61]).sum())

        if p_preg_loss_day_cow > 0.0:
            for l in (1, 2, 3, 4):
                mean_conc = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))
                conc0 = int(round(mean_conc))
                idx = np.arange(gest_days + 1, dtype=int)
                gest_age = (gest_days - idx).astype(int)
                est_dim = np.clip(conc0 + gest_age, 0, MAX_DIM).astype(int)
                for semen in ("trad", "sex"):
                    preg_l_before = state.preg_lact[(l, semen)].copy()
                    preg_d_before = state.preg_dry[(l, semen)].copy()
                    lost_l = preg_l_before * p_preg_loss_day_cow
                    lost_d = preg_d_before * p_preg_loss_day_cow
                    if float(lost_l.sum()) > 0.0:
                        state.preg_lact[(l, semen)] = preg_l_before - lost_l
                        np.add.at(state.open_dim[l], est_dim, lost_l)
                        meta["preg_loss_cows_total"] += float(lost_l.sum())
                    if float(lost_d.sum()) > 0.0:
                        state.preg_dry[(l, semen)] = preg_d_before - lost_d
                        np.add.at(state.open_dim[l], est_dim, lost_d)
                        meta["preg_loss_cows_total"] += float(lost_d.sum())

        if p_preg_loss_day_heifer > 0.0:
            base_heifer_age = int(round(float(cp["avg_heifer_age_days"])))
            idx = np.arange(gest_days + 1, dtype=int)
            gest_age = (gest_days - idx).astype(int)
            est_age = np.clip(base_heifer_age + gest_age, 0, MAX_AGE_DAYS).astype(int)
            for semen in ("trad", "sex"):
                preg_h_before = state.heifer_preg[semen].copy()
                lost_h = preg_h_before * p_preg_loss_day_heifer
                if float(lost_h.sum()) <= 0.0:
                    continue
                state.heifer_preg[semen] = preg_h_before - lost_h
                np.add.at(state.heifer_age, est_age, lost_h)
                meta["preg_loss_heifers_total"] += float(lost_h.sum())

        for l in (1, 2, 3, 4):
            base = float(p_disp_day_base * w[l])
            base = max(0.0, min(0.02, base))

            haz_open = np.clip(base * disp_shape[l], 0.0, 0.05)
            open_before = state.open_dim[l].copy()
            state.open_dim[l] = open_before * (1.0 - haz_open)
            meta["disp_doy_total"] += float((open_before - state.open_dim[l]).sum())

            mean_conc = float(cp["avg_cow_dim_by_lact"].get(l, cp["avg_cow_dim_global"]))
            conc0 = int(round(mean_conc))
            idx = np.arange(gest_days + 1, dtype=int)
            gest_age = (gest_days - idx).astype(int)
            est_dim = np.clip(conc0 + gest_age, 0, MAX_DIM)

            haz_preg = np.clip(base * disp_shape[l][est_dim], 0.0, 0.05)
            for semen in ("trad", "sex"):
                preg_l_before = state.preg_lact[(l, semen)].copy()
                preg_d_before = state.preg_dry[(l, semen)].copy()
                state.preg_lact[(l, semen)] = preg_l_before * (1.0 - haz_preg)
                state.preg_dry[(l, semen)] = preg_d_before * (1.0 - haz_preg)
                meta["disp_doy_total"] += float((preg_l_before - state.preg_lact[(l, semen)]).sum())
                meta["disp_dry_total"] += float((preg_d_before - state.preg_dry[(l, semen)]).sum())

        day_eom = pd.Timestamp(end_of_month(day.date())).normalize()
        if params.get("APPLY_CAPACITY", True) and day == day_eom:
            sold = _apply_capacity_month_end(
                state,
                gest_days=gest_days,
                dry_days=dry_days,
                cap_norm=params.get("HERD_CAPACITY_NORM"),
            )
            if (int(day.year), int(day.month)) == target_month:
                meta["sell_cows"] += float(sold["sell_cows"])
                meta["sell_heifers"] += float(sold["sell_heifers"])
                meta["sell_neteli"] += float(sold["sell_neteli"])
                meta["over_doy"] += float(sold["over_doy"])
                meta["over_dry"] += float(sold["over_dry"])
                meta["over_h0"] += float(sold["over_h0"])
                meta["over_h38"] += float(sold["over_h38"])
                meta["over_h9"] += float(sold["over_h9"])
                meta["over_neteli"] += float(sold["over_neteli"])
            meta["sell_cows_total"] += float(sold.get("sell_cows", 0.0))
            meta["sell_cows_doy_total"] += float(sold.get("sell_cows_doy", 0.0))
            meta["sell_cows_dry_total"] += float(sold.get("sell_cows_dry", 0.0))
            meta["sell_heifers_total"] += float(sold.get("sell_heifers", 0.0))
            meta["sell_heifers_h0_total"] += float(sold.get("sell_heifers_h0", 0.0))
            meta["sell_heifers_h38_total"] += float(sold.get("sell_heifers_h38", 0.0))
            meta["sell_heifers_h9_total"] += float(sold.get("sell_heifers_h9", 0.0))
            meta["sell_neteli_total"] += float(sold.get("sell_neteli", 0.0))

        if day == target_ts:
            snapshot = _copy_state(state)

    if snapshot is None:
        snapshot = _copy_state(state)

    meta.update({
        "calv_total": float(calv_total),
        "calv_cows": float(calv_cows),
        "calv_heifers": float(calv_heifers),
        "exp_bulls": float(exp_bulls),
        "exp_heifers": float(exp_heifers),
    })
    return snapshot, meta


                                                              
                                                              

def compute_forecast_dynamic_from_db(
    target_date: date,
    overrides: dict | None = None,
    as_of_date: date | None = None,
) -> Dict[str, float]:
    import pandas as pd
    from datetime import date, datetime

    def _as_ts(x):
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            return x.normalize()
        if isinstance(x, datetime):
            return pd.Timestamp(x).normalize()
        if isinstance(x, date):
            return pd.Timestamp(x)
        return pd.Timestamp(x).normalize()

    tables = load_tables()
    base = _as_ts(latest_data_date(tables))
    target_date = _as_ts(target_date)

    if as_of_date is None:
        start = min(base, target_date)
    else:
        as_of_ts = _as_ts(as_of_date)
        if as_of_ts is None or pd.isna(as_of_ts):
            raise ValueError(f"as_of_date is invalid: {as_of_date!r}")
        start = min(min(as_of_ts, base), target_date)

    ov = dict(overrides or {})

    if "gestation_days" in ov and "GESTATION_DAYS" not in ov:
        ov["GESTATION_DAYS"] = ov["gestation_days"]
    if "dry_days" in ov and "DRY_DAYS" not in ov:
        ov["DRY_DAYS"] = ov["dry_days"]
    if "annual_disposal_rate" in ov and "ANNUAL_DISPOSAL_RATE" not in ov:
        ov["ANNUAL_DISPOSAL_RATE"] = ov["annual_disposal_rate"]
    if (
        "heifer_precalving_annual_disposal_rate" in ov
        and "HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE" not in ov
    ):
        ov["HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE"] = ov["heifer_precalving_annual_disposal_rate"]
    if "bull_calf_daily_exit_rate" in ov and "BULL_CALF_DAILY_EXIT_RATE" not in ov:
        ov["BULL_CALF_DAILY_EXIT_RATE"] = ov["bull_calf_daily_exit_rate"]
    if "conception" in ov and "CONCEPTION_PARAMS" not in ov:
        ov["CONCEPTION_PARAMS"] = ov["conception"]
    if "insemination_params" in ov and "INSEMINATION_PARAMS" not in ov:
        ov["INSEMINATION_PARAMS"] = ov["insemination_params"]
    if "semen_usage" in ov and "SEMEN_USAGE_SHARES" not in ov:
        ov["SEMEN_USAGE_SHARES"] = ov["semen_usage"]
    if "SEMEN_SEX_RATIOS" in ov and "semen_sex_ratios" not in ov:
        ov["semen_sex_ratios"] = ov["SEMEN_SEX_RATIOS"]
    if "herd_capacity" in ov and "HERD_CAPACITY" not in ov:
        ov["HERD_CAPACITY"] = ov["herd_capacity"]

    params = _resolve_runtime_params(ov)
    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])

    semen_override = params.get("SEMEN_USAGE_SHARES")
    if isinstance(semen_override, dict) and semen_override:
        semen_shares = {
            "cow_trad": float(semen_override.get("cow_trad", 0.0)),
            "cow_sex": float(semen_override.get("cow_sex", 0.0)),
            "heifer_trad": float(semen_override.get("heifer_trad", 0.0)),
            "heifer_sex": float(semen_override.get("heifer_sex", 0.0)),
        }

        def _norm2(a: float, b: float) -> tuple[float, float]:
            s = max(1e-9, a + b)
            return a / s, b / s

        semen_shares["cow_trad"], semen_shares["cow_sex"] = _norm2(semen_shares["cow_trad"], semen_shares["cow_sex"])
        semen_shares["heifer_trad"], semen_shares["heifer_sex"] = _norm2(semen_shares["heifer_trad"], semen_shares["heifer_sex"])
    else:
        semen_shares = compute_semen_usage_from_db(tables)

    ssr_ov = ov.get("semen_sex_ratios")
    if isinstance(ssr_ov, dict) and ssr_ov:
        trad = ssr_ov.get("trad", {}) or {}
        sex = ssr_ov.get("sex", {}) or {}

        def _mk_ratio(d: dict, fallback_obj: SemenSexRatio) -> SemenSexRatio:
            bull_raw = d.get("bull_share")
            heif_raw = d.get("heifer_share")

                                                 
            if bull_raw is None and heif_raw is None:
                bull = float(fallback_obj.bull_share)
                heif = float(fallback_obj.heifer_share)
            elif bull_raw is None:
                heif = float(heif_raw)
                bull = 1.0 - heif
            elif heif_raw is None:
                bull = float(bull_raw)
                heif = 1.0 - bull
            else:
                bull = float(bull_raw)
                heif = float(heif_raw)

            bull = max(0.0, min(1.0, bull))
            heif = max(0.0, min(1.0, heif))
            s = max(1e-9, bull + heif)
            bull /= s
            heif /= s
            return SemenSexRatio(bull_share=bull, heifer_share=heif)

        semen_sex_ratios = {
            "trad": _mk_ratio(trad, _to_semen_ratio(SEMEN_SEX_RATIOS["trad"])),
            "sex": _mk_ratio(sex, _to_semen_ratio(SEMEN_SEX_RATIOS["sex"])),
        }
    else:
        semen_sex_ratios = compute_semen_sex_ratios_from_db(tables)

    warmstart_from_services = bool(ov.get("warmstart_from_services", True))

    state0 = build_initial_state(
        tables,
        as_of=start,
        gest_days=gest_days,
        dry_days=dry_days,
        insemination_params=params["INSEMINATION_PARAMS"],
        warmstart_from_services=warmstart_from_services,
        semen_sex_ratios=semen_sex_ratios,                                                        
    )

    state_at_target, meta = simulate_to_target(
        state0,
        start=start,
        target=target_date,
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        params=params,
    )

    cows_open = sum(state_at_target.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state_at_target.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state_at_target.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)
    neteli = float(state_at_target.heifer_preg["trad"].sum() + state_at_target.heifer_preg["sex"].sum())

    h0_3 = float(state_at_target.heifer_age[:90].sum())
    h3_8 = float(state_at_target.heifer_age[90:270].sum())
    h9p = float(state_at_target.heifer_age[270:].sum())
    b0_2 = float(state_at_target.bull_age[:61].sum())

    calv_total_f = float(meta.get("calv_total", 0.0) or 0.0)
    calv_cows_f = float(meta.get("calv_cows", 0.0) or 0.0)
    calv_heifers_f = float(meta.get("calv_heifers", 0.0) or 0.0)
    exp_bulls_f = float(meta.get("exp_bulls", 0.0) or 0.0)
    exp_heifers_f = float(meta.get("exp_heifers", 0.0) or 0.0)

    out = {
        "Дойные коровы": round(doy),
        "Сухостойные коровы": round(dry),

        "Тёлки 0–3 мес": round(h0_3, 1),
        "Тёлки 0–2 мес": round(h0_3, 1),

        "Бычки 0–2 мес": round(b0_2, 1),
        "Тёлки 3–8 мес": round(h3_8, 1),
        "Тёлки ≥9 мес": round(h9p, 1),
        "Нетели": round(neteli, 1),

        "Ожидаемый отёл, всего": round(calv_total_f, 1),
        "Ожидаемый отёл, из них коров": round(calv_cows_f, 1),
        "Ожидаемый отёл, из них нетелей": round(calv_heifers_f, 1),

        "Ожидаемые бычки": round(exp_bulls_f, 1),
        "Ожидаемые тёлочки": round(exp_heifers_f, 1),

        "К реализации: коровы": round(float(meta.get("sell_cows", 0.0)), 1),
        "К реализации: тёлки": round(float(meta.get("sell_heifers", 0.0)), 1),
        "К реализации: нетели": round(float(meta.get("sell_neteli", 0.0)), 1),

        "Переполнение: Дойные коровы": round(float(meta.get("over_doy", 0.0)), 1),
        "Переполнение: Сухостойные коровы": round(float(meta.get("over_dry", 0.0)), 1),
        "Переполнение: Тёлки 0–3 мес": round(float(meta.get("over_h0", 0.0)), 1),
        "Переполнение: Тёлки 3–8 мес": round(float(meta.get("over_h38", 0.0)), 1),
        "Переполнение: Тёлки 9–24 мес": round(float(meta.get("over_h9", 0.0)), 1),
        "Переполнение: Нетели": round(float(meta.get("over_neteli", 0.0)), 1),
    }

    _apply_expected_calving_prob_fallback_from_tables(
        out,
        tables,
        target_date,
        gest_days=gest_days,
        insemination_params=params["INSEMINATION_PARAMS"],
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        as_of_date=as_of_date,
    )
    _apply_current_month_observed_births_overlay(
        out,
        tables,
        target_date,
        start_ts=start,
        as_of_date=as_of_date,
    )
    _scale_birth_output(
        out,
        _recent_birth_bias_factor_from_tables(
            tables,
            target_date,
            as_of_date=as_of_date,
            overrides=ov,
            gest_days=gest_days,
            insemination_params=params["INSEMINATION_PARAMS"],
            semen_shares=semen_shares,
            semen_sex_ratios=semen_sex_ratios,
        ),
    )
    return out


def _normalize_input_tables(tables: Dict[str, pd.DataFrame] | None) -> Dict[str, pd.DataFrame]:
    src = tables or {}
    out: Dict[str, pd.DataFrame] = {}

    required_cols = {
        "calv": ["reg", "mother_reg", "birth_date", "sex", "event_type", "event_date"],
        "ins": ["reg", "lact", "dim_age", "event_date", "bull", "result"],
        "dry": ["reg", "dim", "event_date"],
        "disp": ["reg", "event_date", "disposal_reason"],
        "bulls": ["bull_code", "bull_type"],
    }

    for key, cols in required_cols.items():
        df = src.get(key)
        if not isinstance(df, pd.DataFrame):
            out[key] = pd.DataFrame(columns=cols)
            continue
        dfx = df.copy()
        for c in cols:
            if c not in dfx.columns:
                dfx[c] = pd.NA
        out[key] = dfx[cols].copy()

    return out


def _as_normalized_ts(x: Any) -> pd.Timestamp | None:
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x.normalize()
    if isinstance(x, datetime):
        return pd.Timestamp(x).normalize()
    if isinstance(x, date):
        return pd.Timestamp(x)
    return pd.Timestamp(x).normalize()


def _run_dynamic_simulation_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: date,
    overrides: dict | None = None,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    tables = _normalize_input_tables(tables)
    base = _as_normalized_ts(latest_data_date(tables))
    target_ts = _as_normalized_ts(target_date)

    if as_of_date is None:
        start = min(base, target_ts)
    else:
        as_of_ts = _as_normalized_ts(as_of_date)
        if as_of_ts is None or pd.isna(as_of_ts):
            raise ValueError(f"as_of_date is invalid: {as_of_date!r}")
        start = min(min(as_of_ts, base), target_ts)

    ov = dict(overrides or {})

    if "gestation_days" in ov and "GESTATION_DAYS" not in ov:
        ov["GESTATION_DAYS"] = ov["gestation_days"]
    if "dry_days" in ov and "DRY_DAYS" not in ov:
        ov["DRY_DAYS"] = ov["dry_days"]
    if "annual_disposal_rate" in ov and "ANNUAL_DISPOSAL_RATE" not in ov:
        ov["ANNUAL_DISPOSAL_RATE"] = ov["annual_disposal_rate"]
    if (
        "heifer_precalving_annual_disposal_rate" in ov
        and "HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE" not in ov
    ):
        ov["HEIFER_PRECALVING_ANNUAL_DISPOSAL_RATE"] = ov["heifer_precalving_annual_disposal_rate"]
    if "bull_calf_daily_exit_rate" in ov and "BULL_CALF_DAILY_EXIT_RATE" not in ov:
        ov["BULL_CALF_DAILY_EXIT_RATE"] = ov["bull_calf_daily_exit_rate"]
    if "conception" in ov and "CONCEPTION_PARAMS" not in ov:
        ov["CONCEPTION_PARAMS"] = ov["conception"]
    if "insemination_params" in ov and "INSEMINATION_PARAMS" not in ov:
        ov["INSEMINATION_PARAMS"] = ov["insemination_params"]
    if "semen_usage" in ov and "SEMEN_USAGE_SHARES" not in ov:
        ov["SEMEN_USAGE_SHARES"] = ov["semen_usage"]
    if "SEMEN_SEX_RATIOS" in ov and "semen_sex_ratios" not in ov:
        ov["semen_sex_ratios"] = ov["SEMEN_SEX_RATIOS"]
    if "herd_capacity" in ov and "HERD_CAPACITY" not in ov:
        ov["HERD_CAPACITY"] = ov["herd_capacity"]

    params = _resolve_runtime_params(ov)
    gest_days = int(params["GESTATION_DAYS"])
    dry_days = int(params["DRY_DAYS"])

    semen_override = params.get("SEMEN_USAGE_SHARES")
    if isinstance(semen_override, dict) and semen_override:
        semen_shares = {
            "cow_trad": float(semen_override.get("cow_trad", 0.0)),
            "cow_sex": float(semen_override.get("cow_sex", 0.0)),
            "heifer_trad": float(semen_override.get("heifer_trad", 0.0)),
            "heifer_sex": float(semen_override.get("heifer_sex", 0.0)),
        }

        def _norm2(a: float, b: float) -> tuple[float, float]:
            s = max(1e-9, a + b)
            return a / s, b / s

        semen_shares["cow_trad"], semen_shares["cow_sex"] = _norm2(semen_shares["cow_trad"], semen_shares["cow_sex"])
        semen_shares["heifer_trad"], semen_shares["heifer_sex"] = _norm2(semen_shares["heifer_trad"], semen_shares["heifer_sex"])
    else:
        semen_shares = compute_semen_usage_from_db(tables)

    ssr_ov = ov.get("semen_sex_ratios")
    if isinstance(ssr_ov, dict) and ssr_ov:
        trad = ssr_ov.get("trad", {}) or {}
        sex = ssr_ov.get("sex", {}) or {}

        def _mk_ratio(d: dict, fallback_obj: SemenSexRatio) -> SemenSexRatio:
            bull_raw = d.get("bull_share")
            heif_raw = d.get("heifer_share")

            if bull_raw is None and heif_raw is None:
                bull = float(fallback_obj.bull_share)
                heif = float(fallback_obj.heifer_share)
            elif bull_raw is None:
                heif = float(heif_raw)
                bull = 1.0 - heif
            elif heif_raw is None:
                bull = float(bull_raw)
                heif = 1.0 - bull
            else:
                bull = float(bull_raw)
                heif = float(heif_raw)

            bull = max(0.0, min(1.0, bull))
            heif = max(0.0, min(1.0, heif))
            s = max(1e-9, bull + heif)
            bull /= s
            heif /= s
            return SemenSexRatio(bull_share=bull, heifer_share=heif)

        semen_sex_ratios = {
            "trad": _mk_ratio(trad, _to_semen_ratio(SEMEN_SEX_RATIOS["trad"])),
            "sex": _mk_ratio(sex, _to_semen_ratio(SEMEN_SEX_RATIOS["sex"])),
        }
    else:
        semen_sex_ratios = compute_semen_sex_ratios_from_db(tables)

    warmstart_from_services = bool(ov.get("warmstart_from_services", True))
    state0 = build_initial_state(
        tables,
        as_of=start,
        gest_days=gest_days,
        dry_days=dry_days,
        insemination_params=params["INSEMINATION_PARAMS"],
        warmstart_from_services=warmstart_from_services,
        semen_sex_ratios=semen_sex_ratios,
    )
    state0_initial = _copy_state(state0)
    state_at_target, meta = simulate_to_target(
        state0,
        start=start,
        target=target_ts,
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        params=params,
    )
    return {
        "tables": tables,
        "target_ts": target_ts,
        "start": start,
        "params": params,
        "gest_days": gest_days,
        "dry_days": dry_days,
        "semen_shares": semen_shares,
        "semen_sex_ratios": semen_sex_ratios,
        "state0": state0_initial,
        "state_at_target": state_at_target,
        "meta": meta,
        "overrides": ov,
        "as_of_date": as_of_date,
    }


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _smape_percent(pred_val: float, fact_val: float) -> float | None:
    scale = abs(float(pred_val)) + abs(float(fact_val))
    if scale < 20.0:
        return None
    return 200.0 * abs(float(pred_val) - float(fact_val)) / scale


def _scale_birth_output(out: Dict[str, float], factor: float) -> None:
    factor = float(factor)
    if abs(factor - 1.0) <= 1e-9:
        return
    for key in BIRTH_OUTPUT_KEYS:
        if key in out:
            out[key] = round(_safe_float(out.get(key), 0.0) * factor, 1)


def _expected_calving_proxy_breakdown_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    gest_days: int,
    insemination_params: Dict[str, Any],
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    as_of_date: date | None = None,
) -> Dict[str, float]:
    out = {key: 0.0 for key in BIRTH_OUTPUT_KEYS}
    ins = tables.get("ins")
    if not isinstance(ins, pd.DataFrame) or ins.empty:
        return out

    d = ins.copy()
    d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
    d["lact_n"] = pd.to_numeric(d.get("lact"), errors="coerce")

    if as_of_date is not None:
        as_of_ts = pd.Timestamp(as_of_date).normalize()
        d = d[d["event_date_n"].notna() & (d["event_date_n"] <= as_of_ts)]
    else:
        d = d[d["event_date_n"].notna()]

    if d.empty:
        return out

    m_start = pd.Timestamp(target_date).normalize().to_period("M").to_timestamp().normalize()
    m_next = (m_start + pd.offsets.MonthBegin(1)).normalize()

    d["due_dt"] = d["event_date_n"] + pd.to_timedelta(int(gest_days), unit="D")
    due = d[(d["due_dt"] >= m_start) & (d["due_dt"] < m_next)].copy()
    if due.empty:
        return out

    cow_spc = _safe_float(
        insemination_params.get("cow_services_per_conception"),
        float(INSEMINATION_PARAMS.cow_services_per_conception),
    )
    heif_spc = _safe_float(
        insemination_params.get("heifer_services_per_conception"),
        float(INSEMINATION_PARAMS.heifer_services_per_conception),
    )

    cow_month_factors = _normalize_month_factor_map(
        insemination_params.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        insemination_params.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    due["base_p"] = np.where(
        due["lact_n"] > 0,
        1.0 / max(1e-9, cow_spc),
        1.0 / max(1e-9, heif_spc),
    )
    due["month_factor"] = np.where(
        due["lact_n"] > 0,
        due["event_date_n"].dt.month.map(lambda m: float(cow_month_factors.get(int(m), 1.0))),
        due["event_date_n"].dt.month.map(lambda m: float(heifer_month_factors.get(int(m), 1.0))),
    )
    due["month_factor"] = pd.to_numeric(due["month_factor"], errors="coerce").fillna(1.0)
    due["exp_weight"] = np.clip(due["base_p"] * due["month_factor"], 0.05, 0.95)

    exp_cow = float(due.loc[due["lact_n"] > 0, "exp_weight"].sum())
    exp_heif = float(due.loc[due["lact_n"] <= 0, "exp_weight"].sum())
    exp_unk = float(due.loc[due["lact_n"].isna(), "exp_weight"].sum())
    exp_total = exp_cow + exp_heif + exp_unk

    trad_bull = float(semen_sex_ratios["trad"].bull_share)
    sex_bull = float(semen_sex_ratios["sex"].bull_share)
    bull_share_cow = float(semen_shares.get("cow_trad", 0.0)) * trad_bull + float(semen_shares.get("cow_sex", 0.0)) * sex_bull
    bull_share_heif = float(semen_shares.get("heifer_trad", 0.0)) * trad_bull + float(semen_shares.get("heifer_sex", 0.0)) * sex_bull
    exp_bulls = (exp_cow + exp_unk) * bull_share_cow + exp_heif * bull_share_heif
    exp_heifers = exp_total - exp_bulls

    out["Ожидаемый отёл, всего"] = float(exp_total)
    out["Ожидаемый отёл, из них коров"] = float(exp_cow + exp_unk)
    out["Ожидаемый отёл, из них нетелей"] = float(exp_heif)
    out["Ожидаемые бычки"] = float(exp_bulls)
    out["Ожидаемые тёлочки"] = float(exp_heifers)
    return out


def _apply_current_month_observed_births_overlay(
    out: Dict[str, float],
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    start_ts: pd.Timestamp,
    as_of_date: date | None = None,
) -> None:
    if as_of_date is not None:
        return
    if start_ts is None or pd.isna(start_ts):
        return

    target_ts = pd.Timestamp(target_date).normalize()
    start_ts = pd.Timestamp(start_ts).normalize()
    if target_ts.to_period("M") != start_ts.to_period("M"):
        return

    observed_cutoff = (start_ts - pd.Timedelta(days=1)).normalize()
    month_start = target_ts.to_period("M").to_timestamp().normalize()
    if observed_cutoff < month_start:
        return

    from core.calving_facts import actual_birth_stats_from_tables

    observed = actual_birth_stats_from_tables(
        tables.get("calv", pd.DataFrame()),
        tables.get("ins", pd.DataFrame()),
        target_ts.date(),
        as_of_date=observed_cutoff.date(),
    )
    if _safe_float(observed.get("Ожидаемый отёл, всего"), 0.0) <= 0:
        return

    for key in BIRTH_OUTPUT_KEYS:
        out[key] = round(_safe_float(out.get(key), 0.0) + _safe_float(observed.get(key), 0.0), 1)


def _recent_birth_bias_factor_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    as_of_date: date | None,
    overrides: dict | None,
    gest_days: int,
    insemination_params: Dict[str, Any],
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
) -> float:
    if as_of_date is None:
        return 1.0

    ov = dict(overrides or {})
    if bool(ov.get("_DISABLE_DYNAMIC_BIRTH_ADJUST", False)):
        return 1.0
    if not bool(ov.get("auto_birth_bias_correction", True)):
        return 1.0

    from core.calving_facts import actual_birth_stats_from_tables, is_calving_month_complete_from_tables

    target_eom = pd.Timestamp(target_date).to_period("M").to_timestamp("M").date()
    asof_eom = pd.Timestamp(as_of_date).to_period("M").to_timestamp("M").date()
    horizon_m = max(0, _months_between_eom(asof_eom, target_eom))

    window = int(ov.get("birth_bias_window_months", 4) or 4)
    window = max(2, min(6, window))
    min_hist = int(ov.get("birth_bias_min_hist_months", 2) or 2)
    min_hist = max(2, min(window, min_hist))
    factor_min = _clamp(_safe_float(ov.get("birth_bias_factor_min"), 0.9), 0.7, 1.1)
    factor_max = _clamp(_safe_float(ov.get("birth_bias_factor_max"), 1.3), 1.0, 1.6)
    smape_threshold = _clamp(_safe_float(ov.get("birth_bias_smape_threshold"), 10.0), 5.0, 40.0)

    hist: list[tuple[float, float, float]] = []
    nested_ov = dict(ov)
    nested_ov["_DISABLE_DYNAMIC_BIRTH_ADJUST"] = True

    for i in range(1, window + 1):
        past_target = _month_end_shift(target_eom, -i)
        past_asof = _month_end_shift(past_target, -horizon_m)
        if past_asof > past_target:
            continue
        if not is_calving_month_complete_from_tables(tables.get("calv", pd.DataFrame()), past_target):
            continue

        past_pred_vals = compute_forecast_dynamic_from_tables(
            tables,
            past_target,
            overrides=nested_ov,
            as_of_date=past_asof,
        ) or {}
        pred_total = _safe_float(past_pred_vals.get("Ожидаемый отёл, всего"), 0.0)
        if pred_total <= 1e-9:
            continue

        fact_total = _safe_float(
            actual_birth_stats_from_tables(
                tables.get("calv", pd.DataFrame()),
                tables.get("ins", pd.DataFrame()),
                past_target,
                as_of_date=None,
            ).get("Ожидаемый отёл, всего"),
            0.0,
        )
        proxy_total = _safe_float(
            _expected_calving_proxy_breakdown_from_tables(
                tables,
                pd.Timestamp(past_target),
                gest_days=gest_days,
                insemination_params=insemination_params,
                semen_shares=semen_shares,
                semen_sex_ratios=semen_sex_ratios,
                as_of_date=past_asof,
            ).get("Ожидаемый отёл, всего"),
            0.0,
        )
        if proxy_total <= 1e-9:
            continue
        if (abs(pred_total) + abs(fact_total)) < 20.0:
            continue
        hist.append((pred_total, proxy_total, fact_total))

    if len(hist) < min_hist:
        return 1.0

    model_err = pd.Series([fact - pred for pred, _proxy, fact in hist], dtype=float)
    proxy_err = pd.Series([fact - proxy for _pred, proxy, fact in hist], dtype=float)

    same_model = max(float((model_err > 0).mean()), float((model_err < 0).mean()))
    same_proxy = max(float((proxy_err > 0).mean()), float((proxy_err < 0).mean()))
    model_smape_values = [_smape_percent(pred, fact) for pred, _proxy, fact in hist]
    proxy_smape_values = [_smape_percent(proxy, fact) for _pred, proxy, fact in hist]
    model_smape = float(pd.Series([x for x in model_smape_values if x is not None], dtype=float).mean() or 0.0)
    proxy_smape = float(pd.Series([x for x in proxy_smape_values if x is not None], dtype=float).mean() or 0.0)

    if not (
        same_model >= 0.75
        and same_proxy >= 0.75
        and float(model_err.mean()) > 0.0
        and float(proxy_err.mean()) > 0.0
        and model_smape >= smape_threshold
        and proxy_smape >= smape_threshold
    ):
        return 1.0

    sum_pred = float(sum(pred for pred, _proxy, _fact in hist))
    sum_fact = float(sum(fact for _pred, _proxy, fact in hist))
    if sum_pred <= 1e-9:
        return 1.0
    return _clamp(sum_fact / sum_pred, float(factor_min), float(factor_max))


def _apply_expected_calving_prob_fallback_from_tables(
    out: Dict[str, float],
    tables: Dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    *,
    gest_days: int,
    insemination_params: Dict[str, Any],
    semen_shares: Dict[str, float],
    semen_sex_ratios: Dict[str, SemenSexRatio],
    as_of_date: date | None = None,
) -> None:
    """
    Фолбэк для ожидаемого отёла в режиме расчёта из DataFrame-таблиц.
    Если основной расчёт дал 0 по "Ожидаемый отёл, всего", считаем прокси:
      expected = count(inseminations_due_in_month) * (1 / services_per_conception)
    """
    existing_total = _safe_float(out.get("Ожидаемый отёл, всего"), 0.0)
    if existing_total > 0:
        return

    ins = tables.get("ins")
    if not isinstance(ins, pd.DataFrame) or ins.empty:
        return

    d = ins.copy()
    d["event_date_n"] = pd.to_datetime(d.get("event_date"), errors="coerce").dt.normalize()
    d["lact_n"] = pd.to_numeric(d.get("lact"), errors="coerce")

    if as_of_date is not None:
        as_of_ts = pd.Timestamp(as_of_date).normalize()
        d = d[d["event_date_n"].notna() & (d["event_date_n"] <= as_of_ts)]
    else:
        d = d[d["event_date_n"].notna()]

    if d.empty:
        return

    m_start = pd.Timestamp(target_date).normalize().to_period("M").to_timestamp().normalize()
    m_next = (m_start + pd.offsets.MonthBegin(1)).normalize()

    d["due_dt"] = d["event_date_n"] + pd.to_timedelta(int(gest_days), unit="D")
    due = d[(d["due_dt"] >= m_start) & (d["due_dt"] < m_next)].copy()
    if due.empty:
        return

    n_cow = int((due["lact_n"] > 0).sum())
    n_heif = int((due["lact_n"] <= 0).sum())
    n_unk = int(due["lact_n"].isna().sum())

    if (n_cow + n_heif + n_unk) == 0:
        return

    cow_spc = _safe_float(
        insemination_params.get("cow_services_per_conception"),
        float(INSEMINATION_PARAMS.cow_services_per_conception),
    )
    heif_spc = _safe_float(
        insemination_params.get("heifer_services_per_conception"),
        float(INSEMINATION_PARAMS.heifer_services_per_conception),
    )

    cow_month_factors = _normalize_month_factor_map(
        insemination_params.get("cow_conception_month_factors", INSEMINATION_PARAMS.cow_conception_month_factors)
    )
    heifer_month_factors = _normalize_month_factor_map(
        insemination_params.get("heifer_conception_month_factors", INSEMINATION_PARAMS.heifer_conception_month_factors)
    )

    due["base_p"] = np.where(
        due["lact_n"] > 0,
        1.0 / max(1e-9, cow_spc),
        1.0 / max(1e-9, heif_spc),
    )
    due["month_factor"] = np.where(
        due["lact_n"] > 0,
        due["event_date_n"].dt.month.map(lambda m: float(cow_month_factors.get(int(m), 1.0))),
        due["event_date_n"].dt.month.map(lambda m: float(heifer_month_factors.get(int(m), 1.0))),
    )
    due["month_factor"] = pd.to_numeric(due["month_factor"], errors="coerce").fillna(1.0)
    due["exp_weight"] = np.where(
        due["lact_n"] > 0,
        np.clip(due["base_p"] * due["month_factor"], 0.05, 0.95),
        np.clip(due["base_p"] * due["month_factor"], 0.05, 0.95),
    )

    exp_cow = float(due.loc[due["lact_n"] > 0, "exp_weight"].sum())
    exp_heif = float(due.loc[due["lact_n"] <= 0, "exp_weight"].sum())
    exp_unk = float(due.loc[due["lact_n"].isna(), "exp_weight"].sum())
    exp_total = exp_cow + exp_heif + exp_unk

    out["Ожидаемый отёл, всего"] = round(float(exp_total), 1)
    out["Ожидаемый отёл, из них коров"] = round(float(exp_cow + exp_unk), 1)
    out["Ожидаемый отёл, из них нетелей"] = round(float(exp_heif), 1)

    trad_bull = float(semen_sex_ratios["trad"].bull_share)
    sex_bull = float(semen_sex_ratios["sex"].bull_share)

    bull_share_cow = float(semen_shares.get("cow_trad", 0.0)) * trad_bull + float(semen_shares.get("cow_sex", 0.0)) * sex_bull
    bull_share_heif = float(semen_shares.get("heifer_trad", 0.0)) * trad_bull + float(semen_shares.get("heifer_sex", 0.0)) * sex_bull

    exp_bulls = (exp_cow + exp_unk) * bull_share_cow + exp_heif * bull_share_heif
    exp_heifers = exp_total - exp_bulls

    out["Ожидаемые бычки"] = round(float(exp_bulls), 1)
    out["Ожидаемые тёлочки"] = round(float(exp_heifers), 1)


def compute_forecast_dynamic_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: date,
    overrides: dict | None = None,
    as_of_date: date | None = None,
) -> Dict[str, float]:
    sim = _run_dynamic_simulation_from_tables(
        tables,
        target_date,
        overrides=overrides,
        as_of_date=as_of_date,
    )
    tables = sim["tables"]
    target_date = sim["target_ts"]
    start = sim["start"]
    params = sim["params"]
    gest_days = int(sim["gest_days"])
    state_at_target = sim["state_at_target"]
    meta = sim["meta"]
    semen_shares = sim["semen_shares"]
    semen_sex_ratios = sim["semen_sex_ratios"]
    ov = sim["overrides"]

    cows_open = sum(state_at_target.open_dim[l].sum() for l in (1, 2, 3, 4))
    cows_preg_lact = sum(state_at_target.preg_lact[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))
    cows_preg_dry = sum(state_at_target.preg_dry[(l, s)].sum() for l in (1, 2, 3, 4) for s in ("trad", "sex"))

    doy = float(cows_open + cows_preg_lact)
    dry = float(cows_preg_dry)
    neteli = float(state_at_target.heifer_preg["trad"].sum() + state_at_target.heifer_preg["sex"].sum())

    h0_3 = float(state_at_target.heifer_age[:90].sum())
    h3_8 = float(state_at_target.heifer_age[90:270].sum())
    h9p = float(state_at_target.heifer_age[270:].sum())
    b0_2 = float(state_at_target.bull_age[:61].sum())

    calv_total_f = float(meta.get("calv_total", 0.0) or 0.0)
    calv_cows_f = float(meta.get("calv_cows", 0.0) or 0.0)
    calv_heifers_f = float(meta.get("calv_heifers", 0.0) or 0.0)
    exp_bulls_f = float(meta.get("exp_bulls", 0.0) or 0.0)
    exp_heifers_f = float(meta.get("exp_heifers", 0.0) or 0.0)

    out = {
        "Дойные коровы": round(doy),
        "Сухостойные коровы": round(dry),
        "Тёлки 0–3 мес": round(h0_3, 1),
        "Тёлки 0–2 мес": round(h0_3, 1),
        "Бычки 0–2 мес": round(b0_2, 1),
        "Тёлки 3–8 мес": round(h3_8, 1),
        "Тёлки ≥9 мес": round(h9p, 1),
        "Нетели": round(neteli, 1),
        "Ожидаемый отёл, всего": round(calv_total_f, 1),
        "Ожидаемый отёл, из них коров": round(calv_cows_f, 1),
        "Ожидаемый отёл, из них нетелей": round(calv_heifers_f, 1),
        "Ожидаемые бычки": round(exp_bulls_f, 1),
        "Ожидаемые тёлочки": round(exp_heifers_f, 1),
        "К реализации: коровы": round(float(meta.get("sell_cows", 0.0)), 1),
        "К реализации: тёлки": round(float(meta.get("sell_heifers", 0.0)), 1),
        "К реализации: нетели": round(float(meta.get("sell_neteli", 0.0)), 1),
        "Переполнение: Дойные коровы": round(float(meta.get("over_doy", 0.0)), 1),
        "Переполнение: Сухостойные коровы": round(float(meta.get("over_dry", 0.0)), 1),
        "Переполнение: Тёлки 0–3 мес": round(float(meta.get("over_h0", 0.0)), 1),
        "Переполнение: Тёлки 3–8 мес": round(float(meta.get("over_h38", 0.0)), 1),
        "Переполнение: Тёлки 9–24 мес": round(float(meta.get("over_h9", 0.0)), 1),
        "Переполнение: Нетели": round(float(meta.get("over_neteli", 0.0)), 1),
    }

    _apply_expected_calving_prob_fallback_from_tables(
        out,
        tables,
        target_date,
        gest_days=gest_days,
        insemination_params=params["INSEMINATION_PARAMS"],
        semen_shares=semen_shares,
        semen_sex_ratios=semen_sex_ratios,
        as_of_date=as_of_date,
    )
    _apply_current_month_observed_births_overlay(
        out,
        tables,
        target_date,
        start_ts=start,
        as_of_date=as_of_date,
    )
    _scale_birth_output(
        out,
        _recent_birth_bias_factor_from_tables(
            tables,
            target_date,
            as_of_date=as_of_date,
            overrides=ov,
            gest_days=gest_days,
            insemination_params=params["INSEMINATION_PARAMS"],
            semen_shares=semen_shares,
            semen_sex_ratios=semen_sex_ratios,
        ),
    )
    return out


def _metric_debug_breakdown_from_simulation(
    metric_name: str,
    *,
    state0: HerdState,
    state_at_target: HerdState,
    meta: Mapping[str, Any],
) -> Dict[str, float | None]:
    meta_start_values = {
        "Дойные коровы": float(meta.get("start_doy", np.nan)),
        "Сухостойные коровы": float(meta.get("start_dry", np.nan)),
        "Тёлки 0–3 мес": float(meta.get("start_h0", np.nan)),
        "Бычки 0–2 мес": float(meta.get("start_b0", np.nan)),
        "Тёлки 3–8 мес": float(meta.get("start_h38", np.nan)),
        "Тёлки ≥9 мес": float(meta.get("start_h9", np.nan)),
        "Нетели": float(meta.get("start_neteli", np.nan)),
    }
    start_snapshot = _state_group_snapshot(state0)
    end_snapshot = _state_group_snapshot(state_at_target)

    start_val = float(meta_start_values.get(metric_name, np.nan))
    if np.isnan(start_val):
        start_val = float(start_snapshot.get(metric_name, 0.0))
    end_val = float(end_snapshot.get(metric_name, 0.0))
    inflow = 0.0
    transition_next = 0.0
    other_outflow = 0.0

    if metric_name == "Дойные коровы":
        inflow = float(meta.get("flow_dry_to_doy", 0.0) or 0.0) + float(meta.get("flow_neteli_to_doy", 0.0) or 0.0)
        transition_next = float(meta.get("flow_doy_to_dry", 0.0) or 0.0)
        other_outflow = float(meta.get("disp_doy_total", 0.0) or 0.0) + float(meta.get("sell_cows_doy_total", 0.0) or 0.0)
    elif metric_name == "Сухостойные коровы":
        inflow = float(meta.get("flow_doy_to_dry", 0.0) or 0.0)
        transition_next = float(meta.get("flow_dry_to_doy", 0.0) or 0.0)
        other_outflow = float(meta.get("disp_dry_total", 0.0) or 0.0) + float(meta.get("sell_cows_dry_total", 0.0) or 0.0)
    elif metric_name == "Нетели":
        inflow = float(meta.get("flow_h9_to_neteli", 0.0) or 0.0)
        transition_next = float(meta.get("flow_neteli_to_doy", 0.0) or 0.0)
        other_outflow = float(meta.get("disp_neteli_total", 0.0) or 0.0) + float(meta.get("sell_neteli_total", 0.0) or 0.0)
    elif metric_name == "Тёлки ≥9 мес":
        inflow = float(meta.get("flow_h38_to_h9", 0.0) or 0.0)
        transition_next = float(meta.get("flow_h9_to_neteli", 0.0) or 0.0)
        other_outflow = float(meta.get("disp_h9_total", 0.0) or 0.0) + float(meta.get("sell_heifers_h9_total", 0.0) or 0.0)
    elif metric_name == "Тёлки 3–8 мес":
        inflow = float(meta.get("flow_h0_to_h38", 0.0) or 0.0)
        transition_next = float(meta.get("flow_h38_to_h9", 0.0) or 0.0)
        other_outflow = float(meta.get("sell_heifers_h38_total", 0.0) or 0.0)
    elif metric_name in {"Тёлки 0–3 мес", "Тёлки 0–2 мес"}:
        inflow = float(meta.get("flow_birth_female_total", 0.0) or 0.0)
        transition_next = float(meta.get("flow_h0_to_h38", 0.0) or 0.0)
        other_outflow = float(meta.get("sell_heifers_h0_total", 0.0) or 0.0)
    elif metric_name == "Бычки 0–2 мес":
        inflow = float(meta.get("flow_birth_male_total", 0.0) or 0.0)
        transition_next = float(meta.get("flow_b0_to_out", 0.0) or 0.0)
        other_outflow = 0.0

    return {
        "Старт группы": round(start_val, 1),
        "Приток": round(float(inflow), 1),
        "Переход в следующую группу": round(float(transition_next), 1),
        "Прочее выбытие": round(float(other_outflow), 1),
        "Выбытие всего": round(float(transition_next + other_outflow), 1),
        "Финиш группы (симуляция)": round(end_val, 1),
    }


def compute_forecast_debug_from_tables(
    tables: Dict[str, pd.DataFrame],
    target_date: date,
    metric_name: str,
    overrides: dict | None = None,
    as_of_date: date | None = None,
) -> Dict[str, float | None]:
    sim = _run_dynamic_simulation_from_tables(
        tables,
        target_date,
        overrides=overrides,
        as_of_date=as_of_date,
    )
    return _metric_debug_breakdown_from_simulation(
        metric_name,
        state0=sim["state0"],
        state_at_target=sim["state_at_target"],
        meta=sim["meta"],
    )
