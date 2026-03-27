from __future__ import annotations

from datetime import date
from typing import Any, Callable, Collection, Mapping

import pandas as pd
from sqlalchemy import text

from core.calving_facts import (
    BIRTH_STATS_KEYS,
    actual_birth_stats_from_tables,
    is_calving_month_complete_from_tables,
)
from core.helpers import month_end, vals_get
from core.insemination_success import infer_confirmed_conceptions
from forecast_dynamic import (
    _estimate_active_cow_regs_at_asof,
    _prepare_calving_rows,
    animal_asof_snapshot_group_totals,
    build_animal_asof_snapshot,
    latest_data_date,
)
from forecast_dynamic_normalization import norm_event_type, norm_id, norm_sex


NETELI_FUTURE_LOOKAHEAD_DAYS = 290
NETELI_EXPLICIT_HISTORY_MIN = 10


def backtest_percent_error(pred_val: float, fact_val: float, *, is_pct: bool) -> float | None:
    err_abs = abs(float(pred_val) - float(fact_val))
    if is_pct:
        return (err_abs / abs(float(fact_val)) * 100.0) if abs(float(fact_val)) > 1e-9 else None
    scale = abs(float(pred_val)) + abs(float(fact_val))
    if scale < 20.0:
        return None
    return 200.0 * err_abs / scale


def month_end_shift(d_end: date, months_delta: int) -> date:
    ts = pd.Timestamp(d_end) + pd.DateOffset(months=months_delta)
    return month_end(int(ts.year), int(ts.month))


def collect_recent_target_months(
    last_month_end: date,
    desired_n: int,
    *,
    complete_checker: Callable[[date], bool] | None = None,
    search_limit_months: int | None = None,
) -> list[date]:
    n = max(0, int(desired_n))
    if n <= 0:
        return []
    if complete_checker is None:
        return [month_end_shift(last_month_end, -i) for i in range(n - 1, -1, -1)]

    limit = max(n, int(search_limit_months or 0), 24)
    found: list[date] = []
    for i in range(limit):
        cand = month_end_shift(last_month_end, -i)
        try:
            if complete_checker(cand):
                found.append(cand)
                if len(found) >= n:
                    break
        except Exception:
            continue
    return list(reversed(found))


def actual_birth_stats_month_from_db(
    con: Any,
    month_end_date: date,
    *,
    stat_keys: Collection[str] = BIRTH_STATS_KEYS,
) -> dict[str, float]:
    m_start = date(month_end_date.year, month_end_date.month, 1)
    if month_end_date.month == 12:
        m_next = date(month_end_date.year + 1, 1, 1)
    else:
        m_next = date(month_end_date.year, month_end_date.month + 1, 1)

    calv_sql = """
    SELECT reg, mother_reg, birth_date, sex, event_type, event_date, lact
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND event_date::date >= :m_start
      AND event_date::date < :m_next
    """
    ins_sql = """
    SELECT reg, lact, event_date
    FROM inseminations_raw
    WHERE event_date IS NOT NULL
      AND event_date::date <= :m_end
    """
    calv_df = pd.read_sql(text(calv_sql), con=con, params={"m_start": m_start, "m_next": m_next})
    if calv_df.empty:
        return {str(k): 0.0 for k in stat_keys}
    ins_df = pd.read_sql(text(ins_sql), con=con, params={"m_end": month_end_date})
    return actual_birth_stats_from_tables(calv_df, ins_df, month_end_date, as_of_date=None)


def actual_nonbirth_snapshot_from_db(con: Any, as_of_date: date) -> dict[str, float]:
    sql_calv = """
    SELECT reg, mother_reg, birth_date, sex, event_type, event_date, lact
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND event_date::date <= :m_end
    """
    sql_ins = """
    SELECT reg, lact, dim_age, event_date, bull, result
    FROM inseminations_raw
    WHERE event_date IS NOT NULL
      AND event_date::date <= :m_end
    """
    sql_dry = """
    SELECT reg, dim, event_date
    FROM dryoff_raw
    WHERE event_date IS NOT NULL
      AND event_date::date <= :m_end
    """
    sql_disp = """
    SELECT reg, event_date, disposal_reason
    FROM disposals_raw
    WHERE event_date IS NOT NULL
      AND event_date::date <= :m_end
    """
    calv_df = pd.read_sql(text(sql_calv), con=con, params={"m_end": as_of_date})
    ins_df = pd.read_sql(text(sql_ins), con=con, params={"m_end": as_of_date})
    dry_df = pd.read_sql(text(sql_dry), con=con, params={"m_end": as_of_date})
    disp_df = pd.read_sql(text(sql_disp), con=con, params={"m_end": as_of_date})
    return actual_nonbirth_snapshot_from_tables(calv_df, ins_df, dry_df, disp_df, as_of_date)


def actual_metric_month_from_db(
    con: Any,
    month_end_date: date,
    metric_name: str,
    *,
    birth_targets: Collection[str],
    percent_targets: Collection[str],
    stat_keys: Collection[str] = BIRTH_STATS_KEYS,
) -> float:
    if metric_name in set(birth_targets) or metric_name in set(percent_targets):
        return float(actual_birth_stats_month_from_db(con, month_end_date, stat_keys=stat_keys).get(metric_name, 0.0))
    return float(actual_nonbirth_snapshot_from_db(con, month_end_date).get(metric_name, 0.0))


def is_fact_month_complete_from_db(con: Any, month_end_date: date) -> bool:
    m_start = date(month_end_date.year, month_end_date.month, 1)
    if month_end_date.month == 12:
        m_next = date(month_end_date.year + 1, 1, 1)
    else:
        m_next = date(month_end_date.year, month_end_date.month + 1, 1)

    sql = """
    SELECT reg, mother_reg, birth_date, sex, event_type, event_date, lact
    FROM calvings_births_raw
    WHERE event_date IS NOT NULL
      AND event_date::date >= :m_start
      AND event_date::date < :m_next
    """
    df = pd.read_sql(text(sql), con=con, params={"m_start": m_start, "m_next": m_next})
    return is_calving_month_complete_from_tables(df, month_end_date)


def actual_birth_stats_month_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame | None,
    month_end_date: date,
    as_of_date: date | None = None,
) -> dict[str, float]:
    return actual_birth_stats_from_tables(calv_df, ins_df, month_end_date, as_of_date=as_of_date)


def _neteli_has_explicit_history_from_tables(
    ins_df: pd.DataFrame,
    calv_df: pd.DataFrame | None,
    as_of_date: date,
) -> bool:
    if not isinstance(ins_df, pd.DataFrame) or ins_df.empty:
        return False
    conc = infer_confirmed_conceptions(ins_df)
    if conc.empty:
        return False
    first_calv_by_reg: dict[str, pd.Timestamp] = {}
    calv = _prepare_calving_rows(calv_df if isinstance(calv_df, pd.DataFrame) else pd.DataFrame())
    if not calv.empty:
        parts: list[pd.DataFrame] = []
        otel = calv.loc[
            (calv["event_type_n"] == "ОТЕЛ") & (calv["reg_s"] != "") & calv["calving_dt_n"].notna(),
            ["reg_s", "calving_dt_n"],
        ].rename(columns={"reg_s": "cow_reg_s", "calving_dt_n": "calv_dt"})
        born = calv.loc[
            (calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & calv["calving_dt_n"].notna(),
            ["mother_reg_s", "calving_dt_n"],
        ].rename(columns={"mother_reg_s": "cow_reg_s", "calving_dt_n": "calv_dt"})
        if not otel.empty:
            parts.append(otel)
        if not born.empty:
            parts.append(born)
        if parts:
            all_calv = (
                pd.concat(parts, ignore_index=True)
                .sort_values(["cow_reg_s", "calv_dt"], kind="mergesort")
                .drop_duplicates(subset=["cow_reg_s", "calv_dt"], keep="first")
            )
            first_calv_by_reg = (
                all_calv.drop_duplicates(subset=["cow_reg_s"], keep="first")
                .set_index("cow_reg_s")["calv_dt"]
                .to_dict()
            )
    as_of_ts = pd.Timestamp(as_of_date).normalize()
    lo = as_of_ts - pd.Timedelta(days=NETELI_FUTURE_LOOKAHEAD_DAYS)
    def _is_pre_first_calving(reg: object) -> bool:
        reg_s = str(reg or "")
        if not reg_s:
            return False
        first_dt = first_calv_by_reg.get(reg_s)
        return pd.isna(first_dt) or (pd.Timestamp(first_dt) > as_of_ts)
    conc["concept_date"] = pd.to_datetime(conc.get("concept_date"), errors="coerce").dt.normalize()
    conc = conc[
        conc["concept_date"].notna()
        & (conc["concept_date"] >= lo)
        & (conc["concept_date"] <= as_of_ts)
        & conc["reg_s"].map(_is_pre_first_calving)
    ].copy()
    if conc.empty:
        return False
    uniq_regs = conc.get("reg_s", pd.Series(dtype=object)).astype(str)
    uniq_regs = uniq_regs[uniq_regs != ""].nunique()
    return bool(int(uniq_regs) >= NETELI_EXPLICIT_HISTORY_MIN)


def _latest_calving_fact_date(calv_df: pd.DataFrame) -> date | None:
    calv = _prepare_calving_rows(calv_df if isinstance(calv_df, pd.DataFrame) else pd.DataFrame())
    if calv.empty:
        return None
    calv = calv[calv["event_type_n"].isin(["ОТЕЛ", "РОЖДЕН"])].copy()
    max_dt = pd.to_datetime(calv["calving_dt_n"], errors="coerce").max()
    if pd.isna(max_dt):
        return None
    return pd.Timestamp(max_dt).date()


def actual_nonbirth_snapshot_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame,
    dry_df: pd.DataFrame,
    disp_df: pd.DataFrame,
    as_of_date: date,
) -> dict[str, float]:
    out = {
        "Дойные коровы": 0.0,
        "Сухостойные коровы": 0.0,
        "Тёлки 0–3 мес": 0.0,
        "Бычки 0–2 мес": 0.0,
        "Тёлки 3–8 мес": 0.0,
        "Тёлки ≥9 мес": 0.0,
        "Нетели": 0.0,
    }
    as_of_ts = pd.Timestamp(as_of_date).normalize()
    cow_lookback_days = 540
    youngstock_max_days = 730
    conception_lookback_days = 320
    future_first_calving_lookahead_days = NETELI_FUTURE_LOOKAHEAD_DAYS
    recent_lo = as_of_ts - pd.Timedelta(days=cow_lookback_days)
    young_lo = as_of_ts - pd.Timedelta(days=youngstock_max_days)
    conception_lo = as_of_ts - pd.Timedelta(days=conception_lookback_days)

    disp = disp_df.copy() if isinstance(disp_df, pd.DataFrame) else pd.DataFrame()
    if not disp.empty:
        disp["event_date_n"] = pd.to_datetime(disp.get("event_date"), errors="coerce").dt.normalize()
        disp["reg_s"] = disp.get("reg", pd.Series(dtype=object)).map(norm_id)
        disp = disp[(disp["event_date_n"].notna()) & (disp["event_date_n"] <= as_of_ts) & (disp["reg_s"] != "")]
    disposed: set[str] = set(disp["reg_s"].astype(str).tolist()) if not disp.empty else set()

    ins = ins_df.copy() if isinstance(ins_df, pd.DataFrame) else pd.DataFrame()
    if not ins.empty:
        ins["event_date_n"] = pd.to_datetime(ins.get("event_date"), errors="coerce").dt.normalize()
        ins["reg_s"] = ins.get("reg", pd.Series(dtype=object)).map(norm_id)
        ins["lact_n"] = pd.to_numeric(ins.get("lact"), errors="coerce")
        ins = ins[(ins["event_date_n"].notna()) & (ins["event_date_n"] <= as_of_ts) & (ins["reg_s"] != "")]
    ins_recent = ins.loc[ins["event_date_n"] >= recent_lo].copy() if not ins.empty else pd.DataFrame()

    dry = dry_df.copy() if isinstance(dry_df, pd.DataFrame) else pd.DataFrame()
    if not dry.empty:
        dry["event_date_n"] = pd.to_datetime(dry.get("event_date"), errors="coerce").dt.normalize()
        dry["reg_s"] = dry.get("reg", pd.Series(dtype=object)).map(norm_id)
        dry = dry[(dry["event_date_n"].notna()) & (dry["event_date_n"] <= as_of_ts) & (dry["reg_s"] != "")]
    if not dry.empty:
        last_dry = (
            dry.sort_values(["reg_s", "event_date_n"], kind="mergesort")
            .drop_duplicates(subset=["reg_s"], keep="last")
            .set_index("reg_s")["event_date_n"]
            .to_dict()
        )
    else:
        last_dry = {}

    calv_all = _prepare_calving_rows(calv_df if isinstance(calv_df, pd.DataFrame) else pd.DataFrame())
    calv = calv_all[calv_all["calving_dt_n"].notna() & (calv_all["calving_dt_n"] <= as_of_ts)].copy()
    born = calv.loc[calv["event_type_n"] == "РОЖДЕН"].copy()

    if not born.empty:
        born["birth_dt_n"] = born["birth_date_n"].where(born["birth_date_n"].notna(), born["event_date_n"])
    else:
        born["birth_dt_n"] = pd.NaT

    if not born.empty:
        last_calv_by_mother = (
            born.loc[born["mother_reg_s"] != "", ["mother_reg_s", "calving_dt_n"]]
            .sort_values(["mother_reg_s", "calving_dt_n"], kind="mergesort")
            .drop_duplicates(subset=["mother_reg_s"], keep="last")
            .set_index("mother_reg_s")["calving_dt_n"]
            .to_dict()
        )
    else:
        last_calv_by_mother = {}

    calv_events_parts: list[pd.DataFrame] = []
    if not calv_all.empty:
        otel_events = calv_all.loc[
            (calv_all["event_type_n"] == "ОТЕЛ") & (calv_all["reg_s"] != "") & calv_all["calving_dt_n"].notna(),
            ["reg_s", "calving_dt_n"],
        ].rename(columns={"reg_s": "cow_reg_s", "calving_dt_n": "calv_dt"})
        born_events = calv_all.loc[
            (calv_all["event_type_n"] == "РОЖДЕН") & (calv_all["mother_reg_s"] != "") & calv_all["calving_dt_n"].notna(),
            ["mother_reg_s", "calving_dt_n"],
        ].rename(columns={"mother_reg_s": "cow_reg_s", "calving_dt_n": "calv_dt"})
        if not otel_events.empty:
            calv_events_parts.append(otel_events)
        if not born_events.empty:
            calv_events_parts.append(born_events)
    if calv_events_parts:
        all_calv_events = (
            pd.concat(calv_events_parts, ignore_index=True)
            .sort_values(["cow_reg_s", "calv_dt"], kind="mergesort")
            .drop_duplicates(subset=["cow_reg_s", "calv_dt"], keep="first")
        )
        first_calv_by_reg = (
            all_calv_events.drop_duplicates(subset=["cow_reg_s"], keep="first")
            .set_index("cow_reg_s")["calv_dt"]
            .to_dict()
        )
    else:
        first_calv_by_reg = {}

    def _is_pre_first_calving(reg: object) -> bool:
        reg_s = str(reg or "")
        if not reg_s:
            return False
        first_dt = first_calv_by_reg.get(reg_s)
        return pd.isna(first_dt) or (pd.Timestamp(first_dt) > as_of_ts)

    conc = infer_confirmed_conceptions(ins if isinstance(ins, pd.DataFrame) else pd.DataFrame())
    if not conc.empty:
        conc["concept_date"] = pd.to_datetime(conc.get("concept_date"), errors="coerce").dt.normalize()
        conc = conc[
            conc["concept_date"].notna()
            & (conc["concept_date"] >= conception_lo)
            & (conc["concept_date"] <= as_of_ts)
            & conc["reg_s"].map(_is_pre_first_calving)
        ].copy()
        conc = (
            conc.sort_values(["reg_s", "concept_date"], kind="mergesort")
            .drop_duplicates(subset=["reg_s"], keep="last")
        )
        neteli_alive = {
            str(reg)
            for reg in conc["reg_s"].astype(str).tolist()
            if reg and reg not in disposed and reg not in last_calv_by_mother
        }
    else:
        neteli_alive = set()

    neteli_from_future_first_calving = {
        str(reg)
        for reg, first_dt in first_calv_by_reg.items()
        if str(reg)
        and pd.notna(first_dt)
        and (pd.Timestamp(first_dt) > as_of_ts)
        and (pd.Timestamp(first_dt) <= as_of_ts + pd.Timedelta(days=future_first_calving_lookahead_days))
        and str(reg) not in disposed
    }
    neteli_alive |= neteli_from_future_first_calving

    if not ins_recent.empty:
        open_heifer_from_ins = {
            reg
            for reg in ins_recent.loc[
                ins_recent["reg_s"].map(_is_pre_first_calving),
                "reg_s",
            ].astype(str).tolist()
            if reg
        }
    else:
        open_heifer_from_ins = set()

    cows_alive = _estimate_active_cow_regs_at_asof(
        calv=calv if isinstance(calv, pd.DataFrame) else pd.DataFrame(),
        ins=ins if isinstance(ins, pd.DataFrame) else pd.DataFrame(),
        dry=dry if isinstance(dry, pd.DataFrame) else pd.DataFrame(),
        as_of_ts=as_of_ts,
        lookback_days=cow_lookback_days,
        first_calv_by_reg=first_calv_by_reg,
    )
    cows_alive = {
        reg for reg in cows_alive
        if reg
        and reg not in disposed
        and (
            reg not in first_calv_by_reg
            or pd.isna(first_calv_by_reg.get(reg))
            or pd.Timestamp(first_calv_by_reg.get(reg)) <= as_of_ts
        )
    }

    dry_count = 0
    for reg in cows_alive:
        dry_dt = last_dry.get(reg)
        if dry_dt is None or pd.isna(dry_dt):
            continue
        calv_dt = last_calv_by_mother.get(reg)
        if calv_dt is None or pd.isna(calv_dt):
            dry_count += 1
        elif pd.Timestamp(dry_dt) > pd.Timestamp(calv_dt):
            dry_count += 1
    doy_count = max(0, len(cows_alive) - dry_count)

    neteli_alive = {
        reg for reg in neteli_alive
        if reg and reg not in cows_alive
    }

    h9_open_alive = {
        reg for reg in open_heifer_from_ins
        if reg and reg not in disposed and reg not in cows_alive and reg not in last_calv_by_mother and reg not in neteli_alive
    }

    calf_excluded = set(cows_alive) | set(neteli_alive) | set(h9_open_alive)
    if not born.empty:
        young = born.loc[(born["birth_dt_n"].notna()) & (born["birth_dt_n"] >= young_lo)].copy()
        calves_f = young.loc[(young["sex_norm"] == "F") & (young["reg_s"] != ""), ["reg_s", "birth_dt_n"]].copy()
        calves_m = young.loc[(young["sex_norm"] == "M") & (young["reg_s"] != ""), ["reg_s", "birth_dt_n"]].copy()
    else:
        calves_f = pd.DataFrame(columns=["reg_s", "birth_dt_n"])
        calves_m = pd.DataFrame(columns=["reg_s", "birth_dt_n"])

    if not calves_f.empty:
        calves_f = (
            calves_f.sort_values(["reg_s", "birth_dt_n"], kind="mergesort")
            .drop_duplicates(subset=["reg_s"], keep="last")
            .copy()
        )
    if not calves_m.empty:
        calves_m = (
            calves_m.sort_values(["reg_s", "birth_dt_n"], kind="mergesort")
            .drop_duplicates(subset=["reg_s"], keep="last")
            .copy()
        )

    def _count_by_age(df: pd.DataFrame) -> pd.Series:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series(dtype=float)
        work = df.copy()
        work = work[work["birth_dt_n"].notna()].copy()
        if work.empty:
            return pd.Series(dtype=float)
        work = work[~work["reg_s"].astype(str).isin(disposed)]
        work = work[~work["reg_s"].astype(str).isin(calf_excluded)]
        if work.empty:
            return pd.Series(dtype=float)
        return (as_of_ts - pd.to_datetime(work["birth_dt_n"], errors="coerce")).dt.days

    age_f = _count_by_age(calves_f)
    age_m = _count_by_age(calves_m)

    out["Дойные коровы"] = float(doy_count)
    out["Сухостойные коровы"] = float(dry_count)
    out["Нетели"] = float(len(neteli_alive))
    out["Тёлки 0–3 мес"] = float(((age_f >= 0) & (age_f < 90)).sum()) if not age_f.empty else 0.0
    out["Бычки 0–2 мес"] = float(((age_m >= 0) & (age_m < 61)).sum()) if not age_m.empty else 0.0
    out["Тёлки 3–8 мес"] = float(((age_f >= 90) & (age_f < 270)).sum()) if not age_f.empty else 0.0
    out["Тёлки ≥9 мес"] = float(((age_f >= 270) & (age_f < youngstock_max_days)).sum()) if not age_f.empty else 0.0

    animal_snapshot = build_animal_asof_snapshot(
        {
            "calv": calv_df if isinstance(calv_df, pd.DataFrame) else pd.DataFrame(),
            "ins": ins_df if isinstance(ins_df, pd.DataFrame) else pd.DataFrame(),
            "dry": dry_df if isinstance(dry_df, pd.DataFrame) else pd.DataFrame(),
            "disp": disp_df if isinstance(disp_df, pd.DataFrame) else pd.DataFrame(),
            "bulls": pd.DataFrame(columns=["bull_code", "bull_type"]),
        },
        as_of_date,
        warmstart_from_services=False,
    )
    out.update(animal_asof_snapshot_group_totals(animal_snapshot))
    return out


def actual_metric_month_from_tables(
    calv_df: pd.DataFrame,
    ins_df: pd.DataFrame | None,
    dry_df: pd.DataFrame | None,
    disp_df: pd.DataFrame | None,
    month_end_date: date,
    metric_name: str,
    *,
    birth_targets: Collection[str],
    percent_targets: Collection[str],
) -> float:
    if metric_name in set(birth_targets) or metric_name in set(percent_targets):
        return float(actual_birth_stats_month_from_tables(calv_df, ins_df, month_end_date).get(metric_name, 0.0))
    snapshot = actual_nonbirth_snapshot_from_tables(
        calv_df,
        ins_df if isinstance(ins_df, pd.DataFrame) else pd.DataFrame(),
        dry_df if isinstance(dry_df, pd.DataFrame) else pd.DataFrame(),
        disp_df if isinstance(disp_df, pd.DataFrame) else pd.DataFrame(),
        month_end_date,
    )
    return float(snapshot.get(metric_name, 0.0))


def pred_metric_value(
    pred_vals: Mapping[str, Any],
    metric_name: str,
    nmap: Mapping[str, Any],
    *,
    percent_targets: Collection[str],
) -> float:
    if metric_name in set(percent_targets):
        pred_bull = float(vals_get(dict(pred_vals), "Ожидаемые бычки", dict(nmap)) or 0.0)
        pred_heif = float(vals_get(dict(pred_vals), "Ожидаемые тёлочки", dict(nmap)) or 0.0)
        den = pred_bull + pred_heif
        if den <= 0:
            return 0.0
        if metric_name == "Доля бычков среди рождений, %":
            return pred_bull / den * 100.0
        return pred_heif / den * 100.0
    return float(vals_get(dict(pred_vals), metric_name, dict(nmap)) or 0.0)


def target_fact_month_complete_from_tables(
    tables: dict[str, pd.DataFrame],
    metric_name: str,
    month_end_date: date,
    *,
    birth_targets: Collection[str],
    percent_targets: Collection[str],
) -> bool:
    if metric_name in set(birth_targets) or metric_name in set(percent_targets):
        return is_calving_month_complete_from_tables(tables.get("calv", pd.DataFrame()), month_end_date)
    if metric_name == "Нетели":
        ins_df = tables.get("ins", pd.DataFrame())
        if _neteli_has_explicit_history_from_tables(ins_df, tables.get("calv", pd.DataFrame()), month_end_date):
            try:
                return bool(latest_data_date(tables) >= month_end_date)
            except Exception:
                return False
        max_calv_dt = _latest_calving_fact_date(tables.get("calv", pd.DataFrame()))
        if max_calv_dt is None:
            return False
        need_until = (pd.Timestamp(month_end_date).normalize() + pd.Timedelta(days=NETELI_FUTURE_LOOKAHEAD_DAYS)).date()
        return bool(max_calv_dt >= need_until)
    try:
        return bool(latest_data_date(tables) >= month_end_date)
    except Exception:
        return False
