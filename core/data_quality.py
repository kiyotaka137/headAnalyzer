from __future__ import annotations

from typing import Any, Mapping

import pandas as pd
from sqlalchemy import text

from core.insemination_success import infer_confirmed_conceptions
from forecast_dynamic_normalization import norm_event_type, norm_id


def _max_event_date_from_tables(tables: Mapping[str, pd.DataFrame]) -> pd.Timestamp | None:
    vals: list[pd.Timestamp] = []
    for key in ("calv", "ins", "dry", "disp"):
        df = tables.get(key)
        if not isinstance(df, pd.DataFrame) or df.empty or "event_date" not in df.columns:
            continue
        s = pd.to_datetime(df["event_date"], errors="coerce").dropna()
        if not s.empty:
            vals.append(pd.Timestamp(s.max()).normalize())
    return max(vals) if vals else None


def _merge_asof_by_reg(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    *,
    direction: str,
) -> pd.DataFrame:
    l = left.copy()
    r = right.copy()
    l[left_on] = pd.to_datetime(l[left_on], errors="coerce").dt.normalize()
    r[right_on] = pd.to_datetime(r[right_on], errors="coerce").dt.normalize()
    l = l.dropna(subset=["reg_s", left_on]).sort_values([left_on, "reg_s"], kind="mergesort")
    r = r.dropna(subset=["reg_s", right_on]).sort_values([right_on, "reg_s"], kind="mergesort")
    if l.empty or r.empty:
        return l
    return pd.merge_asof(
        l,
        r,
        by="reg_s",
        left_on=left_on,
        right_on=right_on,
        direction=direction,
        allow_exact_matches=True,
    )


def _calving_events_by_reg(calv_df: pd.DataFrame) -> pd.DataFrame:
    calv = calv_df.copy() if isinstance(calv_df, pd.DataFrame) else pd.DataFrame()
    if calv.empty:
        return pd.DataFrame(columns=["reg_s", "calv_dt"])
    calv["event_type_n"] = calv.get("event_type", pd.Series(dtype=object)).map(norm_event_type)
    calv["event_date_n"] = pd.to_datetime(calv.get("event_date"), errors="coerce").dt.normalize()
    calv["reg_s"] = calv.get("reg", pd.Series(dtype=object)).map(norm_id)
    calv["mother_reg_s"] = calv.get("mother_reg", pd.Series(dtype=object)).map(norm_id)
    otel = calv.loc[
        (calv["event_type_n"] == "ОТЕЛ") & (calv["reg_s"] != "") & calv["event_date_n"].notna(),
        ["reg_s", "event_date_n"],
    ].rename(columns={"event_date_n": "calv_dt"})
    born = calv.loc[
        (calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & calv["event_date_n"].notna(),
        ["mother_reg_s", "event_date_n"],
    ].rename(columns={"mother_reg_s": "reg_s", "event_date_n": "calv_dt"})
    parts = [x for x in (otel, born) if not x.empty]
    if not parts:
        return pd.DataFrame(columns=["reg_s", "calv_dt"])
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["reg_s", "calv_dt"], keep="first")
        .sort_values(["reg_s", "calv_dt"], kind="mergesort")
        .reset_index(drop=True)
    )


def subdivision_data_quality_report(
    subdivision_name: str,
    tables: Mapping[str, pd.DataFrame],
    *,
    raw_table_names: Mapping[str, str],
    con: Any,
) -> dict[str, pd.DataFrame]:
    sub = str(subdivision_name or "").strip()
    calv = tables.get("calv", pd.DataFrame()).copy()
    ins = tables.get("ins", pd.DataFrame()).copy()
    dry = tables.get("dry", pd.DataFrame()).copy()
    disp = tables.get("disp", pd.DataFrame()).copy()

    last_dt = _max_event_date_from_tables(tables)
    if last_dt is None:
        empty = pd.DataFrame()
        summary = pd.DataFrame(
            [
                {"Проверка": "Нет данных", "Значение": 0, "Комментарий": "В выбранном подразделении нет событий."},
            ]
        )
        return {
            "summary": summary,
            "incomplete_chain": empty,
            "first_seen_midcycle": empty,
            "cross_subdivision": empty,
        }

    ins["event_date_n"] = pd.to_datetime(ins.get("event_date"), errors="coerce").dt.normalize()
    ins["reg_s"] = ins.get("reg", pd.Series(dtype=object)).map(norm_id)
    disp["event_date_n"] = pd.to_datetime(disp.get("event_date"), errors="coerce").dt.normalize()
    disp["reg_s"] = disp.get("reg", pd.Series(dtype=object)).map(norm_id)
    calv_events = _calving_events_by_reg(calv)

    conc = infer_confirmed_conceptions(ins)
    if not conc.empty:
        conc["concept_date_n"] = pd.to_datetime(conc.get("concept_date"), errors="coerce").dt.normalize()
        conc["reg_s"] = conc.get("reg_s", pd.Series(dtype=object)).map(norm_id)
        old_conc = conc[
            conc["concept_date_n"].notna()
            & (conc["concept_date_n"] <= last_dt - pd.Timedelta(days=320))
            & (conc["reg_s"] != "")
        ].copy()
        if not old_conc.empty:
            old_conc = (
                old_conc.sort_values(["reg_s", "concept_date_n"], kind="mergesort")
                .drop_duplicates(subset=["reg_s"], keep="last")
                .rename(columns={"concept_date_n": "anchor_dt"})
            )
            calv_next = _merge_asof_by_reg(
                old_conc[["reg_s", "anchor_dt"]],
                calv_events.rename(columns={"calv_dt": "outcome_calv_dt"}),
                "anchor_dt",
                "outcome_calv_dt",
                direction="forward",
            )
            disp_next = _merge_asof_by_reg(
                old_conc[["reg_s", "anchor_dt"]],
                disp.loc[(disp["reg_s"] != "") & disp["event_date_n"].notna(), ["reg_s", "event_date_n"]].rename(columns={"event_date_n": "outcome_disp_dt"}),
                "anchor_dt",
                "outcome_disp_dt",
                direction="forward",
            )
            incomplete_chain = old_conc[["reg_s", "anchor_dt"]].copy()
            incomplete_chain["Дата отёла после осеменения"] = pd.to_datetime(calv_next.get("outcome_calv_dt"), errors="coerce")
            incomplete_chain["Дата выбытия после осеменения"] = pd.to_datetime(disp_next.get("outcome_disp_dt"), errors="coerce")
            incomplete_chain = incomplete_chain[
                incomplete_chain["Дата отёла после осеменения"].isna()
                & incomplete_chain["Дата выбытия после осеменения"].isna()
            ].copy()
            incomplete_chain["Подтверждённое осеменение"] = pd.to_datetime(incomplete_chain["anchor_dt"], errors="coerce").dt.date
            incomplete_chain = incomplete_chain.drop(columns=["anchor_dt"]).rename(columns={"reg_s": "reg"})
        else:
            incomplete_chain = pd.DataFrame(columns=["reg", "Подтверждённое осеменение", "Дата отёла после осеменения", "Дата выбытия после осеменения"])
    else:
        incomplete_chain = pd.DataFrame(columns=["reg", "Подтверждённое осеменение", "Дата отёла после осеменения", "Дата выбытия после осеменения"])

    first_seen_parts: list[pd.DataFrame] = []
    if isinstance(calv, pd.DataFrame) and not calv.empty:
        calv["event_type_n"] = calv.get("event_type", pd.Series(dtype=object)).map(norm_event_type)
        calv["event_date_n"] = pd.to_datetime(calv.get("event_date"), errors="coerce").dt.normalize()
        calv["birth_date_n"] = pd.to_datetime(calv.get("birth_date"), errors="coerce").dt.normalize()
        calv["reg_s"] = calv.get("reg", pd.Series(dtype=object)).map(norm_id)
        calv["mother_reg_s"] = calv.get("mother_reg", pd.Series(dtype=object)).map(norm_id)
        own_birth = calv.loc[
            (calv["event_type_n"] == "РОЖДЕН") & (calv["reg_s"] != ""),
            ["reg_s", "birth_date_n", "event_date_n"],
        ].copy()
        if not own_birth.empty:
            own_birth["event_dt"] = own_birth["birth_date_n"].where(own_birth["birth_date_n"].notna(), own_birth["event_date_n"])
            own_birth = own_birth.loc[own_birth["event_dt"].notna(), ["reg_s", "event_dt"]]
            own_birth["source"] = "рождение"
            first_seen_parts.append(own_birth)
        cow_calv = calv.loc[
            (calv["event_type_n"] == "ОТЕЛ") & (calv["reg_s"] != "") & calv["event_date_n"].notna(),
            ["reg_s", "event_date_n"],
        ].rename(columns={"event_date_n": "event_dt"})
        if not cow_calv.empty:
            cow_calv["source"] = "отёл"
            first_seen_parts.append(cow_calv)
        mother_birth = calv.loc[
            (calv["event_type_n"] == "РОЖДЕН") & (calv["mother_reg_s"] != "") & calv["event_date_n"].notna(),
            ["mother_reg_s", "event_date_n"],
        ].rename(columns={"mother_reg_s": "reg_s", "event_date_n": "event_dt"})
        if not mother_birth.empty:
            mother_birth["source"] = "отёл"
            first_seen_parts.append(mother_birth)
    if not ins.empty:
        ins_first = ins.loc[(ins["reg_s"] != "") & ins["event_date_n"].notna(), ["reg_s", "event_date_n"]].rename(columns={"event_date_n": "event_dt"})
        if not ins_first.empty:
            ins_first["source"] = "осеменение"
            first_seen_parts.append(ins_first)
    if not dry.empty:
        dry["event_date_n"] = pd.to_datetime(dry.get("event_date"), errors="coerce").dt.normalize()
        dry["reg_s"] = dry.get("reg", pd.Series(dtype=object)).map(norm_id)
        dry_first = dry.loc[(dry["reg_s"] != "") & dry["event_date_n"].notna(), ["reg_s", "event_date_n"]].rename(columns={"event_date_n": "event_dt"})
        if not dry_first.empty:
            dry_first["source"] = "запуск"
            first_seen_parts.append(dry_first)
    if not disp.empty:
        disp_first = disp.loc[(disp["reg_s"] != "") & disp["event_date_n"].notna(), ["reg_s", "event_date_n"]].rename(columns={"event_date_n": "event_dt"})
        if not disp_first.empty:
            disp_first["source"] = "выбытие"
            first_seen_parts.append(disp_first)

    if first_seen_parts:
        first_seen = (
            pd.concat(first_seen_parts, ignore_index=True)
            .dropna(subset=["event_dt"])
            .sort_values(["reg_s", "event_dt"], kind="mergesort")
            .drop_duplicates(subset=["reg_s"], keep="first")
        )
        first_seen_midcycle = first_seen.loc[first_seen["source"] != "рождение", ["reg_s", "event_dt", "source"]].copy()
        first_seen_midcycle["Первая дата"] = pd.to_datetime(first_seen_midcycle["event_dt"], errors="coerce").dt.date
        first_seen_midcycle = first_seen_midcycle.rename(columns={"reg_s": "reg", "source": "Первое событие"}).drop(columns=["event_dt"])
    else:
        first_seen_midcycle = pd.DataFrame(columns=["reg", "Первая дата", "Первое событие"])

    cross_sql = f"""
    WITH target_regs AS (
      SELECT DISTINCT CAST(reg AS text) AS reg FROM {raw_table_names['ins']} WHERE farm_name = :sub AND COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT DISTINCT CAST(reg AS text) AS reg FROM {raw_table_names['dry']} WHERE farm_name = :sub AND COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT DISTINCT CAST(reg AS text) AS reg FROM {raw_table_names['disp']} WHERE farm_name = :sub AND COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT DISTINCT CAST(reg AS text) AS reg FROM {raw_table_names['calv']} WHERE farm_name = :sub AND COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT DISTINCT CAST(mother_reg AS text) AS reg FROM {raw_table_names['calv']} WHERE farm_name = :sub AND COALESCE(TRIM(CAST(mother_reg AS text)), '') <> ''
    ),
    reg_farms AS (
      SELECT CAST(reg AS text) AS reg, farm_name FROM {raw_table_names['ins']} WHERE COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT CAST(reg AS text) AS reg, farm_name FROM {raw_table_names['dry']} WHERE COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT CAST(reg AS text) AS reg, farm_name FROM {raw_table_names['disp']} WHERE COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT CAST(reg AS text) AS reg, farm_name FROM {raw_table_names['calv']} WHERE COALESCE(TRIM(CAST(reg AS text)), '') <> ''
      UNION
      SELECT CAST(mother_reg AS text) AS reg, farm_name FROM {raw_table_names['calv']} WHERE COALESCE(TRIM(CAST(mother_reg AS text)), '') <> ''
    )
    SELECT reg, COUNT(DISTINCT farm_name) AS n_subdivisions, STRING_AGG(DISTINCT farm_name, ', ') AS subdivisions
    FROM reg_farms
    WHERE reg IN (SELECT reg FROM target_regs)
    GROUP BY reg
    HAVING COUNT(DISTINCT farm_name) > 1
    ORDER BY n_subdivisions DESC, reg
    LIMIT 100
    """
    try:
        cross_subdivision = pd.read_sql(text(cross_sql), con=con, params={"sub": sub})
    except Exception:
        cross_subdivision = pd.DataFrame(columns=["reg", "n_subdivisions", "subdivisions"])
    if not cross_subdivision.empty:
        cross_subdivision = cross_subdivision.rename(columns={"n_subdivisions": "Подразделений", "subdivisions": "Где встречается"})

    summary = pd.DataFrame(
        [
            {
                "Проверка": "Подтверждённые осеменения без последующего отёла/выбытия",
                "Значение": int(len(incomplete_chain)),
                "Комментарий": "Старые подтверждённые осеменения, по которым в подразделении не видно ни отёла, ни выбытия.",
            },
            {
                "Проверка": "Животные, впервые появляющиеся не с рождения",
                "Значение": int(len(first_seen_midcycle)),
                "Комментарий": "Первое событие в истории подразделения — не рождение, а уже осеменение/запуск/выбытие/отёл.",
            },
            {
                "Проверка": "Животные со событиями в нескольких подразделениях",
                "Значение": int(len(cross_subdivision)),
                "Комментарий": "Один и тот же reg встречается более чем в одном подразделении в таблицах хозяйства.",
            },
        ]
    )

    return {
        "summary": summary,
        "incomplete_chain": incomplete_chain.head(50),
        "first_seen_midcycle": first_seen_midcycle.head(50),
        "cross_subdivision": cross_subdivision.head(50),
    }
