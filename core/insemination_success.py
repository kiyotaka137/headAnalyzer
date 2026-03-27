from __future__ import annotations

import pandas as pd

from core.date_parse import parse_mixed_datetime
from forecast_dynamic_normalization import norm_id, norm_result

SERVICE_RESULT_MARKERS = {"", "O", "О", "-", "P", "A", "А"}


def _prepare_inseminations(ins: pd.DataFrame) -> pd.DataFrame:
    df = ins.copy()
    df["event_date_n"] = parse_mixed_datetime(df.get("event_date")).dt.normalize()
    df["reg_s"] = df.get("reg", pd.Series(index=df.index, dtype=object)).apply(norm_id)
    df["result_norm"] = df.get("result", pd.Series(index=df.index, dtype=object)).apply(norm_result)
    df["lact_n"] = pd.to_numeric(df.get("lact"), errors="coerce")
    df["dim_age_n"] = pd.to_numeric(df.get("dim_age"), errors="coerce")
    if "bull" in df.columns:
        df["bull_s"] = df["bull"].apply(norm_id)
    else:
        df["bull_s"] = ""
    return df


def infer_confirmed_conceptions(
    ins: pd.DataFrame,
    *,
    service_results: Iterable[str] = SERVICE_RESULT_MARKERS,
    max_confirm_lag_days: int = 120,
) -> pd.DataFrame:
    """
    В текущем формате данных `event_date` — это дата самого осеменения, а `P`
    означает, что именно это осеменение было результативным. Поэтому дата
    зачатия берётся напрямую из `P`-строки, без поиска "предыдущего сервиса".

    Параметры `service_results` и `max_confirm_lag_days` сохранены только для
    совместимости вызовов; внутри этой версии функции они не используются.
    """
    _ = service_results
    _ = max_confirm_lag_days
    if ins is None or ins.empty:
        return pd.DataFrame(
            columns=[
                "reg_s",
                "concept_date",
                "confirm_date",
                "lact_n",
                "dim_age_n",
                "bull_s",
                "lag_days",
                "concept_source",
            ]
        )

    df = _prepare_inseminations(ins)
    df = df[(df["reg_s"] != "") & df["event_date_n"].notna()].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "reg_s",
                "concept_date",
                "confirm_date",
                "lact_n",
                "dim_age_n",
                "bull_s",
                "lag_days",
                "concept_source",
            ]
        )

    p = df[df["result_norm"] == "P"][
        ["reg_s", "event_date_n", "lact_n", "dim_age_n", "bull_s"]
    ].copy()
    if p.empty:
        return pd.DataFrame(
            columns=[
                "reg_s",
                "concept_date",
                "confirm_date",
                "lact_n",
                "dim_age_n",
                "bull_s",
                "lag_days",
                "concept_source",
            ]
        )

    p = p.rename(
        columns={
            "event_date_n": "concept_date",
        }
    )
    p["confirm_date"] = p["concept_date"]
    p["lag_days"] = 0
    p["concept_source"] = "result_p"

    out = p[
        ["reg_s", "concept_date", "confirm_date", "lact_n", "dim_age_n", "bull_s", "lag_days", "concept_source"]
    ].copy()
    out = out.dropna(subset=["concept_date"])
    out = out.sort_values(["reg_s", "concept_date", "confirm_date"], kind="mergesort")
    out = out.drop_duplicates(subset=["reg_s", "concept_date"], keep="last").reset_index(drop=True)
    return out
