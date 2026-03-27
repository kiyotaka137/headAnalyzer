from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text


DEFAULT_DSN_CANDIDATES = (
    os.getenv("POSTGRES_DSN", "").strip(),
    "postgresql+psycopg2://herd_user:herd_password@localhost:15432/herd_forecast",
)

PERCENT_TARGETS = {
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
}

BIRTH_TARGETS = {
    "Ожидаемый отёл, всего",
    "Ожидаемый отёл, из них коров",
    "Ожидаемый отёл, из них нетелей",
    "Ожидаемые бычки",
    "Ожидаемые тёлочки",
}


def _pick_working_dsn() -> str:
    for dsn in DEFAULT_DSN_CANDIDATES:
        if not dsn:
            continue
        engine = create_engine(dsn, echo=False, future=True)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return dsn
        except Exception:
            continue
    raise RuntimeError(
        "Не удалось подключиться к БД ни по POSTGRES_DSN, ни по localhost:15432. "
        "Запусти скрипт там, где виден Postgres приложения."
    )


WORKING_DSN = _pick_working_dsn()
os.environ["POSTGRES_DSN"] = WORKING_DSN

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtesting import (  # noqa: E402
    actual_birth_stats_month_from_tables,
    actual_nonbirth_snapshot_from_tables,
    backtest_percent_error,
    collect_recent_target_months,
    month_end_shift,
    pred_metric_value,
    target_fact_month_complete_from_tables,
)
from core.constants import INDICATORS  # noqa: E402
from core.helpers import month_end, norm_label  # noqa: E402
from core.params import get_or_compute_subdivision_params  # noqa: E402
from forecast_dynamic import compute_forecast_dynamic_from_tables, latest_data_date  # noqa: E402
from ui.tab3_farm_parts.storage import _load_farm_tables_from_db, _subdivision_status_df_from_db  # noqa: E402


BACKTEST_TARGETS = list(INDICATORS) + [
    "Доля бычков среди рождений, %",
    "Доля тёлочек среди рождений, %",
]


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _metric_unit(metric: str) -> str:
    return "pct" if metric in PERCENT_TARGETS else "heads"


def _summarize_metric_rows(df: pd.DataFrame) -> dict[str, Any]:
    metric = _safe_str(df["Показатель"].iloc[0])
    is_pct = metric in PERCENT_TARGETS

    work = df.copy()
    work["Ошибка"] = pd.to_numeric(work["Ошибка"], errors="coerce")
    work["Факт"] = pd.to_numeric(work["Факт"], errors="coerce")
    work["Прогноз"] = pd.to_numeric(work["Прогноз"], errors="coerce")
    work["APE, %"] = pd.to_numeric(work["APE, %"], errors="coerce")

    mae = float(work["Ошибка"].abs().mean()) if not work.empty else float("nan")
    bias = float(work["Ошибка"].mean()) if not work.empty else float("nan")

    if is_pct:
        perc_series = work["APE, %"].dropna()
        perc_err = float(perc_series.mean()) if not perc_series.empty else float("nan")
    else:
        scale = work["Прогноз"].abs() + work["Факт"].abs()
        stable_mask = scale >= 20.0
        den = float(scale.loc[stable_mask].sum())
        num = float(work.loc[stable_mask, "Ошибка"].abs().sum())
        perc_err = (200.0 * num / den) if den > 1e-9 else float("nan")

    return {
        "Погрешность, %": perc_err,
        "Смещение, гол./п.п.": bias,
        "Средняя абс. ошибка, гол./п.п.": mae,
        "Средний прогноз": float(work["Прогноз"].mean()) if not work.empty else float("nan"),
        "Средний факт": float(work["Факт"].mean()) if not work.empty else float("nan"),
        "Точек": int(len(work)),
        "Первый месяц": _safe_str(work["Месяц факта"].iloc[0]) if not work.empty else "",
        "Последний месяц": _safe_str(work["Месяц факта"].iloc[-1]) if not work.empty else "",
    }


def _build_subdivision_metric_rows(
    subdivision: str,
    *,
    months: int,
    horizon: int,
    complete_only: bool,
    status_row: pd.Series,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tables = _load_farm_tables_from_db(subdivision)
    base_date = latest_data_date(tables)
    last_me = month_end(base_date.year, base_date.month)

    params = dict(get_or_compute_subdivision_params(subdivision, tables))
    params["DISABLE_CAPACITY"] = True

    target_months_by_metric: dict[str, list[Any]] = {}
    all_target_months: set[Any] = set()
    for metric in BACKTEST_TARGETS:
        if complete_only:
            metric_months = collect_recent_target_months(
                last_me,
                months,
                complete_checker=lambda d, metric_name=metric: target_fact_month_complete_from_tables(
                    tables,
                    metric_name,
                    d,
                    birth_targets=BIRTH_TARGETS,
                    percent_targets=PERCENT_TARGETS,
                ),
                search_limit_months=max(24, int(months) * 12),
            )
        else:
            metric_months = [month_end_shift(last_me, -i) for i in range(months - 1, -1, -1)]
        target_months_by_metric[metric] = metric_months
        all_target_months.update(metric_months)

    month_cache: dict[Any, dict[str, Any]] = {}
    for target_me in sorted(all_target_months):
        as_of_me = month_end_shift(target_me, -int(horizon))
        pred_vals = compute_forecast_dynamic_from_tables(
            tables,
            target_me,
            overrides=params,
            as_of_date=as_of_me,
        ) or {}
        nmap = {norm_label(k): v for k, v in pred_vals.items()}
        fact_birth = actual_birth_stats_month_from_tables(
            tables.get("calv", pd.DataFrame()),
            tables.get("ins", pd.DataFrame()),
            target_me,
        )
        fact_snapshot = actual_nonbirth_snapshot_from_tables(
            tables.get("calv", pd.DataFrame()),
            tables.get("ins", pd.DataFrame()),
            tables.get("dry", pd.DataFrame()),
            tables.get("disp", pd.DataFrame()),
            target_me,
        )
        month_cache[target_me] = {
            "as_of_me": as_of_me,
            "pred_vals": pred_vals,
            "nmap": nmap,
            "fact_birth": fact_birth,
            "fact_snapshot": fact_snapshot,
        }

    monthly_rows: list[dict[str, Any]] = []
    for metric in BACKTEST_TARGETS:
        for target_me in target_months_by_metric.get(metric, []):
            cached = month_cache[target_me]
            as_of_me = cached["as_of_me"]
            pred_vals = cached["pred_vals"]
            nmap = cached["nmap"]
            fact_birth = cached["fact_birth"]
            fact_snapshot = cached["fact_snapshot"]
            pred_val = float(pred_metric_value(pred_vals, metric, nmap, percent_targets=PERCENT_TARGETS))
            fact_source = fact_birth if (metric in BIRTH_TARGETS or metric in PERCENT_TARGETS) else fact_snapshot
            fact_val = float(fact_source.get(metric, 0.0))
            err = pred_val - fact_val
            ape = backtest_percent_error(pred_val, fact_val, is_pct=(metric in PERCENT_TARGETS))
            monthly_rows.append(
                {
                    "Хозяйство": _safe_str(status_row.get("Хозяйство")),
                    "Подразделение": subdivision,
                    "Последняя дата данных": status_row.get("Последняя дата данных"),
                    "Месяц факта": target_me.strftime("%Y-%m"),
                    "as-of": as_of_me.strftime("%Y-%m"),
                    "Показатель": metric,
                    "Ед.": _metric_unit(metric),
                    "Прогноз": round(pred_val, 1),
                    "Факт": round(fact_val, 1),
                    "Ошибка": round(err, 1),
                    "APE, %": None if ape is None else round(float(ape), 1),
                }
            )

    monthly_df = pd.DataFrame(monthly_rows)
    summary_rows: list[dict[str, Any]] = []
    if monthly_df.empty:
        for metric in BACKTEST_TARGETS:
            summary_rows.append(
                {
                    "Хозяйство": _safe_str(status_row.get("Хозяйство")),
                    "Подразделение": subdivision,
                    "Последняя дата данных": status_row.get("Последняя дата данных"),
                    "Показатель": metric,
                    "Ед.": _metric_unit(metric),
                    "Погрешность, %": float("nan"),
                    "Смещение, гол./п.п.": float("nan"),
                    "Средняя абс. ошибка, гол./п.п.": float("nan"),
                    "Средний прогноз": float("nan"),
                    "Средний факт": float("nan"),
                    "Точек": 0,
                    "Первый месяц": "",
                    "Последний месяц": "",
                    "Глубина истории": int(months),
                    "Горизонт as-of": int(horizon),
                }
            )
        return summary_rows, monthly_rows

    for metric, grp in monthly_df.groupby("Показатель", sort=False):
        metric_summary = _summarize_metric_rows(grp)
        summary_rows.append(
            {
                "Хозяйство": _safe_str(status_row.get("Хозяйство")),
                "Подразделение": subdivision,
                "Последняя дата данных": status_row.get("Последняя дата данных"),
                "Показатель": metric,
                "Ед.": _metric_unit(metric),
                **metric_summary,
                "Глубина истории": int(months),
                "Горизонт as-of": int(horizon),
            }
        )
    return summary_rows, monthly_rows


def export_report(
    *,
    output_path: str | Path,
    months: int,
    horizon: int,
    complete_only: bool,
    subdivision_filter: str = "",
) -> Path:
    status = _subdivision_status_df_from_db().copy()
    if "Статус" in status.columns:
        status = status[status["Статус"].astype(str) == "готово"].copy()
    if subdivision_filter:
        mask = status["Подразделение"].astype(str).str.upper().str.contains(str(subdivision_filter).upper(), regex=False)
        status = status[mask].copy()
    status = status.sort_values(["Хозяйство", "Подразделение"], kind="mergesort").reset_index(drop=True)

    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []

    total = max(len(status), 1)
    for idx, row in status.iterrows():
        subdivision = _safe_str(row.get("Подразделение"))
        print(f"[{idx + 1}/{total}] {subdivision}")
        try:
            one_summary, one_monthly = _build_subdivision_metric_rows(
                subdivision,
                months=months,
                horizon=horizon,
                complete_only=complete_only,
                status_row=row,
            )
            summary_rows.extend(one_summary)
            monthly_rows.extend(one_monthly)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            for metric in BACKTEST_TARGETS:
                summary_rows.append(
                    {
                        "Хозяйство": _safe_str(row.get("Хозяйство")),
                        "Подразделение": subdivision,
                        "Последняя дата данных": row.get("Последняя дата данных"),
                        "Показатель": metric,
                        "Ед.": _metric_unit(metric),
                        "Погрешность, %": float("nan"),
                        "Смещение, гол./п.п.": float("nan"),
                        "Средняя абс. ошибка, гол./п.п.": float("nan"),
                        "Средний прогноз": float("nan"),
                        "Средний факт": float("nan"),
                        "Точек": 0,
                        "Первый месяц": "",
                        "Последний месяц": "",
                        "Глубина истории": int(months),
                        "Горизонт as-of": int(horizon),
                        "Ошибка расчёта": str(exc),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    monthly_df = pd.DataFrame(monthly_rows)

    if summary_df.empty:
        raise RuntimeError("Не удалось собрать ни одной строки backtesting-отчёта.")

    pct_pivot = summary_df.pivot(index="Подразделение", columns="Показатель", values="Погрешность, %").reset_index()
    bias_pivot = summary_df.pivot(index="Подразделение", columns="Показатель", values="Смещение, гол./п.п.").reset_index()
    mae_pivot = summary_df.pivot(index="Подразделение", columns="Показатель", values="Средняя абс. ошибка, гол./п.п.").reset_index()
    overall_metric = (
        summary_df.groupby("Показатель", as_index=False)
        .agg(
            Подразделений=("Подразделение", "nunique"),
            Средняя_погрешность_pct=("Погрешность, %", "mean"),
            Среднее_смещение=("Смещение, гол./п.п.", "mean"),
            Средняя_абс_ошибка=("Средняя абс. ошибка, гол./п.п.", "mean"),
        )
        .sort_values("Показатель", kind="mergesort")
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary_long", index=False)
        pct_pivot.to_excel(writer, sheet_name="pct_error_pivot", index=False)
        bias_pivot.to_excel(writer, sheet_name="bias_pivot", index=False)
        mae_pivot.to_excel(writer, sheet_name="mae_pivot", index=False)
        overall_metric.to_excel(writer, sheet_name="overall_by_metric", index=False)
        monthly_df.to_excel(writer, sheet_name="monthly_details", index=False)
        status.to_excel(writer, sheet_name="subdivision_info", index=False)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="exports/subdivision_backtesting_report.xlsx")
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--subdivision", default="", help="Optional substring filter for one subdivision")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Use recent months even if fact month is incomplete",
    )
    args = parser.parse_args()

    out = export_report(
        output_path=args.output,
        months=int(args.months),
        horizon=int(args.horizon),
        complete_only=not bool(args.include_incomplete),
        subdivision_filter=str(args.subdivision or ""),
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
