from __future__ import annotations

import re

import pandas as pd


_YEAR_FIRST_RE = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T].*)?$")


def parse_mixed_datetime(s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series):
        return pd.to_datetime(s, errors="coerce")

    as_str = s.astype("string").fillna("")
    year_first_mask = as_str.str.match(_YEAR_FIRST_RE, na=False)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    if bool(year_first_mask.any()):
        out.loc[year_first_mask] = pd.to_datetime(
            s.loc[year_first_mask],
            errors="coerce",
            format="mixed",
            dayfirst=False,
        )
    if bool((~year_first_mask).any()):
        out.loc[~year_first_mask] = pd.to_datetime(
            s.loc[~year_first_mask],
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )
    return out


def parse_mixed_date(s: pd.Series) -> pd.Series:
    return parse_mixed_datetime(s).dt.date
