"""Loaders used by the research code. Each reads parquet written by scripts/pull_data.py."""

from __future__ import annotations

import pandas as pd

from nvai.config import raw_dir, settings


def _read(source: str, name: str) -> pd.DataFrame:
    p = raw_dir(source) / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run `make data-free` / `make data-wrds`")
    return pd.read_parquet(p)


def crsp_returns(col: str = "ret") -> pd.DataFrame:
    df = _read("wrds", "crsp_daily")
    df[col] = df[col].astype(float)          # WRDS returns nullable Float64; use numpy floats
    return df.pivot_table(index="date", columns="ticker", values=col).sort_index()


def yf_returns(col: str = "ret") -> pd.DataFrame:
    df = _read("free", "yfinance")
    return df.pivot_table(index="date", columns="ticker", values=col).sort_index()


def returns(col: str = "ret", source: str = "best") -> pd.DataFrame:
    """Wide daily return panel (dates x tickers), decimals.

    col: "ret" = total return, "retx" = price return.
    source: "crsp", "yf", or "best" = CRSP where it has a value, yfinance otherwise.
    CRSP ends 2025-12-31, so "best" means 2026 returns come from yfinance.
    """
    return _mask_predecessors(_returns(col, source))


def _mask_predecessors(wide: pd.DataFrame) -> pd.DataFrame:
    for t, d in settings()["universe"].get("first_date", {}).items():
        if t in wide:
            wide.loc[wide.index < d, t] = float("nan")
    return wide


def _returns(col: str, source: str) -> pd.DataFrame:
    if source == "yf":
        return yf_returns(col)
    try:
        crsp = crsp_returns(col)
    except FileNotFoundError:
        if source == "crsp":
            raise
        return yf_returns(col)
    if source == "crsp":
        return crsp
    yf = yf_returns(col)
    return crsp.combine_first(yf).sort_index()


def return_source() -> pd.DataFrame:
    """Same shape as returns(): 'crsp' / 'yf' / NaN, showing where each value came from."""
    crsp, yf = crsp_returns(), yf_returns()
    src = pd.DataFrame("yf", index=yf.index, columns=yf.columns).where(yf.notna())
    c = pd.DataFrame("crsp", index=crsp.index, columns=crsp.columns).where(crsp.notna())
    return c.combine_first(src).sort_index()


def risk_free() -> pd.Series:
    """Daily risk-free rate (decimal per trading day): 3-month T-bill yield / 252.

    The Ken French daily RF is only published to 0.01% per day (it jumps between 2.52% and
    5.04% a year), too coarse for financing a daily position, so it is used only for days
    the T-bill series is missing.
    """
    tb = _read("free", "tbill_3m").set_index("date")["value"] / 252.0
    ff = _read("free", "ff3_daily").set_index("date")["RF"]
    rf = tb.combine_first(ff).sort_index()
    rf.name = "rf"
    return rf[rf.index >= settings()["sample"]["start"]]


def markit_raw() -> pd.DataFrame:
    return _read("wrds", "markit_borrow")


def borrow_fees(field: str = "indicativefee", fill_default: bool = False) -> pd.DataFrame:
    """Wide (date x ticker) annualized Markit borrow fee, one row per security-day.

    Markit reports a row per listing (e.g. "NVDA UW", "NVDA* MF"); we keep the row with
    a fee and the largest lendable value, which is the main US line.
    fill_default=True fills gaps with settings costs.default_borrow_fee.
    """
    df = markit_raw().dropna(subset=["ticker"])
    df = df.assign(has_fee=df["indicativefee"].notna())
    df = (df.sort_values(["has_fee", "lendablevalue"], na_position="first")
            .drop_duplicates(["ticker", "datadate"], keep="last"))
    wide = df.pivot(index="datadate", columns="ticker", values=field).sort_index().astype(float)
    wide.index.name = "date"
    wide = _mask_predecessors(wide)
    if fill_default:
        wide = wide.fillna(settings()["costs"]["default_borrow_fee"])
    return wide


def etf_holdings() -> pd.DataFrame:
    """Month-end ETF holdings. ETF Global repeats the whole fund on some dates (weights
    then sum to 2), so exact duplicate holdings rows are dropped."""
    df = _read("wrds", "etf_holdings_monthly")
    return df.drop_duplicates(["composite_ticker", "as_of_date", "constituent_ticker", "cusip"])


def nvda_weight() -> pd.DataFrame:
    """Daily NVDA weight (decimal) in each ETF, wide by ETF."""
    df = _read("wrds", "etf_nvda_weight_daily")
    w = df.pivot(index="as_of_date", columns="composite_ticker", values="weight").sort_index()
    # ETF Global has a few bad days (e.g. QQQ near 0 in mid-2020); drop values below 20% of
    # the 21-day rolling median. Genuine rebalances (XLK's 2024 capping) are far smaller.
    med = w.rolling(21, min_periods=5, center=True).median()
    return w.mask(w < 0.2 * med)


def optionm_borrow(target_days: int = 30) -> pd.DataFrame:
    """OptionMetrics implied borrow rate (annualized decimal), wide by ticker, taking the
    expiration closest to `target_days` each day."""
    df = _read("wrds", "optionm_borrate")
    df = df[df.days > 0].copy()
    df["gap"] = (df.days - target_days).abs()
    df = df.sort_values("gap").drop_duplicates(["ticker", "date"])
    return df.pivot(index="date", columns="ticker", values="borrowrate").sort_index()
