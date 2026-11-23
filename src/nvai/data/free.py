"""Data without credentials: yfinance prices and the daily risk-free rate.

yfinance: adjusted close gives the total return (dividends reinvested); close gives the
price return; dividends are the cash amounts per share on ex-dates.

Risk-free: 3-month T-bill (FRED DTB3, or yfinance ^IRX when FRED is unreachable),
annualized percent -> daily /252. The Ken French daily RF is also saved, but it is rounded
to 0.01% per day, so `load.risk_free()` uses it only where the T-bill is missing.
"""

from __future__ import annotations

import io
import os
import re
import zipfile

import pandas as pd
import requests

from nvai.config import all_tickers, clean_env, raw_dir, settings

HEADERS = {"User-Agent": "research-data-client/0.1"}


def pull_yfinance(tickers: list[str] | None = None) -> pd.DataFrame:
    import yfinance as yf

    s = settings()["sample"]
    tickers = tickers or all_tickers()
    px = yf.download(tickers, start=s["start"], end=s["end"], auto_adjust=False,
                     actions=True, progress=False, group_by="column")
    out = []
    for t in tickers:
        d = pd.DataFrame({
            "adj_close": px["Adj Close"][t], "close": px["Close"][t],
            "dividends": px["Dividends"][t] if "Dividends" in px else 0.0,
        }).dropna(subset=["close"])
        d["ret"] = d["adj_close"].pct_change()      # total return
        d["retx"] = d["close"].pct_change()         # price return (close is split- but not dividend-adjusted)
        d["ticker"] = t
        out.append(d.reset_index().rename(columns={"Date": "date"}))
    df = pd.concat(out, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df.to_parquet(raw_dir("free") / "yfinance.parquet", index=False)
    return df


def parse_french_daily(text: str) -> pd.DataFrame:
    """First block of YYYYMMDD rows from a French daily CSV, converted from percent."""
    header, rows = None, []
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if re.fullmatch(r"\d{8}", cells[0] or ""):
            rows.append(cells)
        elif rows:
            break
        elif len(cells) > 1 and cells[0] == "" and any(cells[1:]):
            header = cells[1:]
    if header is None or not rows:
        raise ValueError("no daily block found")
    df = pd.DataFrame([r[1:] for r in rows], columns=[h.replace("-", "_") for h in header],
                      dtype=float)
    df.index = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    df.index.name = "date"
    return df.mask(df <= -99.99) / 100.0


def pull_french() -> pd.DataFrame:
    url = settings()["sources"]["french_base"] + "F-F_Research_Data_Factors_daily_CSV.zip"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    df = parse_french_daily(text)
    df.reset_index().to_parquet(raw_dir("free") / "ff3_daily.parquet", index=False)
    return df


def _fred(series: str) -> pd.DataFrame:
    src = settings()["sources"]
    key = clean_env(os.environ.get("FRED_API_KEY"))
    if key:
        r = requests.get(src["fred_api"], timeout=60, params={
            "series_id": series, "api_key": key, "file_type": "json"})
        r.raise_for_status()
        return pd.DataFrame(r.json()["observations"])[["date", "value"]]
    r = requests.get(src["fred_csv"].format(series=series), headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "value"]
    return df


def pull_tbill() -> pd.DataFrame:
    """3-month T-bill yield, annualized decimal: FRED DTB3, or yfinance ^IRX if FRED is
    unreachable (the public fredgraph CSV often times out for scripts without an API key)."""
    try:
        df = _fred("DTB3")
        df["source"] = "fred_dtb3"
    except requests.RequestException as e:
        print(f"[free] FRED unavailable ({type(e).__name__}); using yfinance ^IRX")
        import yfinance as yf

        irx = yf.download("^IRX", start=settings()["sample"]["start"], progress=False,
                          auto_adjust=False)["Close"].squeeze()
        df = irx.rename("value").reset_index().rename(columns={"Date": "date"})
        df["source"] = "yf_irx"
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["value"] = pd.to_numeric(df["value"], errors="coerce") / 100.0  # percent -> decimal
    df = df.dropna(subset=["value"])
    df.to_parquet(raw_dir("free") / "tbill_3m.parquet", index=False)
    return df


def pull_all() -> None:
    print("[free] yfinance"); pull_yfinance()
    print("[free] Ken French daily RF"); pull_french()
    print("[free] 3m T-bill"); pull_tbill()
