"""WRDS pulls: CRSP daily returns, Markit borrow fees, ETF Global holdings, OptionMetrics
implied borrow rates. Everything is saved as parquet under data/raw/wrds/.

Login: WRDS_USERNAME from .env (CR/whitespace stripped), password from ~/.pgpass.

Notes on the tables (checked 2026-11-22):
- CRSP: legacy crsp.dsf/stocknames stop at 2024-12-31; crsp.dsf_v2 (CIZ) runs to
  2025-12-31. Years <= legacy_end_year come from dsf, later years from dsf_v2, renamed to
  the legacy names by `normalize_ciz`. ETFs (shrcd 73) and ADRs (TSM, ASML, ARM) are kept:
  we select by permno, not by share code.
- ticker -> permno uses the crsp.stocknames row with the latest end date, so old users of
  a recycled ticker (e.g. ARM = Armstrong Rubber, AIQ = Alliance Imaging) are skipped.
- Markit MSF uses 9-digit CUSIPs; CRSP uses 8. We match the first 8 Markit digits against
  every CRSP ncusip the permno had since the sample start (plus the dsf_v2 cusip).
  Common stocks are in markit_msf_analytics_eqty_amer; ETFs and ADRs only appear in
  markit_msf_analytics_eqty_other. Markit has several rows per security-day (one per
  listing, e.g. "NVDA UW" and "NVDA* MF", "TSM UN" and "TSFA GF"); all are saved and
  `load.borrow_fees` keeps the main one.
  indicativefee / indicativerebate are annualized decimals (fee + rebate ~ policy rate).
- ETF Global constituents are daily from 2012 to 2026-10-30. We save month-end holdings
  for each ETF plus the daily NVDA weight.
- OptionMetrics borrate: put-call-parity implied borrow rate by expiration, decimals,
  -99.99 = missing. Linked to CRSP via wrdsapps_link_crsp_optionm.opcrsphist.
"""

from __future__ import annotations

import pandas as pd

from nvai.config import all_tickers, raw_dir, settings, wrds_username


def connect():
    import wrds

    user = wrds_username()
    if not user:
        raise OSError("Set WRDS_USERNAME in .env (see .env.example)")
    return wrds.Connection(wrds_username=user)


def _in(values) -> str:
    """SQL IN-list. Values come from our own config, never from user input."""
    return ",".join(f"'{v}'" if isinstance(v, str) else str(int(v)) for v in values)


def _out():
    return raw_dir("wrds")


# ---------------------------------------------------------------- CRSP identifiers
def pull_permno_map(db, tickers: list[str]) -> pd.DataFrame:
    """One row per ticker: current permno, plus every ncusip that permno used since start."""
    start = settings()["sample"]["start"]
    names = db.raw_sql(
        f"select permno, ticker, ncusip, comnam, shrcd, namedt, nameenddt "
        f"from crsp.stocknames where ticker in ({_in(tickers)})",
        date_cols=["namedt", "nameenddt"])
    # latest by name END date: TSM also has a short-lived 2000 share line (permno 88315)
    # with a later start date than the main ADR (85442).
    latest = names.sort_values(["nameenddt", "namedt"]).groupby("ticker").tail(1)
    m = latest[["ticker", "permno", "comnam", "shrcd"]].copy()
    allnames = db.raw_sql(
        f"select permno, ncusip, nameenddt from crsp.stocknames "
        f"where permno in ({_in(m.permno)})", date_cols=["nameenddt"])
    allnames = allnames[allnames.nameenddt >= start]
    v2 = db.raw_sql(f"select distinct permno, cusip as ncusip from crsp.dsf_v2 "
                    f"where permno in ({_in(m.permno)}) and dlycaldt >= '2025-01-01'")
    cus = pd.concat([allnames[["permno", "ncusip"]], v2]).dropna().drop_duplicates()
    m = m.merge(cus.groupby("permno")["ncusip"].agg(lambda s: sorted(set(s))).rename("cusips"),
                left_on="permno", right_index=True, how="left")
    missing = sorted(set(tickers) - set(m.ticker))
    if missing:
        print(f"[crsp] no stocknames match for {missing}")
    m.to_parquet(_out() / "permno_map.parquet", index=False)
    return m


# ---------------------------------------------------------------- CRSP daily
LEGACY_SQL = """
select permno, date, ret, retx, prc, shrout, vol
from crsp.dsf where permno in ({permnos}) and date between '{start}' and '{end}'
"""
CIZ_SQL = """
select permno, dlycaldt, dlyret, dlyretx, dlyprc, shrout, dlyvol
from crsp.dsf_v2 where permno in ({permnos}) and dlycaldt between '{start}' and '{end}'
"""
CIZ_RENAME = {"dlycaldt": "date", "dlyret": "ret", "dlyretx": "retx", "dlyprc": "prc",
              "dlyvol": "vol"}


def normalize_ciz(df: pd.DataFrame) -> pd.DataFrame:
    """Rename a dsf_v2 extract to the legacy dsf column names."""
    out = df.rename(columns=CIZ_RENAME).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["source"] = "ciz"
    return out[["permno", "date", "ret", "retx", "prc", "shrout", "vol", "source"]]


def pull_crsp(db, pmap: pd.DataFrame) -> pd.DataFrame:
    s = settings()
    start, end = s["sample"]["start"], s["sample"]["end"]
    legacy_end = f"{s['wrds']['legacy_end_year']}-12-31"
    permnos = _in(pmap.permno)
    leg = db.raw_sql(LEGACY_SQL.format(permnos=permnos, start=start, end=legacy_end),
                     date_cols=["date"])
    leg["source"] = "legacy"
    ciz = normalize_ciz(db.raw_sql(
        CIZ_SQL.format(permnos=permnos, start=f"{int(legacy_end[:4]) + 1}-01-01", end=end),
        date_cols=["dlycaldt"]))
    df = pd.concat([leg, ciz], ignore_index=True)
    df = df.merge(pmap[["permno", "ticker"]], on="permno", how="left")
    df.to_parquet(_out() / "crsp_daily.parquet", index=False)
    return df


# ---------------------------------------------------------------- Markit borrow fees
MARKIT_COLS = ("datadate, cusip, isin, bb_ticker, instrumentname, indicativefee, "
               "indicativerebate, indicativefee1day, indicativefee7day, dcbs, utilisation, "
               "lendablevalue, valueonloan, shortloanvalue")


def pull_markit(db, pmap: pd.DataFrame) -> pd.DataFrame:
    s = settings()
    cus2tic = {c: t for t, cs in zip(pmap.ticker, pmap.cusips)
               for c in (list(cs) if cs is not None else [])}
    out = []
    for lib, prefix in s["wrds"]["markit_tables"].items():
        tables = set(db.list_tables(lib))
        for y in range(int(s["sample"]["start"][:4]), int(s["sample"]["end"][:4]) + 1):
            tbl = f"{prefix}{y}"
            if tbl not in tables:
                continue
            print(f"[markit] {tbl}")
            d = db.raw_sql(f"select {MARKIT_COLS}, marketarea from {lib}.{tbl} "
                           f"where left(cusip, 8) in ({_in(cus2tic)})", date_cols=["datadate"])
            d["markit_table"] = prefix
            out.append(d)
    df = pd.concat(out, ignore_index=True)
    df["ticker"] = df.cusip.str[:8].map(cus2tic)
    df.to_parquet(_out() / "markit_borrow.parquet", index=False)
    return df


# ---------------------------------------------------------------- ETF holdings
ETFG_MONTHLY_SQL = """
with m as (
  select composite_ticker, max(as_of_date) as d
  from etfg_constituents.constituents
  where composite_ticker in ({etfs}) and as_of_date >= '{start}'
  group by composite_ticker, date_trunc('month', as_of_date)
)
select c.as_of_date, c.composite_ticker, c.constituent_ticker, c.constituent_name,
       c.cusip, c.weight, c.market_value, c.shares_held, c.asset_class
from etfg_constituents.constituents c
join m on c.composite_ticker = m.composite_ticker and c.as_of_date = m.d
"""
ETFG_NVDA_SQL = """
select as_of_date, composite_ticker, weight, market_value
from etfg_constituents.constituents
where composite_ticker in ({etfs}) and as_of_date >= '{start}'
  and (constituent_ticker = 'NVDA' or cusip = '67066G104')
"""


def pull_etf_holdings(db) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = settings()
    etfs = _in(s["universe"]["etfs"] + [s["universe"]["market"]])
    start = s["sample"]["start"]
    hold = db.raw_sql(ETFG_MONTHLY_SQL.format(etfs=etfs, start=start), date_cols=["as_of_date"])
    hold.to_parquet(_out() / "etf_holdings_monthly.parquet", index=False)
    nv = db.raw_sql(ETFG_NVDA_SQL.format(etfs=etfs, start=start), date_cols=["as_of_date"])
    nv = nv.drop_duplicates(["as_of_date", "composite_ticker"])
    nv.to_parquet(_out() / "etf_nvda_weight_daily.parquet", index=False)
    return hold, nv


# ---------------------------------------------------------------- OptionMetrics
def pull_optionm_borrow(db, pmap: pd.DataFrame) -> pd.DataFrame:
    s = settings()
    link = db.raw_sql(f"select secid, permno, sdate, edate, score "
                      f"from wrdsapps_link_crsp_optionm.opcrsphist "
                      f"where permno in ({_in(pmap.permno)})", date_cols=["sdate", "edate"])
    # score 1 = best link (CUSIP and ticker agree); keep the best, most recent secid per permno
    link = link.sort_values(["score", "edate"], ascending=[False, True]).groupby("permno").tail(1)
    sec2tic = dict(zip(link.secid.astype(int),
                       link.permno.map(dict(zip(pmap.permno, pmap.ticker)))))
    tables = set(db.list_tables("optionm"))
    out = []
    for y in range(int(s["sample"]["start"][:4]), int(s["sample"]["end"][:4]) + 1):
        if f"borrate{y}" not in tables:  # borrate stops at 2025-08-29 (checked 2026-11-22)
            continue
        print(f"[optionm] {y}")
        out.append(db.raw_sql(
            f"select securityid, date, expirationdate, days, borrowrate from optionm.borrate{y} "
            f"where securityid in ({_in(sec2tic)}) and borrowrate > -99", date_cols=["date"]))
    df = pd.concat(out, ignore_index=True)
    df["ticker"] = df.securityid.astype(int).map(sec2tic)
    df.to_parquet(_out() / "optionm_borrate.parquet", index=False)
    return df


def pull_all() -> None:
    db = connect()
    try:
        print("[crsp] permno map")
        pmap = pull_permno_map(db, all_tickers())
        print("[crsp] daily"); pull_crsp(db, pmap)
        pull_markit(db, pmap)
        print("[etfg] holdings"); pull_etf_holdings(db)
        pull_optionm_borrow(db, pmap)
    finally:
        db.close()
