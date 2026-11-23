# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 06 - Live tracker: long NVDA / short 1 MNQ (MNQH27)
#
# Fills live in `config/trade.toml`. Run `make track` for a printed summary and figure,
# or open this notebook. Prices come from yfinance (delayed and sometimes stale; check
# against your broker). With no fills entered yet, it shows a **what-if** that enters at
# the latest price.

# %%
import datetime as dt
import sys
import tomllib
import warnings

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from nvai import hedge as hg
from nvai.config import FIGURES, ROOT

warnings.filterwarnings("ignore")
pd.set_option("display.width", 140, "display.float_format", lambda v: f"{v:,.2f}")
T = tomllib.loads((ROOT / "config" / "trade.toml").read_text())
MULT = T["mnq_multiplier"]
TODAY = pd.Timestamp(dt.date.today())

# %% [markdown]
# ## Prices

# %%
def latest(ticker):
    """Latest price: last 1-minute bar today (incl. extended hours), else last daily close."""
    t = yf.Ticker(ticker)
    intra = t.history(period="1d", interval="1m", prepost=True)
    if len(intra):
        return float(intra.Close.iat[-1]), intra.index[-1]
    d = t.history(period="5d")
    return (float(d.Close.iat[-1]), d.index[-1]) if len(d) else (np.nan, None)


def daily(ticker, start):
    d = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    d.index = d.index.tz_localize(None).normalize()
    return d.Close


MNQ_CHAIN = [("MNQH27.CME", "MNQ March 2027 (exact)"), ("NQH27.CME", "NQ March 2027 (same price)"),
             ("MNQ=F", "front month, NOT H27 (roll spread!)"),
             ("NQ=F", "front month, NOT H27 (roll spread!)")]
mnq_src = None
for tk, desc in MNQ_CHAIN:
    px, ts = latest(tk)
    if np.isfinite(px):
        mnq_src, mnq_px, mnq_ts = (tk, desc), px, ts
        break
nvda_px, nvda_ts = latest("NVDA")
front_px, _ = latest("MNQ=F")
ndx_px, ndx_ts = latest("^NDX")
if mnq_src is None:            # last resort: NDX x entry basis
    basis = (T["mnq_entry_px"] / ndx_px) if T["mnq_entry_px"] else 1.0
    mnq_src, mnq_px, mnq_ts = ("^NDX x basis", "proxy: index times entry basis"), ndx_px * basis, ndx_ts
prices = pd.DataFrame({
    "price": [nvda_px, mnq_px, front_px, ndx_px],
    "as of": [nvda_ts, mnq_ts, None, ndx_ts],
}, index=["NVDA", f"{T['mnq_contract']} via {mnq_src[0]}", "MNQ=F (front month)", "^NDX"])
print(f"MNQ source: {mnq_src[0]} - {mnq_src[1]}")
if ndx_ts is not None and mnq_ts is not None and ndx_ts.date() != mnq_ts.date():
    print(f"note: ^NDX is from {ndx_ts.date()} but futures trade overnight ({mnq_ts}), so the "
          "futures/index gap below includes the overnight move, not just basis.")
print(f"H27 minus front month: {mnq_px - front_px:+,.2f} pts; H27 / NDX - 1: "
      f"{100 * (mnq_px / ndx_px - 1):+.2f}%")
prices

# %% [markdown]
# ## Position and P&L

# %%
live = T["nvda_shares"] > 0 and T["mnq_entry_px"] > 0
C = T["mnq_contracts"]                                      # -1
if live:
    lots = [(pd.Timestamp(T["entry_date"]), T["nvda_shares"], T["nvda_entry_px"])]
    lots += [(pd.Timestamp(a["date"]), a["shares"], a["price"]) for a in T.get("adds", [])]
    books = {"live": (lots, T["mnq_entry_px"])}
else:
    books = {f"what-if ${n / 1e3:.0f}k": ([(TODAY, round(n / nvda_px), nvda_px)], mnq_px)
             for n in T["what_if"]["nvda_notional"]}
    print("No fills in config/trade.toml yet: showing what-if books entered at the latest price.")

com = T["commissions"]
rows = {}
for name, (lots, f0) in books.items():
    sh = sum(s for _, s, _ in lots)
    cost = sum(s * p for _, s, p in lots)
    pnl_nv = sh * nvda_px - cost
    pnl_fut = C * MULT * (mnq_px - f0)
    fees = len(lots) * com["nvda_per_trade"] + 2 * abs(C) * com["mnq_per_side"]  # incl. exit
    tot = pnl_nv + pnl_fut - fees
    L = sh * nvda_px
    rows[name] = {"NVDA shares": sh, "NVDA avg cost": cost / sh, "NVDA notional": L,
                  "MNQ notional": abs(C) * MULT * mnq_px,
                  "effective hedge ratio": abs(C) * MULT * mnq_px / L,
                  "P&L NVDA": pnl_nv, "P&L MNQ": pnl_fut, "commissions (round trip)": -fees,
                  "P&L combined": tot,
                  "to take-profit $": T["take_profit"] - tot, "to stop $": tot - T["stop"],
                  "to TP, % NVDA residual move": 100 * (T["take_profit"] - tot) / L,
                  "to stop, % NVDA residual move": 100 * (tot - T["stop"]) / L}
POS = pd.DataFrame(rows)
POS

# %% [markdown]
# ## Hedge ratio vs beta, risk and time

# %%
nv_d = daily("NVDA", "2019-01-01").pct_change()
ndx_d = daily("^NDX", "2019-01-01").pct_change()        # futures proxy (carry ignored)
d = pd.concat([nv_d, ndx_d], axis=1, keys=["n", "f"]).dropna()
b63 = hg.rolling_beta(d.n, d.f, 63).iat[-1]
b252 = hg.rolling_beta(d.n, d.f, 252).iat[-1]
bfull = d.n.cov(d.f) / d.f.var()
shrunk = 0.5 * b252 + 0.5 * bfull

days_exit = int(np.busday_count(TODAY.date(), T["exit_by"]))
days_exp = int(np.busday_count(TODAY.date(), T["mnq_expiry"]))
rng = np.random.default_rng(0)
risk = {}
for name, r in POS.items():
    L, F = r["NVDA notional"], r["MNQ notional"]
    pnl_d = L * d.n - F * d.f                               # $ P&L of today's position
    sd = pnl_d.ewm(alpha=0.06).std().iat[-1]                # EWMA (lambda 0.94) $ vol
    x, a, b = r["P&L combined"], T["stop"], T["take_profit"]
    # driftless random walk: P(hit +TP first) = (x - stop) / (TP - stop); E[days] = (x-a)(b-x)/s^2
    p_up = (x - a) / (b - a)
    e_days = (x - a) * (b - x) / sd ** 2
    paths = x + np.cumsum(rng.normal(0, sd, (20_000, max(days_exit, 1))), axis=1)
    hit = (paths >= b) | (paths <= a)
    risk[name] = {"1-day 99% VaR (EWMA)": hg.Z99 * sd,
                  "monthly 99% VaR (EWMA x sqrt21)": hg.Z99 * sd * np.sqrt(21),
                  "daily $ vol (EWMA)": sd,
                  "P(+TP before stop), driftless": p_up,
                  "expected trading days to hit either": e_days,
                  f"P(neither hit by {T['exit_by']})": 1 - hit.any(axis=1).mean()}
    if not a < x < b:                                       # an exit level is already crossed
        for kk in list(risk[name])[3:]:
            risk[name][kk] = np.nan
        print(f"*** {name}: EXIT TRIGGERED, combined P&L {x:+,.0f} is past "
              f"{'take-profit' if x >= b else 'stop'} ***")
    elif TODAY.date() >= T["exit_by"]:
        print(f"*** {name}: exit-by date {T['exit_by']} reached ***")
RISK = pd.DataFrame(risk)
print(f"beta to NDX: 63d {b63:.2f} | 252d {b252:.2f} | full {bfull:.2f} | shrunk {shrunk:.2f}  "
      f"(effective hedge ratio: {', '.join(f'{k}: {v:.2f}' for k, v in POS.loc['effective hedge ratio'].items())})")
print(f"trading days to exit-by {T['exit_by']}: {days_exit}; to {T['mnq_contract']} expiry "
      f"{T['mnq_expiry']}: {days_exp}")
print("With symmetric +/- $5k exits and no drift, it is a coin flip (~50%); the edge is only "
      "whatever NVDA-specific drift you believe in.")
RISK

# %% [markdown]
# ## Daily combined P&L since entry

# %%
if live:
    start = pd.Timestamp(T["entry_date"])
    nv_c = daily("NVDA", start - pd.Timedelta(days=3))
    fu_c = daily(mnq_src[0], start - pd.Timedelta(days=3)) if "^" not in mnq_src[0] \
        else daily("^NDX", start - pd.Timedelta(days=3)) * (T["mnq_entry_px"] / ndx_px)
    idx = nv_c.index[nv_c.index >= start]
    lots, f0 = books["live"]
    sh_t = sum(pd.Series(np.where(idx >= dte, s, 0), idx) for dte, s, _ in lots)
    cost_t = sum(pd.Series(np.where(idx >= dte, s * p, 0), idx) for dte, s, p in lots)
    leg_nv = sh_t * nv_c.reindex(idx) - cost_t
    leg_fu = C * MULT * (fu_c.reindex(idx).ffill() - f0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(idx, leg_nv, color="#2a78d6", lw=1, label="NVDA leg")
    ax.plot(idx, leg_fu, color="#eb6834", lw=1, label="MNQ leg")
    ax.plot(idx, leg_nv + leg_fu, color="#0b0b0b", lw=2, label="combined")
    for y in (T["take_profit"], T["stop"]):
        ax.axhline(y, color="#e34948", ls="--", lw=1)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_title("Live P&L since entry ($)", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(FIGURES / "06_live_pnl.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No fills yet: the P&L chart starts once config/trade.toml has the entry fills.")

# %%
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    print("\n" + POS.to_string(float_format=lambda v: f"{v:,.2f}"))
    print("\n" + RISK.to_string(float_format=lambda v: f"{v:,.2f}"))
