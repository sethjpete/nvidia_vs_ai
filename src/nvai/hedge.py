"""Small, reusable pieces for the hedge research: basket construction, betas, risk.

Conventions: inputs are daily simple returns (decimals) in wide DataFrames indexed by
date; "excess" means over the daily risk-free rate. All betas are slope coefficients of
NVDA on the hedge, so the hedge ratio h (dollars short per dollar long) equals beta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z99 = 2.3263478740408408          # one-sided 99% normal quantile
ES99_NORMAL = 2.6652142203458072  # phi(Z99) / 0.01: normal 99% ES in sigmas


def ew_monthly_basket(ret: pd.DataFrame) -> pd.Series:
    """Equal-weight basket rebalanced at each month start; weights drift within the month.

    Names enter at the first month in which they have a return on the month's first day
    (so ARM joins in Oct 2023). A missing return mid-month counts as zero.
    """
    out = []
    for _, r in ret.groupby(ret.index.to_period("M")):
        live = r.columns[r.iloc[0].notna()]
        if len(live) == 0:
            continue
        growth = (1 + r[live].fillna(0)).cumprod()
        value = growth.mean(axis=1)                       # $1 split equally on day 0
        prev = value.shift(1).fillna(1.0)
        out.append(value / prev - 1)
    return pd.concat(out).rename("basket")


def rolling_beta(y: pd.Series, x: pd.Series, window: int, min_periods: int | None = None
                 ) -> pd.Series:
    """OLS slope of y on x over a trailing window (uses data through day t)."""
    mp = min_periods or int(0.8 * window)
    cov = y.rolling(window, min_periods=mp).cov(x)
    return cov / x.rolling(window, min_periods=mp).var()


def ewma_beta(y: pd.Series, x: pd.Series, halflife: float = 63) -> pd.Series:
    """Exponentially weighted beta (time-varying, smoother than a hard rolling window)."""
    cov = y.ewm(halflife=halflife, min_periods=60).cov(x)
    return cov / x.ewm(halflife=halflife, min_periods=60).var()


def hold_monthly(signal: pd.Series) -> pd.Series:
    """Take the value known at the end of the previous day, fix it on the first trading day
    of each month, and hold it all month (no look-ahead)."""
    lagged = signal.shift(1)
    return lagged.groupby(lagged.index.to_period("M")).transform("first")


def ewma_vol(u: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility; value at t uses returns through t (a forecast for t+1)."""
    var = np.empty(len(u))
    x = u.fillna(0).to_numpy()
    var[0] = np.nanvar(x[:20]) if len(x) >= 20 else x[0] ** 2
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t] ** 2
    return pd.Series(np.sqrt(var), index=u.index)


def hist_var(u: pd.Series, q: float = 0.01) -> float:
    """Historical VaR as a positive loss fraction."""
    return -float(np.nanquantile(u.dropna(), q))


def hist_es(u: pd.Series, q: float = 0.01) -> float:
    """Historical expected shortfall: mean loss on days at or beyond the q quantile."""
    u = u.dropna()
    cut = np.quantile(u, q)
    return -float(u[u <= cut].mean())


def max_drawdown(pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of a cumulative $ P&L path (positive number)."""
    cum = pnl.fillna(0).cumsum()
    return float((cum.cummax() - cum).max())


def monthly_reset_backtest(xs_long: pd.Series, g_long: pd.Series, xs_short: pd.Series,
                           g_short: pd.Series, L_tgt: pd.Series, S_tgt: pd.Series,
                           cost_fn) -> pd.DataFrame:
    """Daily $ P&L of a long/short book whose legs are reset to target notionals on the
    first trading day of each month and drift with prices in between.

    xs_*   : daily excess return earned per $ of notional (short leg: what the short PAYS,
             e.g. QQQ excess return + borrow fee, or the futures return)
    g_*    : daily return the notional drifts with (price/total return of the instrument)
    L_tgt, S_tgt : target $ notionals, read on each month's first day (index = days)
    cost_fn(prev_L, prev_S, tgt_L, tgt_S, day) -> $ cost charged on the reset day
    Returns columns pnl, long, short (notional at the start of each day).
    """
    idx = L_tgt.dropna().index.intersection(S_tgt.dropna().index)
    out, prev_L, prev_S = [], 0.0, 0.0
    for _, d in pd.Series(idx, index=idx).groupby(idx.to_period("M")):
        d = d.index
        tL, tS = float(L_tgt.loc[d[0]]), float(S_tgt.loc[d[0]])
        Lt = tL * (1 + g_long.loc[d]).cumprod().shift(1, fill_value=1.0)
        St = tS * (1 + g_short.loc[d]).cumprod().shift(1, fill_value=1.0)
        pnl = Lt * xs_long.loc[d] - St * xs_short.loc[d]
        pnl.iloc[0] -= cost_fn(prev_L, prev_S, tL, tS, d[0])
        out.append(pd.DataFrame({"pnl": pnl, "long": Lt, "short": St}))
        prev_L = Lt.iat[-1] * (1 + g_long.loc[d[-1]])
        prev_S = St.iat[-1] * (1 + g_short.loc[d[-1]])
    return pd.concat(out)


def perf_stats(p: pd.Series, equity: float, ann: int = 252) -> pd.Series:
    """Summary stats for a daily $ P&L series measured over T-bills."""
    m = p.groupby(p.index.to_period("M")).sum()
    cum = p.cumsum()
    dd = cum - cum.cummax()
    under = dd < 0
    runs = under.groupby((~under).cumsum()).sum()
    down = p[p < 0]
    return pd.Series({
        "annual P&L $": ann * p.mean(), "annual % of equity": 100 * ann * p.mean() / equity,
        "vol $/yr": np.sqrt(ann) * p.std(), "Sharpe": np.sqrt(ann) * p.mean() / p.std(),
        "Sortino": np.sqrt(ann) * p.mean() / np.sqrt((down ** 2).sum() / len(p)),
        "max drawdown $": dd.min(), "max drawdown % equity": 100 * dd.min() / equity,
        "worst day $": p.min(), "worst day": p.idxmin().date(),
        "worst month $": m.min(), "worst month": str(m.idxmin()),
        "months up %": 100 * (m > 0).mean(), "longest drawdown (days)": int(runs.max()),
    })
