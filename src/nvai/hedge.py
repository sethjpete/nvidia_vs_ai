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
