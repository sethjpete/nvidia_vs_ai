# nvidia_vs_ai

How much of an AI basket should you short against a long NVIDIA position? This repo
estimates NVDA hedge ratios against several baskets of AI stocks and ETFs (QQQ, SMH,
SOXX, XLK, ...), then builds the hedged position's total return net of borrow fees
(Markit) and financing.

## Layout

```
config/settings.toml      sample dates, tickers, basket definitions, default borrow fee
src/nvai/config.py        paths, settings, .env loading
src/nvai/data/free.py     yfinance prices, T-bill and Ken French risk-free rate
src/nvai/data/wrds_pull.py CRSP daily, Markit borrow fees, ETF Global holdings, OptionMetrics borrow
src/nvai/data/load.py     loaders: wide return panel (CRSP, else yfinance), fees, holdings, rf
scripts/pull_data.py      `free` / `wrds` entry point
research/                 ALL research code: 01_hedge_ratios.py (jupytext, # %% cells),
                          its executed .ipynb, and figures/
```

## Setup

```bash
make install            # .venv + editable install
cp .env.example .env    # set WRDS_USERNAME; WRDS password goes in ~/.pgpass
make data               # data-free (no credentials) + data-wrds (~1 minute)
make notebook           # runs research/01_hedge_ratios.py -> .ipynb + figures
make test
```

Raw data lands in `data/raw/` (gitignored; CRSP, Markit and ETF Global are licensed).

## Data notes

- CRSP ends 2025-12-31 (legacy `dsf` to 2024, `dsf_v2` for 2025). 2026 returns come
  from yfinance.
- Markit: common stocks are in `markit_msf_analytics_eqty_amer`, ETFs and ADRs only in
  `markit_msf_analytics_eqty_other`. Matched to CRSP on 8-digit CUSIP.
- OptionMetrics `borrate` ends 2025-08-29.
- ARM starts at its 2023-09 IPO; VRT before its 2020-02-07 listing (a SPAC) is masked.
