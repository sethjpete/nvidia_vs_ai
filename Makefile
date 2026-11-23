PY ?= .venv/bin/python

.PHONY: install data data-free data-wrds notebook test

install:     ## venv + editable install
	python3 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

data: data-free data-wrds

data-free:   ## yfinance prices, Ken French / FRED risk-free rate
	$(PY) scripts/pull_data.py free

data-wrds:   ## CRSP daily, Markit borrow fees, ETF Global holdings, OptionMetrics borrow rates
	$(PY) scripts/pull_data.py wrds

notebook:    ## run the research notebook end to end (writes research/*.ipynb and figures)
	cd research && ../.venv/bin/jupytext --to ipynb --execute 01_hedge_ratios.py

test:
	$(PY) -m pytest -q
