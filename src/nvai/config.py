"""Settings and canonical paths. Everything else imports paths from here."""

from __future__ import annotations

import datetime as dt
import os
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:  # load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DATA = Path(os.environ.get("NVAI_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
FIGURES = ROOT / "research" / "figures"


def clean_env(value: str | None) -> str | None:
    """Strip whitespace and stray CR characters (a .env saved with CRLF line endings
    leaves a trailing \\r that breaks the WRDS login)."""
    if value is None:
        return None
    value = value.replace("\r", "").strip().strip('"').strip("'")
    return value or None


def wrds_username() -> str | None:
    return clean_env(os.environ.get("WRDS_USERNAME"))


@lru_cache
def settings() -> dict:
    with open(ROOT / "config" / "settings.toml", "rb") as f:
        s = tomllib.load(f)
    if s["sample"]["end"] == "today":
        s["sample"]["end"] = dt.date.today().isoformat()
    # every ETF is also a single-name basket
    for etf in s["universe"]["etfs"]:
        s["baskets"].setdefault(etf, [etf])
    return s


def all_tickers() -> list[str]:
    u = settings()["universe"]
    return list(dict.fromkeys([u["target"], u["market"], *u["stocks"], *u["etfs"]]))


def raw_dir(source: str) -> Path:
    p = RAW / source
    p.mkdir(parents=True, exist_ok=True)
    return p
