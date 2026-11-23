import pandas as pd

from nvai.config import clean_env
from nvai.data.wrds_pull import normalize_ciz


def test_clean_env_strips_crlf():
    assert clean_env("sp8655\r") == "sp8655"
    assert clean_env(' "abc" \r\n') == "abc"
    assert clean_env("\r") is None


def test_normalize_ciz_renames_to_legacy():
    raw = pd.DataFrame({"permno": [86580], "dlycaldt": ["2025-01-02"], "dlyret": [0.01],
                        "dlyretx": [0.01], "dlyprc": [138.3], "shrout": [24e6], "dlyvol": [1e6]})
    out = normalize_ciz(raw)
    assert list(out.columns) == ["permno", "date", "ret", "retx", "prc", "shrout", "vol", "source"]
    assert out["date"].dtype.kind == "M" and out["source"].iat[0] == "ciz"
