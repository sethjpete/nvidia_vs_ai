"""Pull raw data.  Usage:  python scripts/pull_data.py free|wrds|all"""

import sys

from nvai.data import free, wrds_pull

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("free", "all"):
        free.pull_all()
    if what in ("wrds", "all"):
        wrds_pull.pull_all()
