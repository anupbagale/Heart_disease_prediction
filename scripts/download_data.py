"""Re-create data/raw/*.csv directly from the UCI Machine Learning Repository.

The CSVs are committed so the project runs offline, but you can regenerate and
check them against the original source at any time:

    python scripts/download_data.py            # writes to data/raw/
    python scripts/download_data.py --check    # compare with committed files only

Source: https://archive.ics.uci.edu/dataset/45/heart+disease (CC BY 4.0).
The archive contains ``processed.<site>.data`` files: 14 comma-separated
columns, no header, ``?`` for missing values.
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
           "exang", "oldpeak", "slope", "ca", "thal", "num"]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch() -> dict[str, pd.DataFrame]:
    with urllib.request.urlopen(URL, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    out = {}
    for site, name in FILES.items():
        raw = z.read(name).decode("latin-1")
        df = pd.read_csv(io.StringIO(raw), header=None, names=COLUMNS,
                         na_values="?", skipinitialspace=True)
        out[site] = df.apply(pd.to_numeric, errors="coerce")
    return out


def main(check: bool) -> int:
    data = fetch()
    ok = True
    for site, df in data.items():
        target = RAW / f"{site}.csv"
        if check:
            committed = pd.read_csv(target)
            same = committed.shape == df.shape and committed.equals(
                pd.read_csv(io.StringIO(df.to_csv(index=False, float_format="%g"))))
            print(f"{site:12s} {df.shape}  {'matches' if same else 'DIFFERS'}")
            ok &= same
        else:
            RAW.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=RAW, suffix=".tmp") as f:
                df.to_csv(f, index=False, float_format="%g")
            Path(f.name).replace(target)
            print(f"wrote {target.relative_to(RAW.parents[1])}  {df.shape}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="only compare UCI data with the committed CSVs")
    sys.exit(main(ap.parse_args().check))
