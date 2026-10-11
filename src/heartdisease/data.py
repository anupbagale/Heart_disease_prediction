"""Loading and cleaning the four UCI Heart Disease cohorts.

The UCI Heart Disease dataset (Janosi, Steinbrunn, Pfisterer & Detrano, 1989)
contains patients referred for coronary angiography at four centres:

* Cleveland Clinic Foundation, USA          (n = 303)
* Hungarian Institute of Cardiology, Budapest (n = 294)
* University Hospitals Zurich and Basel, Switzerland (n = 123)
* VA Medical Center, Long Beach, USA        (n = 200)

The outcome ``num`` is the angiographic diagnosis: 0 means < 50 % diameter
narrowing in every major vessel; 1-4 indicate > 50 % narrowing. Following the
original publications, we predict the presence of disease (``num > 0``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SITES = {
    "cleveland": "Cleveland",
    "hungarian": "Hungary",
    "switzerland": "Switzerland",
    "va": "VA Long Beach",
}

TARGET = "target"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Features that are mostly missing outside Cleveland (see data/README.md).
SPARSE_EXTERNAL = ["slope", "ca", "thal"]
FEATURES_REDUCED = [f for f in FEATURES if f not in SPARSE_EXTERNAL]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (male)",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mmHg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Vessels coloured by fluoroscopy",
    "thal": "Thallium stress test",
}


def load_site(site: str, data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load one cohort with a binary ``target`` column and cleaned missing values.

    Cleaning steps (all documented in data/README.md):

    * ``num`` (0-4) is binarised to ``target`` = 1 if ``num > 0``.
    * A serum cholesterol of 0 mg/dl is physiologically impossible; it is the
      Swiss and VA centres' code for "not measured", so it is set to missing.
    * A resting blood pressure of 0 mmHg (one VA record) is set to missing.
    """
    if site not in SITES:
        raise ValueError(f"Unknown site {site!r}; choose from {list(SITES)}")
    df = pd.read_csv(Path(data_dir) / f"{site}.csv")
    df["chol"] = df["chol"].replace(0, np.nan)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)
    df[TARGET] = (df["num"] > 0).astype(int)
    df["site"] = site
    return df


def load_all(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """All four cohorts stacked, with a ``site`` column."""
    return pd.concat([load_site(s, data_dir) for s in SITES], ignore_index=True)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of missing values per feature and site."""
    pct = df.groupby("site")[FEATURES].apply(lambda g: g.isna().mean() * 100).T
    pct = pct[[s for s in SITES if s in pct.columns]]
    pct.columns = [SITES[c] for c in pct.columns]
    return pct.round(1)


def cohort_table(df: pd.DataFrame) -> pd.DataFrame:
    """Basic characteristics per site (Table 1 style)."""
    rows = {}
    for site, g in df.groupby("site", sort=False):
        rows[SITES[site]] = {
            "n": len(g),
            "disease prevalence (%)": round(g[TARGET].mean() * 100, 1),
            "age, mean (SD)": f"{g['age'].mean():.1f} ({g['age'].std():.1f})",
            "male (%)": round(g["sex"].mean() * 100, 1),
            "asymptomatic chest pain (%)": round((g["cp"] == 4).mean() * 100, 1),
            "exercise angina (%)": round(g["exang"].mean() * 100, 1),
        }
    return pd.DataFrame(rows)
