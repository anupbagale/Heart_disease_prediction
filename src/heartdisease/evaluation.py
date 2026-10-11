"""Validation procedures and performance metrics.

Discrimination, calibration and clinical utility are reported separately, as
recommended by the TRIPOD+AI statement (Collins et al., BMJ 2024).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from .models import RANDOM_STATE, make_model

EPS = 1e-6


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney rank formulation (handles ties)."""
    y = np.asarray(y)
    p = np.asarray(p)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(p).rank().to_numpy()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Calibration-in-the-large (intercept, slope fixed at 1) and calibration slope.

    Ideal values: intercept 0, slope 1. Intercept < 0 means the model
    over-predicts risk on average; slope < 1 means predictions are too extreme.
    """
    y = np.asarray(y, float)
    lp = logit(np.clip(p, EPS, 1 - EPS))

    # Intercept with the linear predictor as an offset (1-D Newton-Raphson).
    a = 0.0
    for _ in range(100):
        mu = expit(a + lp)
        step = (y - mu).sum() / max((mu * (1 - mu)).sum(), EPS)
        a += step
        if abs(step) < 1e-10:
            break

    # Slope: logistic regression of y on the linear predictor (2-D Newton).
    X = np.column_stack([np.ones_like(lp), lp])
    beta = np.zeros(2)
    for _ in range(100):
        mu = expit(X @ beta)
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(beta[1])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Discrimination, threshold and calibration metrics for one set of predictions."""
    y = np.asarray(y)
    p = np.asarray(p)
    pred = p >= threshold
    tp = np.sum(pred & (y == 1))
    tn = np.sum(~pred & (y == 0))
    n1 = np.sum(y == 1)
    n0 = np.sum(y == 0)
    cil, slope = calibration_intercept_slope(y, p)
    return {
        "AUC": auc(y, p),
        "Sensitivity": tp / n1 if n1 else np.nan,
        "Specificity": tn / n0 if n0 else np.nan,
        "Accuracy": (tp + tn) / len(y),
        "Brier": np.mean((p - y) ** 2),
        "Calibration intercept": cil,
        "Calibration slope": slope,
        "Mean predicted risk": p.mean(),
        "Observed prevalence": y.mean(),
    }


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit (Vickers & Elkin, 2006) at each threshold."""
    y = np.asarray(y)
    n = len(y)
    out = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1)) / n
        fp = np.sum(pred & (y == 0)) / n
        out.append(tp - fp * t / (1 - t))
    return np.array(out)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_indices(n: int, n_boot: int, seed: int = RANDOM_STATE,
                      strata: np.ndarray | None = None) -> np.ndarray:
    """Bootstrap resamples of row indices, stratified by outcome if given."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return rng.integers(0, n, size=(n_boot, n))
    idx = np.arange(n)
    parts = []
    for s in np.unique(strata):
        members = idx[strata == s]
        parts.append(rng.choice(members, size=(n_boot, len(members)), replace=True))
    return np.concatenate(parts, axis=1)


def summarise_with_ci(y: np.ndarray, P: np.ndarray, n_boot: int = 2000,
                      seed: int = RANDOM_STATE, threshold: float = 0.5) -> pd.DataFrame:
    """Point estimates and 95 % bootstrap CIs.

    ``P`` has shape (n_repeats, n_patients): one row of out-of-fold predictions
    per CV repeat (or a single row for an external test set). Each metric is
    computed per row and averaged; the bootstrap resamples patients, keeping
    the same resample across rows, so the CI reflects sampling variability of
    the test patients.
    """
    y = np.asarray(y)
    P = np.atleast_2d(P)
    point = pd.DataFrame([metrics(y, p, threshold) for p in P]).mean()
    boots = []
    for idx in bootstrap_indices(len(y), n_boot, seed, strata=y):
        boots.append(pd.DataFrame([metrics(y[idx], p[idx], threshold) for p in P]).mean())
    boots = pd.DataFrame(boots)
    return pd.DataFrame({
        "estimate": point,
        "ci_low": boots.quantile(0.025),
        "ci_high": boots.quantile(0.975),
    })


def paired_auc_difference(y: np.ndarray, P_a: np.ndarray, P_b: np.ndarray,
                          n_boot: int = 2000, seed: int = RANDOM_STATE) -> dict:
    """AUC(a) - AUC(b) with a paired bootstrap CI (same patients resampled)."""
    y = np.asarray(y)
    P_a, P_b = np.atleast_2d(P_a), np.atleast_2d(P_b)

    def diff(idx):
        a = np.mean([auc(y[idx], p[idx]) for p in P_a])
        b = np.mean([auc(y[idx], p[idx]) for p in P_b])
        return a - b

    point = diff(np.arange(len(y)))
    boots = np.array([diff(i) for i in bootstrap_indices(len(y), n_boot, seed, strata=y)])
    return {"difference": point, "ci_low": np.quantile(boots, 0.025),
            "ci_high": np.quantile(boots, 0.975)}


# ---------------------------------------------------------------------------
# Validation procedures
# ---------------------------------------------------------------------------

def _fit_predict(model, X, y, train, test):
    m = clone(model).fit(X.iloc[train], y[train])
    return test, m.predict_proba(X.iloc[test])[:, 1], m.best_params_


def repeated_cv_predictions(name: str, X: pd.DataFrame, y: np.ndarray,
                            features: list[str], n_splits: int = 5,
                            n_repeats: int = 10, n_jobs: int = -1):
    """Nested, repeated, stratified CV.

    Returns an array (n_repeats, n_patients) of out-of-fold predicted
    probabilities, plus the hyperparameters chosen in every outer fold.
    """
    model = make_model(name, features)
    jobs = []
    for r in range(n_repeats):
        outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE + r)
        for train, test in outer.split(X, y):
            jobs.append((r, train, test))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(model, X[features], y, tr, te) for _, tr, te in jobs
    )
    P = np.full((n_repeats, len(y)), np.nan)
    chosen = []
    for (r, _, _), (test, p, params) in zip(jobs, results):
        P[r, test] = p
        chosen.append(params)
    assert not np.isnan(P).any()
    return P, chosen


def fit_final(name: str, X: pd.DataFrame, y: np.ndarray, features: list[str]):
    """Tune with CV on the whole development cohort and refit."""
    return make_model(name, features).fit(X[features], y)
