"""Checks on preprocessing, leakage and metrics."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from heartdisease.data import FEATURES, FEATURES_REDUCED, TARGET, load_site
from heartdisease.evaluation import (
    auc, calibration_intercept_slope, metrics, net_benefit, repeated_cv_predictions,
)
from heartdisease.models import MODEL_NAMES, make_model


@pytest.fixture(scope="module")
def cleveland():
    return load_site("cleveland")


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200), 1)  # many ties
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_perfect_calibration_gives_zero_intercept_unit_slope():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20000)
    y = rng.random(20000) < p
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.05
    assert abs(b - 1) < 0.05


def test_net_benefit_of_treat_all():
    y = np.array([1, 1, 0, 0])
    nb = net_benefit(y, np.ones(4), np.array([0.5]))
    assert nb[0] == pytest.approx(0.5 - 0.5 * 1.0)


def test_preprocessing_learns_only_from_training_rows(cleveland):
    """Imputation statistics must come from the training fold, not the test fold."""
    train = cleveland.iloc[:200]
    test = cleveland.iloc[200:].copy()
    test["chol"] = np.nan  # make the test fold's median useless
    model = make_model("logreg", FEATURES).fit(train[FEATURES], train[TARGET])
    imputer = model.best_estimator_.named_steps["prep"].named_transformers_["num"]["impute"]
    chol_idx = list(imputer.feature_names_in_).index("chol")
    assert imputer.statistics_[chol_idx] == pytest.approx(train["chol"].median())
    assert np.isfinite(model.predict_proba(test[FEATURES])).all()


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_every_model_beats_chance_in_cv(cleveland, name):
    y = cleveland[TARGET].to_numpy()
    P, chosen = repeated_cv_predictions(name, cleveland, y, FEATURES_REDUCED,
                                        n_splits=3, n_repeats=1, n_jobs=1)
    assert P.shape == (1, len(y))
    assert ((P >= 0) & (P <= 1)).all()
    assert auc(y, P[0]) > 0.75
    assert len(chosen) == 3


def test_models_handle_fully_missing_external_columns(cleveland):
    """Swiss cholesterol is entirely missing; prediction must still work."""
    model = make_model("xgb", FEATURES).fit(cleveland[FEATURES], cleveland[TARGET])
    swiss = load_site("switzerland")
    p = model.predict_proba(swiss[FEATURES])[:, 1]
    assert np.isfinite(p).all() and len(p) == len(swiss)


def test_metrics_keys():
    m = metrics(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.6, 0.4]))
    assert {"AUC", "Sensitivity", "Specificity", "Brier",
            "Calibration intercept", "Calibration slope"} <= set(m)
    assert m["Sensitivity"] == 0.5 and m["Specificity"] == 0.5
