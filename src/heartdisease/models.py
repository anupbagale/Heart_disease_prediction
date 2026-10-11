"""Preprocessing pipeline and candidate models.

Every preprocessing step (imputation, scaling, one-hot encoding) lives inside
an sklearn ``Pipeline``. When the pipeline is cross-validated or fitted on a
training cohort, these steps learn only from the training data, which removes
the leakage present in the original notebook (imputer fitted before the split).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import BINARY, CATEGORICAL, NUMERIC

RANDOM_STATE = 42

MODEL_NAMES = {
    "logreg": "Logistic regression",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "mlp": "Neural network (MLP)",
}


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    """Impute, scale and encode only the columns present in ``features``."""
    num = [f for f in NUMERIC if f in features]
    binary = [f for f in BINARY if f in features]
    cat = [f for f in CATEGORICAL if f in features]
    transformers = []
    if num:
        transformers.append(
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num)
        )
    if binary:
        transformers.append(
            ("bin", SimpleImputer(strategy="most_frequent"), binary)
        )
    if cat:
        transformers.append(
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat)
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=True)


def _estimators(seed: int) -> dict:
    """Base estimators and small hyperparameter grids.

    Grids are deliberately small: with 303 patients, large searches mostly fit
    noise. The MLP mirrors the spirit of the original Keras model but is sized
    for a small tabular dataset and regularised with weight decay (alpha).
    """
    return {
        "logreg": (
            LogisticRegression(max_iter=5000),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "rf": (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
            {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 5]},
        ),
        "xgb": (
            XGBClassifier(
                n_estimators=200, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss",
                random_state=seed, n_jobs=1, verbosity=0,
            ),
            {"clf__max_depth": [2, 3], "clf__min_child_weight": [1, 5]},
        ),
        "mlp": (
            # L-BFGS converges reliably on small datasets (no mini-batches needed).
            MLPClassifier(
                hidden_layer_sizes=(16,), solver="lbfgs", max_iter=10000,
                random_state=seed,
            ),
            {"clf__alpha": [0.1, 1.0, 10.0], "clf__hidden_layer_sizes": [(16,), (32, 16)]},
        ),
    }


def make_model(name: str, features: list[str], seed: int = RANDOM_STATE,
               inner_splits: int = 3) -> GridSearchCV:
    """A tuned model: preprocessing + estimator wrapped in an inner CV search.

    Wrapping the search in the returned object means that, when this object is
    itself cross-validated, hyperparameters are chosen inside each outer
    training fold (nested cross-validation).
    """
    estimator, grid = _estimators(seed)[name]
    pipe = Pipeline([("prep", make_preprocessor(features)), ("clf", estimator)])
    inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    return GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=1, refit=True)
