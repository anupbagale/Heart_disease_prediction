"""Run every analysis in the project and write tables and figures to results/.

    python scripts/run_experiments.py            # full run (~15-20 min on 2 CPUs)
    python scripts/run_experiments.py --quick    # smoke test (~2 min), used in CI

Analyses
--------
1. Cohort description and missing-data pattern for the four hospitals.
2. Internal validation on Cleveland: nested, 10x repeated, stratified 5-fold CV
   of four models, with 95 % bootstrap CIs and paired AUC differences.
3. External validation: models trained on all of Cleveland, tested unchanged at
   Hungary, Switzerland and VA Long Beach (13 features and 10 features).
4. Leave-one-site-out validation (10 features): train on three hospitals,
   test on the fourth.
5. Calibration (intercept, slope, reliability curves) and decision curves.
6. Interpretability: logistic-regression odds ratios and XGBoost SHAP values.
7. Subgroup performance by sex.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heartdisease import plots  # noqa: E402
from heartdisease.data import (  # noqa: E402
    BINARY, CATEGORICAL, FEATURE_LABELS, FEATURES, FEATURES_REDUCED, NUMERIC,
    SITES, TARGET, cohort_table, load_all, missingness_table,
)
from heartdisease.evaluation import (  # noqa: E402
    fit_final, net_benefit, paired_auc_difference, repeated_cv_predictions,
    summarise_with_ci,
)
from heartdisease.models import MODEL_NAMES, RANDOM_STATE  # noqa: E402

FEATURE_SETS = {
    "All 13 features": FEATURES,
    "10 features (no slope, ca, thal)": FEATURES_REDUCED,
}
FS_SHORT = {"All 13 features": "full", "10 features (no slope, ca, thal)": "reduced"}

CATEGORY_NAMES = {
    "cp": {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"},
    "restecg": {0: "normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversible defect"},
}
SHORT_LABELS = {
    "age": "Age", "sex": "Male sex", "trestbps": "Resting BP", "chol": "Cholesterol",
    "fbs": "Fasting glucose >120", "thalach": "Max heart rate", "exang": "Exercise angina",
    "oldpeak": "ST depression", "ca": "Vessels on fluoroscopy",
    "cp": "Chest pain", "restecg": "Resting ECG", "slope": "ST slope", "thal": "Thallium",
}


UNITS = {"age": "yr", "trestbps": "mmHg", "chol": "mg/dl", "thalach": "bpm",
         "oldpeak": "mm", "ca": "vessels"}
UNIT_DIGITS = {"age": 0, "trestbps": 0, "chol": 0, "thalach": 0, "oldpeak": 1, "ca": 1}


def readable(name: str) -> str:
    """'cat__cp_4.0' -> 'Chest pain: asymptomatic'."""
    _, feat = name.split("__", 1)
    for c in CATEGORICAL:
        if feat.startswith(c + "_"):
            level = int(float(feat[len(c) + 1:]))
            return f"{SHORT_LABELS[c]}: {CATEGORY_NAMES[c][level]}"
    return SHORT_LABELS[feat]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(quick: bool) -> None:
    n_repeats = 2 if quick else 10
    n_boot = 200 if quick else 2000
    out = ROOT / ("results_quick" if quick else "results")
    tables, figs, preds = out / "tables", out / "figures", out / "predictions"
    for d in (tables, figs, preds):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------ 1
    log("Cohorts and missing data")
    df = load_all()
    cohort_table(df).to_csv(tables / "cohort_characteristics.csv")
    miss = missingness_table(df)
    miss.to_csv(tables / "missingness.csv")
    plots.missingness_heatmap(miss, FEATURE_LABELS, figs / "missingness.png")

    dev = df[df.site == "cleveland"].reset_index(drop=True)
    y_dev = dev[TARGET].to_numpy()

    # ------------------------------------------------------------------ 2
    internal_rows, oof_store, chosen_rows = [], {}, []
    for fs_name, feats in FEATURE_SETS.items():
        for m in MODEL_NAMES:
            log(f"Internal CV: {MODEL_NAMES[m]} / {fs_name}")
            P, chosen = repeated_cv_predictions(m, dev, y_dev, feats, n_repeats=n_repeats)
            oof_store[(fs_name, m)] = P
            s = summarise_with_ci(y_dev, P, n_boot=n_boot)
            for metric, r in s.iterrows():
                internal_rows.append({"feature_set": fs_name, "model": m, "metric": metric, **r})
            for params, n in Counter(json.dumps(c, default=str) for c in chosen).items():
                chosen_rows.append({"feature_set": fs_name, "model": m,
                                    "params": params, "outer_folds": n})
            pd.DataFrame(P.T, columns=[f"repeat_{i}" for i in range(n_repeats)]).assign(
                target=y_dev).to_csv(preds / f"oof_{FS_SHORT[fs_name]}_{m}.csv", index=False)
    internal = pd.DataFrame(internal_rows)
    internal["model_name"] = internal["model"].map(MODEL_NAMES)
    internal.to_csv(tables / "internal_validation_long.csv", index=False)
    pd.DataFrame(chosen_rows).to_csv(tables / "chosen_hyperparameters.csv", index=False)

    diff_rows = []
    for fs_name in FEATURE_SETS:
        for m in MODEL_NAMES:
            if m == "logreg":
                continue
            d = paired_auc_difference(y_dev, oof_store[(fs_name, m)],
                                      oof_store[(fs_name, "logreg")], n_boot=n_boot)
            diff_rows.append({"feature_set": fs_name, "model": MODEL_NAMES[m],
                              "comparison": "minus logistic regression", **d})
    pd.DataFrame(diff_rows).to_csv(tables / "internal_auc_differences.csv", index=False)

    full = "All 13 features"
    auc_full = internal[(internal.feature_set == full) & (internal.metric == "AUC")].set_index("model")
    plots.roc_internal(y_dev, {m: oof_store[(full, m)] for m in MODEL_NAMES}, auc_full,
                       figs / "roc_internal.png")
    plots.calibration_plot(
        [(MODEL_NAMES[m], y_dev, oof_store[(full, m)].mean(0), plots.MODEL_COLORS[m],
          plots.MODEL_MARKERS[m]) for m in MODEL_NAMES],
        "Calibration, Cleveland (out-of-fold)", figs / "calibration_internal.png")

    thr = np.round(np.arange(0.05, 0.81, 0.01), 2)
    nb = {m: np.mean([net_benefit(y_dev, p, thr) for p in oof_store[(full, m)]], axis=0)
          for m in MODEL_NAMES}
    nb_all = net_benefit(y_dev, np.ones_like(y_dev, float), thr)
    pd.DataFrame({"threshold": thr, "treat_all": nb_all, **{MODEL_NAMES[m]: v for m, v in nb.items()}}
                 ).to_csv(tables / "decision_curve.csv", index=False)
    plots.decision_curve(thr, nb, nb_all, y_dev.mean(), figs / "decision_curve.png")

    # ------------------------------------------------------------------ 3
    ext_rows, final_models, ext_preds = [], {}, []
    for fs_name, feats in FEATURE_SETS.items():
        for m in MODEL_NAMES:
            log(f"External validation: {MODEL_NAMES[m]} / {fs_name}")
            model = fit_final(m, dev, y_dev, feats)
            final_models[(fs_name, m)] = model
            for site in ["hungarian", "switzerland", "va"]:
                test = df[df.site == site]
                p = model.predict_proba(test[feats])[:, 1]
                y = test[TARGET].to_numpy()
                s = summarise_with_ci(y, p, n_boot=n_boot)
                for metric, r in s.iterrows():
                    ext_rows.append({"validation": "Cleveland -> site", "feature_set": fs_name,
                                     "model": m, "test_site": site, "metric": metric, **r})
                ext_preds.append(pd.DataFrame({"validation": "cleveland_to_site",
                                               "feature_set": FS_SHORT[fs_name], "model": m,
                                               "test_site": site, "target": y, "prob": p}))

    # ------------------------------------------------------------------ 4
    for m in MODEL_NAMES:
        log(f"Leave-one-site-out: {MODEL_NAMES[m]}")
        for site in SITES:
            train, test = df[df.site != site], df[df.site == site]
            model = fit_final(m, train, train[TARGET].to_numpy(), FEATURES_REDUCED)
            p = model.predict_proba(test[FEATURES_REDUCED])[:, 1]
            y = test[TARGET].to_numpy()
            s = summarise_with_ci(y, p, n_boot=n_boot)
            for metric, r in s.iterrows():
                ext_rows.append({"validation": "Leave one site out", "feature_set":
                                 "10 features (no slope, ca, thal)", "model": m,
                                 "test_site": site, "metric": metric, **r})
            ext_preds.append(pd.DataFrame({"validation": "leave_one_site_out",
                                           "feature_set": "reduced", "model": m,
                                           "test_site": site, "target": y, "prob": p}))
    ext = pd.DataFrame(ext_rows)
    ext["model_name"] = ext["model"].map(MODEL_NAMES)
    ext["test_site_name"] = ext["test_site"].map(SITES)
    ext.to_csv(tables / "external_validation_long.csv", index=False)
    ext_pred_df = pd.concat(ext_preds, ignore_index=True)
    ext_pred_df.to_csv(preds / "external_predictions.csv", index=False)

    # Forest plot: internal CV row + external sites, per feature set, + LOSO panel.
    forest = []
    for fs_name in FEATURE_SETS:
        a = internal[(internal.feature_set == fs_name) & (internal.metric == "AUC")]
        for _, r in a.iterrows():
            forest.append({"feature_set": f"Trained on Cleveland\n{fs_name}",
                           "setting": "Cleveland", "model": r.model,
                           "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high})
        e = ext[(ext.validation == "Cleveland -> site") & (ext.feature_set == fs_name)
                & (ext.metric == "AUC")]
        for _, r in e.iterrows():
            forest.append({"feature_set": f"Trained on Cleveland\n{fs_name}",
                           "setting": SITES[r.test_site], "model": r.model,
                           "estimate": r.estimate, "ci_low": r.ci_low, "ci_high": r.ci_high})
    lo = ext[(ext.validation == "Leave one site out") & (ext.metric == "AUC")]
    for _, r in lo.iterrows():
        forest.append({"feature_set": "Leave one site out\n10 features",
                       "setting": SITES[r.test_site],
                       "model": r.model, "estimate": r.estimate,
                       "ci_low": r.ci_low, "ci_high": r.ci_high})
    forest = pd.DataFrame(forest)
    forest.to_csv(tables / "auc_forest_data.csv", index=False)
    plots.external_forest(
        forest, figs / "external_validation_auc.png",
        note="Cleveland row, left and middle panels: internal nested cross-validation. "
             "All other rows: the model never saw any patient from the hospital it is tested on.")

    # Calibration at external sites: logistic regression, 10 features.
    red = "10 features (no slope, ca, thal)"
    curves = [("Cleveland (internal CV)", y_dev, oof_store[(red, "logreg")].mean(0),
               plots.SITE_COLORS["cleveland"], plots.SITE_MARKERS["cleveland"])]
    for site in ["hungarian", "switzerland", "va"]:
        s = ext_pred_df[(ext_pred_df.validation == "cleveland_to_site")
                        & (ext_pred_df.feature_set == "reduced")
                        & (ext_pred_df.model == "logreg") & (ext_pred_df.test_site == site)]
        curves.append((SITES[site], s.target.to_numpy(), s.prob.to_numpy(),
                       plots.SITE_COLORS[site], plots.SITE_MARKERS[site]))
    plots.calibration_plot(curves, "Calibration across hospitals (logistic regression)",
                           figs / "calibration_external.png", n_bins=5,
                           note="trained on Cleveland, 10 features")

    # ------------------------------------------------------------------ 6
    log("Interpretability")
    best_c = final_models[(full, "logreg")].best_params_["clf__C"]
    num = NUMERIC
    # Fixed category lists keep the columns identical across bootstrap resamples.
    levels = [sorted(CATEGORY_NAMES[c]) for c in CATEGORICAL]
    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(categories=levels, drop="first",
                                        sparse_output=False))]), CATEGORICAL),
    ])
    explain = Pipeline([("prep", prep), ("clf", LogisticRegression(C=best_c, max_iter=5000))])
    explain.fit(dev[FEATURES], y_dev)
    names = explain.named_steps["prep"].get_feature_names_out()
    coefs = [explain.named_steps["clf"].coef_[0]]
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(200 if quick else 1000):
        idx = rng.integers(0, len(dev), len(dev))
        explain.fit(dev[FEATURES].iloc[idx], y_dev[idx])
        coefs.append(explain.named_steps["clf"].coef_[0])
    coefs = np.array(coefs)
    sd = dev[num].std()
    labels = []
    for n in names:
        lab = readable(n)
        f = n.split("__", 1)[1]
        if f in num:
            lab += f" (per SD: {sd[f]:.{UNIT_DIGITS[f]}f} {UNITS[f]})"
        labels.append(lab)
    orr = pd.DataFrame({"feature": names, "label": labels, "or": np.exp(coefs[0]),
                        "ci_low": np.exp(np.quantile(coefs[1:], 0.025, axis=0)),
                        "ci_high": np.exp(np.quantile(coefs[1:], 0.975, axis=0))})
    orr = orr.sort_values("or", ascending=False).reset_index(drop=True)
    orr.to_csv(tables / "logistic_odds_ratios.csv", index=False)
    plots.odds_ratio_forest(orr, figs / "odds_ratios.png")

    import shap
    xgb = final_models[(full, "xgb")].best_estimator_
    Xt = pd.DataFrame(xgb.named_steps["prep"].transform(dev[FEATURES]),
                      columns=xgb.named_steps["prep"].get_feature_names_out())
    sv = shap.TreeExplainer(xgb.named_steps["clf"]).shap_values(Xt)
    Xt.columns = [readable(c) for c in Xt.columns]
    shap_imp = pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)
    shap_imp.rename("mean_abs_shap").to_csv(tables / "shap_importance.csv")
    plots.shap_beeswarm(sv, Xt, figs / "shap_xgboost.png")

    # ------------------------------------------------------------------ 7
    log("Subgroups by sex")
    sub_rows = []
    for m in MODEL_NAMES:
        P = oof_store[(full, m)]
        for code, grp in [(0, "Female"), (1, "Male")]:
            mask = dev["sex"].to_numpy() == code
            s = summarise_with_ci(y_dev[mask], P[:, mask], n_boot=n_boot)
            for metric in ["AUC", "Sensitivity", "Specificity", "Observed prevalence"]:
                sub_rows.append({"model": m, "group": f"{grp} (n={mask.sum()})",
                                 "metric": metric, **s.loc[metric]})
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(tables / "subgroup_sex.csv", index=False)
    plots.subgroup_forest(sub[sub.metric != "Observed prevalence"], figs / "subgroup_sex.png")

    # ------------------------------------------------------------------ summary
    import sklearn
    import xgboost
    (out / "run_info.json").write_text(json.dumps({
        "quick": quick, "n_repeats": n_repeats, "n_boot": n_boot,
        "runtime_minutes": round((time.time() - t0) / 60, 1),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "shap": shap.__version__,
    }, indent=2))
    log(f"Done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="few repeats/bootstraps (smoke test)")
    main(ap.parse_args().quick)
