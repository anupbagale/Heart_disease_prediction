"""Figures. One consistent style: thin marks, recessive grid, neutral text,
a fixed colour per model/site, and marker shapes as a second, colour-blind- and
print-safe encoding."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .data import SITES  # noqa: E402
from .models import MODEL_NAMES  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
REFERENCE = "#898781"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
MODEL_COLORS = dict(zip(MODEL_NAMES, CATEGORICAL))
MODEL_MARKERS = dict(zip(MODEL_NAMES, MARKERS))
SITE_ORDER = ["cleveland", "hungarian", "switzerland", "va"]
SITE_COLORS = {"cleveland": INK_2, "hungarian": CATEGORICAL[0],
               "switzerland": CATEGORICAL[1], "va": CATEGORICAL[2]}
SITE_MARKERS = {"cleveland": "o", "hungarian": "s", "switzerland": "^", "va": "D"}
BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#86b6ef", "#3987e5", "#256abf", "#104281"])
BLUE_HEAT = LinearSegmentedColormap.from_list(
    "blue_heat", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.labelcolor": INK_2, "legend.fontsize": 8.5,
        "lines.linewidth": 2, "lines.markersize": 6,
    })


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------

def missingness_heatmap(miss: pd.DataFrame, labels: dict, path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(miss.to_numpy(), cmap=BLUE_HEAT, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(miss.shape[1]), miss.columns)
    ax.set_yticks(range(miss.shape[0]), [labels[f] for f in miss.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            v = miss.iat[i, j]
            txt = "–" if v == 0 else ("<1%" if v < 1 else f"{v:.0f}%")
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=SURFACE if v > 55 else INK_2)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    cb.set_label("% missing", color=INK_2)
    ax.set_title("Missing data by feature and hospital", pad=28)
    _save(fig, path)


def roc_internal(y: np.ndarray, oof: dict, auc_table: pd.DataFrame, path: Path) -> None:
    """ROC from the first CV repeat; AUCs in the legend are repeat-averaged."""
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=REFERENCE)
    for name, P in oof.items():
        fpr, tpr, _ = roc_curve(y, P[0])
        a = auc_table.loc[name]
        ax.plot(fpr, tpr, color=MODEL_COLORS[name],
                label=f"{MODEL_NAMES[name]}  {a['estimate']:.3f} "
                      f"({a['ci_low']:.3f}–{a['ci_high']:.3f})")
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", title="AUC (95% CI)", title_fontsize=8.5)
    ax.set_title("Internal validation, Cleveland (nested CV)")
    _save(fig, path)


def _binned(y, p, n_bins):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    bins = np.digitize(p, q[1:-1])
    xs, ys = [], []
    for b in range(n_bins):
        m = bins == b
        if m.sum() >= 3:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def calibration_plot(curves: list[tuple], title: str, path: Path, n_bins: int = 8,
                     note: str | None = None) -> None:
    """``curves``: list of (label, y, p, color, marker)."""
    set_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=REFERENCE, label="Perfect calibration")
    for label, y, p, color, marker in curves:
        xs, ys = _binned(np.asarray(y), np.asarray(p), n_bins)
        ax.plot(xs, ys, color=color, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label)
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed proportion with disease")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    if note:
        ax.text(0.98, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color=MUTED)
    ax.set_title(title)
    _save(fig, path)


def decision_curve(thresholds, nb: dict, nb_all, prevalence, path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(thresholds, nb_all, color=REFERENCE, lw=1.2, ls="--", label="Treat all")
    ax.axhline(0, color=INK_2, lw=1.0, label="Treat none")
    for name, v in nb.items():
        ax.plot(thresholds, v, color=MODEL_COLORS[name], label=MODEL_NAMES[name])
    ax.set_ylim(-0.05, prevalence + 0.04)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_xlabel("Threshold probability for referral")
    ax.set_ylabel("Net benefit")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_title("Decision curve analysis, Cleveland (out-of-fold)")
    _save(fig, path)


def external_forest(df: pd.DataFrame, path: Path, note: str | None = None) -> None:
    """AUC with 95% CI by validation setting, one panel per feature set.

    ``df`` columns: feature_set, setting, model, estimate, ci_low, ci_high.
    """
    set_style()
    panels = list(dict.fromkeys(df["feature_set"]))
    settings = list(dict.fromkeys(df["setting"]))
    models = [m for m in MODEL_NAMES if m in set(df["model"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels) + 1.4, 3.9),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    offsets = np.linspace(-0.27, 0.27, len(models))
    for ax, panel in zip(axes, panels):
        sub = df[df["feature_set"] == panel]
        for k, m in enumerate(models):
            s = sub[sub["model"] == m].set_index("setting").reindex(settings)
            ypos = np.arange(len(settings)) + offsets[k]
            ax.hlines(ypos, s["ci_low"], s["ci_high"], color=MODEL_COLORS[m], lw=1.6)
            ax.plot(s["estimate"], ypos, ls="none", marker=MODEL_MARKERS[m],
                    color=MODEL_COLORS[m], markeredgecolor=SURFACE, markeredgewidth=1,
                    label=MODEL_NAMES[m])
        ax.axvline(0.5, color=REFERENCE, lw=1, ls="--")
        ax.set_title(panel)
        ax.set_xlabel("AUC (95% CI)")
        ax.grid(axis="y", visible=False)
        for i in range(1, len(settings)):
            ax.axhline(i - 0.5, color=GRID, lw=0.6)
    axes[0].set_yticks(range(len(settings)), settings)
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    axes[0].set_xlim(0.45, 1.0)
    fig.suptitle("Discrimination at each hospital (AUC, 95% CI)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    if note:
        fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=7.5, color=MUTED)
    _save(fig, path)


def odds_ratio_forest(orr: pd.DataFrame, path: Path) -> None:
    """``orr`` columns: label, or, ci_low, ci_high (sorted for display)."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.0, 0.28 * len(orr) + 1.0))
    y = np.arange(len(orr))
    sig = (orr["ci_low"] > 1) | (orr["ci_high"] < 1)
    color = np.where(sig, CATEGORICAL[0], MUTED)
    ax.hlines(y, orr["ci_low"], orr["ci_high"], color=color, lw=1.6)
    ax.scatter(orr["or"], y, color=color, s=30, zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.axvline(1, color=INK_2, lw=1)
    ax.set_xscale("log")
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6)
             if orr["ci_low"].min() * 0.9 <= t <= orr["ci_high"].max() * 1.1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks(y, orr["label"])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Odds ratio (95% bootstrap CI), log scale")
    ax.set_title("Penalised logistic regression, Cleveland")
    ax.text(1.0, 1.01, "blue = CI excludes 1", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7.5, color=MUTED)
    _save(fig, path)


def subgroup_forest(df: pd.DataFrame, path: Path) -> None:
    """``df`` columns: model, metric, group, estimate, ci_low, ci_high."""
    set_style()
    metrics = list(dict.fromkeys(df["metric"]))
    models = [m for m in MODEL_NAMES if m in set(df["model"])]
    groups = list(dict.fromkeys(df["group"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics) + 1.3, 2.6),
                             sharey=True)
    offsets = np.linspace(-0.16, 0.16, len(models))
    for ax, met in zip(axes, metrics):
        sub = df[df["metric"] == met]
        for k, m in enumerate(models):
            s = sub[sub["model"] == m].set_index("group").reindex(groups)
            yp = np.arange(len(groups)) + offsets[k]
            ax.hlines(yp, s["ci_low"], s["ci_high"], color=MODEL_COLORS[m], lw=1.6)
            ax.plot(s["estimate"], yp, ls="none", marker=MODEL_MARKERS[m],
                    color=MODEL_COLORS[m], markeredgecolor=SURFACE, markeredgewidth=1,
                    label=MODEL_NAMES[m])
        ax.set_title(met)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0.4, 1.03)
    axes[0].set_yticks(range(len(groups)), groups)
    axes[0].set_ylim(len(groups) - 0.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Performance by sex, Cleveland (out-of-fold, 95% CI)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, path)


def shap_beeswarm(shap_values, X_display: pd.DataFrame, path: Path, max_display=12) -> None:
    import shap
    set_style()
    plt.figure(figsize=(6.4, 4.8))
    shap.summary_plot(shap_values, X_display, max_display=max_display, show=False,
                      cmap=BLUE_SEQ, plot_size=None, color_bar_label="Feature value")
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on log-odds of disease)", color=INK_2)
    ax.set_title("XGBoost feature effects, Cleveland")
    ax.tick_params(axis="y", labelcolor=INK_2)
    _save(fig, path)
