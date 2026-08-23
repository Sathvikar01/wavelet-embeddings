"""Phase-4 visualisations: predictor correlation scatter + ranked-pruning curves."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _save(fig, save_dir: str, basename: str):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, basename + ".png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, basename + ".pdf"),
                bbox_inches="tight")
    plt.close(fig)


def predictor_correlation_bars(
    predictors_report: Dict[str, "PredictorValidationReport"],
    save_dir: str,
    basename: str = "predictor_correlation_bars",
):
    names = list(predictors_report.keys())
    r_vals = [predictors_report[n].pearson_r if not np.isnan(predictors_report[n].pearson_r) else 0.0
              for n in names]
    rho_vals = [predictors_report[n].spearman_rho if not np.isnan(predictors_report[n].spearman_rho) else 0.0
                for n in names]
    tau_vals = [predictors_report[n].kendall_tau if not np.isnan(predictors_report[n].kendall_tau) else 0.0
                for n in names]
    x = np.arange(len(names))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, r_vals, width, label="Pearson r", color="tab:blue")
    ax.bar(x,         rho_vals, width, label="Spearman rho", color="tab:green")
    ax.bar(x + width, tau_vals, width, label="Kendall tau", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Correlation (predicted unimportance vs measured loss)")
    ax.axhline(0, color="black", alpha=0.3)
    ax.set_title("Head-redundancy prediction: rank correlation with measured effect")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, basename)


def predictor_scatter(
    predictors_report: Dict[str, "PredictorValidationReport"],
    save_dir: str,
    basename: str = "predictor_scatter",
):
    n_plots = len(predictors_report)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    for ax, (name, rep) in zip(axes.ravel(), predictors_report.items()):
        x = np.array([p.predicted_unimportance for p in rep.per_head])
        y_cos = np.array([p.cosine_drop for p in rep.per_head])
        ax.scatter(x, y_cos, s=18, alpha=0.7, color="tab:purple")
        ax.set_xlabel(f"{name}: predicted unimportance (-score)")
        ax.set_ylabel("Measured cosine drop")
        ax.set_title(f"{name}\nr={rep.pearson_r:+.3f}  rho={rep.spearman_rho:+.3f}  tau={rep.kendall_tau:+.3f}")
        ax.grid(alpha=0.3)
    for ax in list(axes.ravel())[len(predictors_report):]:
        ax.axis("off")
    fig.suptitle("Predictor scores vs measured ablation loss")
    fig.tight_layout()
    _save(fig, save_dir, basename)


def ranked_pruning_curves(
    aggregate_by_predictor: Dict[str, List["AggregateAblationEffect"]],
    save_dir: str,
    basename: str = "ranked_pruning_curves",
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [
        ("cosine_drop", "Cosine drop"),
        ("kl_div", "KL(orig || ablated)"),
        ("attention_drift", "Attention drift"),
    ]
    for ax, (attr, label) in zip(axes, metrics):
        for name, lst in aggregate_by_predictor.items():
            xs = [a.ratio for a in lst]
            ys = [getattr(a, attr) for a in lst]
            ys = [np.nan_to_num(y, nan=0.0) for y in ys]
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_xlabel("Fraction of heads pruned")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Aggregate effect of pruning the predictively-lowest heads")
    fig.tight_layout()
    _save(fig, save_dir, basename)
