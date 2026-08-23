"""Layer-by-layer evolution plots: per-layer comparison across models.

This module differs from :mod:`layer_progression` (which computes data) in
that ``layer_evolution`` here *compares* the per-layer summaries across
multiple model snapshots in one figure.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def _save(fig, save_dir: str, basename: str):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, basename + ".png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, basename + ".pdf"),
                bbox_inches="tight")
    plt.close(fig)


# Plot keys -> (display_label, attr_name)
PLOT_MAP = [
    ("Total energy",                "mean_total_energy"),
    ("Low freq energy",             "mean_low_freq_energy"),
    ("High freq energy",            "mean_high_freq_energy"),
    ("Energy ratio L/H",            "mean_energy_ratio_LH"),
    ("Shannon entropy",             "mean_shannon_entropy"),
    ("Spectral entropy",            "mean_spectral_entropy"),
    ("Gini sparsity",               "mean_gini"),
    ("Reconstruction err (30%)",   "mean_rec_err_30pct"),
    ("Dominant level std",          "std_dominant_level"),
]


def plot_layer_evolution_across_models(
    summaries_by_model: Dict[str, list],
    save_dir: str,
    basename: str = "layer_evolution_comparison",
):
    """Plot every per-layer metric across all models side by side."""
    n_panels = len(PLOT_MAP)
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.4 * n_rows))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(1, len(summaries_by_model))))
    for ax, (label, attr) in zip(axes.ravel(), PLOT_MAP):
        for i, (m, summaries) in enumerate(summaries_by_model.items()):
            xs = [s.layer for s in summaries]
            ys = [getattr(s, attr) for s in summaries]
            ax.plot(xs, ys, marker="o", label=m, color=cmap[i])
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Layer")
            ax.grid(alpha=0.3)
    # Hide unused axes
    for ax in list(axes.ravel())[n_panels:]:
        ax.axis("off")
    handles, labels_ = axes.ravel()[0].get_legend_handles_labels()
    if labels_:
        fig.legend(handles, labels_, loc="lower center", ncol=3)
    fig.suptitle(
        "Layer progression of wavelet-attention metrics across models",
        y=1.02
    )
    fig.tight_layout()
    _save(fig, save_dir, basename)
