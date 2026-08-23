"""Track layer-by-layer evolution of attention frequency characteristics.

For every layer L = 0..(n-1):

  * mean entropy across heads (Shannon + spectral)
  * mean energy (low / high / total)
  * mean sparsity (Gini)
  * mean energy ratio (low/high)
  * head_specialization spread (std of dominant level)
  * mean reconstruction error at 30% compression

Saves a CSV "layer_progression.csv" of average metrics per layer, plus PDF/PNG
plots.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from attention import AttentionLoader, AttentionWaveletDecomposer, compute_head_metrics


# --------------------------------------------------------------------------- #
# Per-layer summary
# --------------------------------------------------------------------------- #

@dataclass
class LayerSummary:
    layer: int
    n_heads: int
    mean_total_energy: float
    mean_shannon_entropy: float
    mean_spectral_entropy: float
    mean_gini: float
    mean_energy_ratio_LH: float
    mean_rec_err_30pct: float
    std_dominant_level: float
    mean_low_freq_energy: float
    mean_high_freq_energy: float


def compute_layer_summaries(
    loader: AttentionLoader,
    decomposer: AttentionWaveletDecomposer,
) -> List[LayerSummary]:
    out: List[LayerSummary] = []
    for L in range(loader.n_layers):
        rows: List[Dict[str, float]] = []
        for H in range(loader.n_heads):
            head = loader.load_head(L, H)
            rows.append(compute_head_metrics(head.normalized, decomposer))
        def m(k): return float(np.mean([r[k] for r in rows]))
        out.append(LayerSummary(
            layer=L, n_heads=loader.n_heads,
            mean_total_energy=m("total_energy"),
            mean_shannon_entropy=m("shannon_entropy"),
            mean_spectral_entropy=m("spectral_entropy"),
            mean_gini=m("gini_sparsity"),
            mean_energy_ratio_LH=m("energy_ratio_low_high"),
            mean_rec_err_30pct=m("reconstruction_error_30pct"),
            std_dominant_level=float(np.std([r["dominant_level"]
                                              for r in rows])),
            mean_low_freq_energy=m("low_freq_energy"),
            mean_high_freq_energy=m("high_freq_energy"),
        ))
    return out


# --------------------------------------------------------------------------- #
# Persist + plot
# --------------------------------------------------------------------------- #

def save_layer_progression(
    summaries: List[LayerSummary],
    save_dir: str,
    suffix: str = "",
):
    os.makedirs(save_dir, exist_ok=True)
    # CSV
    import csv
    path = os.path.join(save_dir, "layer_progression.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "n_heads", "E_total", "E_low", "E_high", "E_ratio_LH",
            "shannon", "spec_entropy", "gini", "rec_err_30pct",
            "std_dom_level",
        ])
        for s in summaries:
            writer.writerow([
                s.layer, s.n_heads,
                f"{s.mean_total_energy:.5f}", f"{s.mean_low_freq_energy:.5f}",
                f"{s.mean_high_freq_energy:.5f}",
                f"{s.mean_energy_ratio_LH:.4f}",
                f"{s.mean_shannon_entropy:.4f}",
                f"{s.mean_spectral_entropy:.4f}",
                f"{s.mean_gini:.4f}",
                f"{s.mean_rec_err_30pct:.5f}",
                f"{s.std_dominant_level:.4f}",
            ])
    # PNG + PDF multipanel plot
    xs = [s.layer for s in summaries]
    plots = {
        "Shannon entropy":              [s.mean_shannon_entropy for s in summaries],
        "Spectral entropy":             [s.mean_spectral_entropy for s in summaries],
        "Total energy":                  [s.mean_total_energy for s in summaries],
        "Energy ratio L/H":              [s.mean_energy_ratio_LH for s in summaries],
        "Gini sparsity":                 [s.mean_gini for s in summaries],
        "Reconstruction error (30%)":   [s.mean_rec_err_30pct for s in summaries],
        "Low freq energy":               [s.mean_low_freq_energy for s in summaries],
        "High freq energy":              [s.mean_high_freq_energy for s in summaries],
        "Dominant level std":            [s.std_dominant_level for s in summaries],
    }
    save_path_png = os.path.join(save_dir, f"layer_evolution{suffix}.png")
    save_path_pdf = os.path.join(save_dir, f"layer_evolution{suffix}.pdf")
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, (label, ys) in zip(axes.ravel(), plots.items()):
        ax.plot(xs, ys, marker="o", color="tab:blue")
        ax.set_title(label)
        ax.set_xlabel("Layer")
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Layer-by-layer wavelet metrics ({summaries[0].n_heads} heads/layer)"
        if summaries else "Layer-by-layer wavelet metrics"
    )
    fig.tight_layout()
    fig.savefig(save_path_png, dpi=140, bbox_inches="tight")
    fig.savefig(save_path_pdf, bbox_inches="tight")
    plt.close(fig)


def compute_and_save_layer_progression(
    loader: AttentionLoader,
    decomposer_factory,
    save_dir: str,
) -> List[LayerSummary]:
    summaries = compute_layer_summaries(loader, decomposer_factory())
    save_layer_progression(summaries, save_dir)
    return summaries
