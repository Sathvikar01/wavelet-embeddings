"""Comparative analysis across BERT-base / DistilBERT / GPT-2 Phase-3.

For each model, compute aggregate statistics from per-head metrics
(``attention/analyzer.compute_head_metrics``) and write a CSV comparing
average-entropy, average-energy, redundancy and compression robustness.

Per-model aggregates:

  * mean total_energy           (mean across heads)
  * mean shannon_entropy
  * mean spectral_entropy
  * mean gini                   (sparsity)
  * mean energy_ratio_low_high
  * mean reconstruction_error_30pct   (across heads)
  * mean compression_ratio_99
  * redundancy_density          (computed by experiments/redundancy.py)
  * frequency_diversity        (std of dominant_level across heads)

We also save a side-by-side CSV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from attention import (
    AttentionLoader, AttentionWaveletDecomposer, compute_head_metrics,
)
from attention.loader import find_available_models


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class ModelAggregate:
    model: str
    n_layers: int
    n_heads: int
    mean_total_energy: float
    mean_shannon_entropy: float
    mean_spectral_entropy: float
    mean_gini: float
    mean_energy_ratio_low_high: float
    mean_reconstruction_error_30pct: float
    mean_compression_ratio_99: float
    std_dominant_level: float
    redundancy_density: float
    frequency_diversity: float


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def aggregate_model(loader: AttentionLoader,
                    decomposer: AttentionWaveletDecomposer,
                    redundancy_volume: int = 0) -> Tuple[ModelAggregate,
                                                          List[Dict[str, float]]]:
    """Compute aggregate metrics for a single model snapshot."""
    per_head_rows: List[Dict[str, float]] = []
    for L in range(loader.n_layers):
        for H in range(loader.n_heads):
            head = loader.load_head(L, H)
            m = compute_head_metrics(head.normalized, decomposer)
            row = {"layer": L, "head": H, **m}
            per_head_rows.append(row)
    if not per_head_rows:
        return ModelAggregate(
            model=os.path.basename(loader.model_dir), n_layers=0, n_heads=0,
            mean_total_energy=0, mean_shannon_entropy=0,
            mean_spectral_entropy=0, mean_gini=0,
            mean_energy_ratio_low_high=0,
            mean_reconstruction_error_30pct=0,
            mean_compression_ratio_99=0, std_dominant_level=0,
            redundancy_density=0, frequency_diversity=0
        ), []
    keys = ("total_energy", "shannon_entropy", "spectral_entropy",
            "gini_sparsity", "energy_ratio_low_high",
            "reconstruction_error_30pct", "compression_ratio_99", "dominant_level")
    means = {k: float(np.mean([r[k] for r in per_head_rows])) for k in keys}
    std_dom = float(np.std([r["dominant_level"] for r in per_head_rows]))
    return ModelAggregate(
        model=os.path.basename(loader.model_dir),
        n_layers=loader.n_layers, n_heads=loader.n_heads,
        mean_total_energy=means["total_energy"],
        mean_shannon_entropy=means["shannon_entropy"],
        mean_spectral_entropy=means["spectral_entropy"],
        mean_gini=means["gini_sparsity"],
        mean_energy_ratio_low_high=means["energy_ratio_low_high"],
        mean_reconstruction_error_30pct=means["reconstruction_error_30pct"],
        mean_compression_ratio_99=means["compression_ratio_99"],
        std_dominant_level=std_dom,
        redundancy_density=float(redundancy_volume),
        frequency_diversity=std_dom,
    ), per_head_rows


def run_model_comparison(
    attention_root: str,
    decomposer_factory,
    save_dir: str,
) -> Dict[str, ModelAggregate]:
    """Iterate all model dirs under ``attention_root`` and produce reports."""
    os.makedirs(save_dir, exist_ok=True)
    model_dirs = find_available_models(attention_root)
    out: Dict[str, ModelAggregate] = {}
    for m in model_dirs:
        loader = AttentionLoader(os.path.join(attention_root, m))
        agg, _ = aggregate_model(loader, decomposer_factory())
        out[m] = agg
    # CSV summary
    import csv
    with open(os.path.join(save_dir, "model_comparison.csv"), "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "n_layers", "n_heads", "mean_E", "mean_entropy",
            "mean_spec_entropy", "mean_gini", "mean_E_ratio_LH",
            "mean_rec_error_30pct", "mean_cr_99",
            "std_dominant_level", "freq_diversity", "redundancy_density",
        ])
        for k, v in out.items():
            writer.writerow([
                v.model, v.n_layers, v.n_heads,
                f"{v.mean_total_energy:.4f}", f"{v.mean_shannon_entropy:.4f}",
                f"{v.mean_spectral_entropy:.4f}", f"{v.mean_gini:.4f}",
                f"{v.mean_energy_ratio_low_high:.4f}",
                f"{v.mean_reconstruction_error_30pct:.4f}",
                f"{v.mean_compression_ratio_99:.4f}",
                f"{v.std_dominant_level:.4f}", f"{v.frequency_diversity:.4f}",
                f"{v.redundancy_density:.4f}",
            ])
    return out
