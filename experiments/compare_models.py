"""Compare wavelet-analysis results across BERT-base, DistilBERT, GPT-2.

For every model:
  * decompose the entire (or sampled) embedding matrix with the user's chosen wavelet
  * compute mean energy, variance, entropy, skewness, kurtosis, compression ratio, SNR
  * produce a comparison table + plots

Public API:
  * ``run_model_comparison(...)``
"""

from __future__ import annotations

import os
import zlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from embeddings.loader import EmbeddingLoader
from wavelets.base import WaveletDecomposer
from analysis.energy import batch_energy_summary, energy_summary, total_energy
from analysis.entropy import batch_entropy_summary, coefficient_entropy
from analysis.sparsity import batch_sparsity_summary, compression_ratio, gini_coefficient
from analysis.compression import compress_batch, DEFAULT_COMPRESSION_RATIOS
from visualization.spectrum import reconstruction_error_curve


# --------------------------------------------------------------------------- #
# Data class: per-model statistics
# --------------------------------------------------------------------------- #

@dataclass
class ModelStats:
    model_name: str
    wavelet: str
    vocab_sampled: int
    embed_dim: int
    mean_energy: float
    var_energy: float
    mean_entropy: float
    var_entropy: float
    mean_skewness: float
    mean_kurtosis: float
    mean_gini: float
    mean_compression_ratio: float
    snr_db_per_compression_level: List[float]
    cosine_per_compression_level: List[float]


# --------------------------------------------------------------------------- #
# Comparison runner
# --------------------------------------------------------------------------- #

def run_model_comparison(
    loaders: Dict[str, EmbeddingLoader],
    decomposer_factory,
    sample_size: int = 2000,
    compression_ratios=DEFAULT_COMPRESSION_RATIOS,
    seed: int = 0,
    save_dir: str | None = None,
    show: bool = False,
) -> Dict[str, ModelStats]:
    """Run identical analysis on multiple model loaders.

    ``decomposer_factory`` should be a callable returning a fresh
    :class:`WaveletDecomposer` (so we don't reuse internal state).
    """
    out: Dict[str, ModelStats] = {}
    for i, (name, loader) in enumerate(loaders.items()):
        n = min(sample_size, loader.vocab_size)
        # crc32 gives a run-stable seed offset (Python's hash() is
        # randomised per process by PYTHONHASHSEED).
        rng = np.random.default_rng(seed + zlib.crc32(name.encode("utf-8")))
        idx = rng.choice(loader.vocab_size, size=n, replace=False)
        X = loader.embeddings[idx]
        decomposer = decomposer_factory()
        # Batch decompose
        decomps = decomposer.batch_decompose(X)
        # Stats
        e_summary = batch_energy_summary(decomps)
        ent_summary = batch_entropy_summary(decomps)
        sp_summary = batch_sparsity_summary(decomps)
        # Compression - per ratio, aggregate SNR + cosine
        # Subsample to keep cost bounded
        cs_idx = rng.choice(n, size=min(200, n), replace=False)
        cs_X = X[cs_idx]
        recs, stats = compress_batch(cs_X, decomposer, ratios=compression_ratios)
        snrs = [stats[r].mean_snr_db for r in compression_ratios]
        cosines = [stats[r].mean_cosine for r in compression_ratios]

        ms = ModelStats(
            model_name=name,
            wavelet=decomposer.wavelet_name,
            vocab_sampled=n,
            embed_dim=loader.embed_dim,
            mean_energy=float(e_summary["total"].mean()),
            var_energy=float(e_summary["total"].var()),
            mean_entropy=float(ent_summary["global_entropy"].mean()),
            var_entropy=float(ent_summary["global_entropy"].var()),
            mean_skewness=float(ent_summary["skewness"].mean()),
            mean_kurtosis=float(ent_summary["kurtosis"].mean()),
            mean_gini=float(sp_summary["gini"].mean()),
            mean_compression_ratio=float(sp_summary["compression_ratio"].mean()),
            snr_db_per_compression_level=snrs,
            cosine_per_compression_level=cosines,
        )
        out[name] = ms

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        # Save CSV summary
        import csv
        with open(os.path.join(save_dir, "model_comparison.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["model", "wavelet", "vocab", "dim", "mean_energy",
                      "var_energy", "mean_entropy", "mean_skewness",
                      "mean_kurtosis", "mean_gini", "mean_compression_ratio",
                      "snr_db_levels", "cosine_levels"]
            writer.writerow(header)
            for name, s in out.items():
                writer.writerow([
                    s.model_name, s.wavelet, s.vocab_sampled, s.embed_dim,
                    f"{s.mean_energy:.4f}", f"{s.var_energy:.4f}",
                    f"{s.mean_entropy:.4f}", f"{s.mean_skewness:.4f}",
                    f"{s.mean_kurtosis:.4f}", f"{s.mean_gini:.4f}",
                    f"{s.mean_compression_ratio:.4f}",
                    ";".join(f"{v:.3f}" for v in s.snr_db_per_compression_level),
                    ";".join(f"{v:.3f}" for v in s.cosine_per_compression_level),
                ])
        # Plot reconstruction cosine curves overlaid
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for name, s in out.items():
            ax.plot(list(compression_ratios), s.cosine_per_compression_level,
                    marker="o", label=name)
        ax.set_xlabel("Ratio of small detail coefficients zeroed")
        ax.set_ylabel("Mean cosine(orig, reconstruction)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(f"Reconstruction cosine across models ({list(loaders)[0] and 'wavelet'})")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "model_reconstruction_comparison.png"),
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
    return out
