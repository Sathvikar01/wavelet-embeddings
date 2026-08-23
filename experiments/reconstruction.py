"""Reconstruction experiment.

For one or more probe tokens:
  * Decompose with chosen wavelet
  * For each compression ratio r in [0.1, 0.2, ..., 0.5]:
      - Reconstruct the embedding
      * Measure cosine(orig, rec), L2 drift, SNR
      * Compute top-k neighbor preservation in the full embedding matrix
      * Compute t-SNE / PCA before-after on a sample
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from embeddings.loader import EmbeddingLoader
from wavelets.base import WaveletDecomposer
from analysis.compression import (
    compress_embedding,
    compress_batch,
    neighbor_preservation,
    DEFAULT_COMPRESSION_RATIOS,
)


@dataclass
class ReconstructionReport:
    token: str
    wavelet: str
    ratios: List[float]
    cosines: List[float]
    snrs: List[float]
    drifts: List[float]
    neighbor_preservations: List[float]   # mean Jaccard overlap with full vocab
    neighbor_drifts: List[float]

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Single-token reconstruction
# --------------------------------------------------------------------------- #

def reconstruct_token(
    loader: EmbeddingLoader,
    token: str,
    decomposer: WaveletDecomposer,
    ratios: Sequence[float] = DEFAULT_COMPRESSION_RATIOS,
    k_neighbors: int = 10,
    neighbor_sample: int = 500,
    seed: int = 0,
) -> Optional[ReconstructionReport]:
    """Run the full reconstruction experiment for one token.

    Neighbor preservation is measured against a random sub-sample of the vocab
    to keep computation tractable.
    """
    if token not in loader.tokens:
        return None
    idx = loader.index_of(token)
    rng = np.random.default_rng(seed)
    # Sample neighbour candidates
    sample_idx = rng.choice(
        loader.vocab_size, size=min(neighbor_sample + 1, loader.vocab_size),
        replace=False,
    )
    if idx not in sample_idx:
        sample_idx[0] = idx
    sample_X = loader.embeddings[sample_idx].copy()

    decomposer_copy = decomposer
    recs = compress_embedding(sample_X[sample_idx == idx][0] if False else
                              loader.embeddings[idx], decomposer_copy, ratios=ratios)
    # Build reconstructed matrix for neighbour analysis at each ratio
    orig_vec = loader.embeddings[idx]
    cosines, snrs, drifts = [], [], []
    neighbor_pres_list, neighbor_drift_list = [], []
    sample_matrix = sample_X.copy()
    sample_pos = int(np.where(sample_idx == idx)[0][0])
    for cr in recs:
        rec = cr.reconstructed
        cosines.append(cr.cosine)
        snrs.append(cr.snr_db)
        drifts.append(cr.drift)
        # Substitute reconstructed vector into the sample matrix
        mod_matrix = sample_matrix.copy()
        mod_matrix[sample_pos] = rec
        pres, drift = neighbor_preservation(
            sample_matrix, mod_matrix, k=k_neighbors, sample=np.array([sample_pos]),
        )
        neighbor_pres_list.append(pres)
        neighbor_drift_list.append(drift)
    return ReconstructionReport(
        token=token,
        wavelet=decomposer.wavelet_name,
        ratios=list(ratios),
        cosines=cosines,
        snrs=snrs,
        drifts=drifts,
        neighbor_preservations=neighbor_pres_list,
        neighbor_drifts=neighbor_drift_list,
    )


# --------------------------------------------------------------------------- #
# Runner over many tokens
# --------------------------------------------------------------------------- #

def run_reconstruction_experiment(
    loader: EmbeddingLoader,
    decomposer_factory,
    tokens: Optional[List[str]] = None,
    ratios: Sequence[float] = DEFAULT_COMPRESSION_RATIOS,
    save_dir: str | None = None,
    show: bool = False,
    k_neighbors: int = 10,
    neighbor_sample: int = 500,
) -> List[ReconstructionReport]:
    if tokens is None:
        from embeddings.loader import NEIGHBOR_TOKENS, DEFAULT_PROBE_TOKENS
        tokens = list(set(NEIGHBOR_TOKENS + DEFAULT_PROBE_TOKENS))[:30]
    reports: List[ReconstructionReport] = []
    for t in tokens:
        r = reconstruct_token(loader, t, decomposer_factory(),
                              ratios=ratios, k_neighbors=k_neighbors,
                              neighbor_sample=neighbor_sample)
        if r is not None:
            reports.append(r)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import csv
        with open(os.path.join(save_dir, "reconstruction_results.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token", "wavelet", "ratio", "cosine", "snr_db",
                             "drift", "neighbor_preservation", "neighbor_drift"])
            for r in reports:
                for k in range(len(r.ratios)):
                    writer.writerow([
                        r.token, r.wavelet, f"{r.ratios[k]:.2f}",
                        f"{r.cosines[k]:.4f}", f"{r.snrs[k]:.3f}",
                        f"{r.drifts[k]:.4f}", f"{r.neighbor_preservations[k]:.3f}",
                        f"{r.neighbor_drifts[k]:.4f}",
                    ])
        # Plot per-token cosine curves
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in reports:
            ax.plot(r.ratios, r.cosines, marker="o", label=r.token, alpha=0.7)
        ax.set_xlabel("Ratio of small detail coefficients zeroed")
        ax.set_ylabel("Cosine(original, reconstructed)")
        ax.set_title(f"Reconstruction cosine per token "
                     f"({reports[0].wavelet if reports else 'wavelet'})")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "reconstruction_cosine_per_token.png"),
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
        # Plot neighbour preservation curve
        fig, ax = plt.subplots(figsize=(10, 6))
        for r in reports:
            ax.plot(r.ratios, r.neighbor_preservations, marker="s",
                    label=r.token, alpha=0.7)
        ax.set_xlabel("Ratio of small detail coefficients zeroed")
        ax.set_ylabel("Top-k neighbour preservation (Jaccard)")
        ax.set_title("Neighbour preservation under compression")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "reconstruction_neighborhood_preservation.png"),
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
        if show:
            plt.show()
    return reports
