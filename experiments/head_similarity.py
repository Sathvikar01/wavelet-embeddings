"""Head-pair similarity across (layer, head) pairs for one model.

Per pair of heads (A, B):

  * cosine_similarity on raw attention vectorisations
  * wavelet coefficient cosine (concatenated approx + per-level subbands,
    truncated to common length)
  * energy_profile_similarity : cosine of the per-level energy share vectors
  * earth_mover_distance      : exact EMD on histograms of attention values
  * pearson_correlation       : Pearson r of raw flattenened matrices
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from attention import (AttentionLoader, AttentionWaveletDecomposer,
                        head_wavelet_flat_coeffs, HeadDecomposition)
from attention.analyzer import _flatten_decomp_magnitudes, _spectral_entropy


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def _cos_to_same_len(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return _cos(a[:n], b[:n])


def _pearson(A: np.ndarray, B: np.ndarray) -> float:
    n = min(A.size, B.size)
    a = A.ravel()[:n]
    b = B.ravel()[:n]
    if n < 2:
        return 0.0
    sa = a.std(); sb = b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _energy_share_vector(decomp: HeadDecomposition) -> np.ndarray:
    """Per-level energy shares (smoothed) including approx as level 0."""
    items = [(0, decomp.energy_approx)] + list(decomp.energy_per_level.items())
    items.sort()
    energies = np.array([e for _, e in items], dtype=np.float64)
    tot = energies.sum() or 1.0
    return energies / tot


def _emd_uniform_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Earth Mover Distance between the *histograms* of two attention-matrix
    values (not the matrices themselves - that has dimension T*T for OT).

    Uses the W1 distance (1-D EMD) between the cumulative distributions as
    a fast approximation.
    """
    a = h1.ravel()
    b = h2.ravel()
    # Bin range from combined values
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi - lo < 1e-12:
        return 0.0
    bins = np.linspace(lo, hi, 64)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    # Normalise to sum-1 distribution
    pa = np.maximum(pa / (pa.sum() + 1e-12), 0.0)
    pb = np.maximum(pb / (pb.sum() + 1e-12), 0.0)
    cdf_a = np.cumsum(pa)
    cdf_b = np.cumsum(pb)
    dx = (bins[1] - bins[0])
    return float(np.sum(np.abs(cdf_a - cdf_b)) * dx)


# --------------------------------------------------------------------------- #
# Similarity runner
# --------------------------------------------------------------------------- #

@dataclass
class HeadPairSimilarity:
    layer_i: int
    head_i: int
    layer_j: int
    head_j: int
    cosine: float
    wavelet_similarity: float
    energy_profile: float
    emd: float
    pearson: float


def compute_pair_matrix(
    loader: AttentionLoader,
    decomposer_factory,
    save_dir: Optional[str] = None,
    flat_max_len: int = 256,
) -> Dict[str, np.ndarray]:
    """Compute pairwise similarity matrices for every (layer, head) pair."""
    dec = decomposer_factory()
    n_total = loader.n_layers * loader.n_heads
    # Pre-decompose everything
    decomps: List[HeadDecomposition] = []
    raws: List[np.ndarray] = []
    for L in range(loader.n_layers):
        for H in range(loader.n_heads):
            head = loader.load_head(L, H)
            A = head.normalized
            d = dec.decompose(A)
            decomps.append(d)
            raws.append(A)
    # Allocate matrices
    cos = np.zeros((n_total, n_total), dtype=np.float32)
    wsim = np.zeros((n_total, n_total), dtype=np.float32)
    esha = np.zeros((n_total, n_total), dtype=np.float32)
    emd = np.zeros((n_total, n_total), dtype=np.float32)
    pear = np.zeros((n_total, n_total), dtype=np.float32)
    # Flattened coeffs for wavelet sim
    flat_coeffs = [head_wavelet_flat_coeffs(d, max_len=flat_max_len)
                   for d in decomps]
    energy_shares = [_energy_share_vector(d) for d in decomps]
    flat_raws = [r.ravel() for r in raws]
    for i in range(n_total):
        cos[i, i] = 1.0
        wsim[i, i] = 1.0
        esha[i, i] = 1.0
        pear[i, i] = 1.0
        emd[i, i] = 0.0
        for j in range(i + 1, n_total):
            cs = _cos(flat_raws[i], flat_raws[j])
            ws = _cos_to_same_len(flat_coeffs[i], flat_coeffs[j])
            es = _cos_to_same_len(energy_shares[i], energy_shares[j])
            rs = _pearson(raws[i], raws[j])
            edd = _emd_uniform_distance(raws[i], raws[j])
            cos[i, j] = cos[j, i] = cs
            wsim[i, j] = wsim[j, i] = ws
            esha[i, j] = esha[j, i] = es
            pear[i, j] = pear[j, i] = rs
            emd[i, j] = emd[j, i] = edd
    out = {"cosine": cos, "wavelet": wsim, "energy_profile": esha,
            "pearson": pear, "emd": emd}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for k, mat in out.items():
            np.save(os.path.join(save_dir, f"similarity_{k}.npy"), mat)
        # Also save a CSV of pair list
        import csv
        with open(os.path.join(save_dir, "pair_similarity.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w_ = csv.writer(f)
            w_.writerow(["li", "hi", "lj", "hj",
                          "cosine", "wavelet", "energy_profile",
                          "pearson", "emd"])
            for i in range(n_total):
                for j in range(n_total):
                    Li, Hi = divmod(i, loader.n_heads)
                    Lj, Hj = divmod(j, loader.n_heads)
                    w_.writerow([
                        Li, Hi, Lj, Hj,
                        f"{cos[i,j]:.4f}", f"{wsim[i,j]:.4f}",
                        f"{esha[i,j]:.4f}", f"{pear[i,j]:.4f}",
                        f"{emd[i,j]:.4f}",
                    ])
    return out
