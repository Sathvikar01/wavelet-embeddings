"""Sparsity & compression-ratio metrics on wavelet coefficients.

Sparsity indicates how concentrated the embedding signal's energy is in
just a few large coefficients.  Several flavours are provided:

  * ``l1_sparsity``        : norm(c1)/norm(c0) - related to Gini of magnitudes
  * ``count_sparsity``   : fraction of coefficients close to zero
  * ``gini``            : Gini coefficient of |coefficient| distribution
  * ``neyman_pearson`` : number of coefficients accounting for (1-ε) of energy

Compression ratio is the complement of sparsity:

  CR = (#non-zero after thresholding) / (#total)  and we convert to Kb saved.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from wavelets.base import WaveletDecomposition


def _flat_magnitudes(d: WaveletDecomposition) -> np.ndarray:
    parts = [np.abs(d.approx).ravel()] + [np.abs(c).ravel() for c in d.details]
    return np.concatenate(parts)


def l1_l2_sparsity(d: WaveletDecomposition) -> float:
    """(||c||_1) / (||c||_2) - higher = sparser."""
    flat_mag = _flat_magnitudes(d)
    if flat_mag.size == 0:
        return 0.0
    l1 = flat_mag.sum()
    l2 = np.sqrt((flat_mag ** 2).sum())
    if l2 == 0:
        return 0.0
    return float(l1 / l2)


def count_sparsity(d: WaveletDecomposition, tol: float = 1e-4) -> float:
    """Fraction of near-zero coefficients (|c| ≤ tol * max(|c|))."""
    mags = _flat_magnitudes(d)
    if mags.size == 0:
        return 0.0
    thr = tol * mags.max() if mags.max() > 0 else 0.0
    return float(np.mean(mags <= thr))


def gini_coefficient(d: WaveletDecomposition) -> float:
    """Gini over coefficient magnitudes - 0 means perfectly uniform, 1 maximally sparse."""
    mags = _flat_magnitudes(d)
    if mags.size < 2 or mags.sum() == 0:
        return 0.0
    s = np.sort(mags)
    n = s.size
    cumsum = np.cumsum(s)
    B = cumsum[-1]
    if B == 0:
        return 0.0
    G = (2 * np.sum(np.arange(1, n + 1) * s) / (n * B)) - (n + 1) / n
    return float(G)


def energy_concentration_count(d: WaveletDecomposition, eps: float = 0.01) -> int:
    """#coefficients needed to retain (1-ε) of total energy - smaller = sparser."""
    mags = _flat_magnitudes(d)
    if mags.size == 0:
        return 0
    e = mags ** 2
    order = np.argsort(-e)
    cum = np.cumsum(e[order])
    if cum.size == 0 or cum[-1] == 0:
        return 0
    target = (1 - eps) * cum[-1]
    idx = int(np.searchsorted(cum, target))
    return idx + 1


def compression_ratio(d: WaveletDecomposition, eps: float = 0.01) -> float:
    """compression ratio = total / #coefficients retaining (1-ε) energy.

    Higher value = more compressible.
    """
    n_keep = energy_concentration_count(d, eps=eps)
    total = sum(len(p) for p in [d.approx] + d.details)
    if n_keep == 0 or total == 0:
        return 0.0
    return float(total / n_keep)


def sparsity_summary(d: WaveletDecomposition) -> Dict[str, float]:
    """Compact sparsity summary."""
    return {
        "l1_l2_sparsity": l1_l2_sparsity(d),
        "count_sparsity": count_sparsity(d),
        "gini": gini_coefficient(d),
        "compression_ratio": compression_ratio(d),
        "energy_concentration_count": float(energy_concentration_count(d)),
    }


def batch_sparsity_summary(decomps: List[WaveletDecomposition]) -> Dict[str, np.ndarray]:
    if not decomps:
        return {}
    keys = list(sparsity_summary(decomps[0]).keys())
    out = {k: np.zeros(len(decomps)) for k in keys}
    for i, d in enumerate(decomps):
        s = sparsity_summary(d)
        for k, v in s.items():
            out[k][i] = v
    return out
