"""Entropy and distribution-shape statistics over wavelet coefficients.

We compute:

  * Shannon entropy over the *energy-normalised* coefficient magnitudes
    H = -Σ p_i log2 p_i
  * joint entropy of approx + detail bands
  * entropy per band (where applicable)
  * high-order statistics: skewness, kurtosis of coefficient distributions
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from wavelets.base import WaveletDecomposition


def _shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def coefficient_entropy(d: WaveletDecomposition) -> float:
    """Shannon entropy over absolute coefficient magnitudes (normalised).

    combines approx + all detail coefficients into one distribution.
    """
    flat = np.concatenate(
        [np.abs(d.approx).ravel()] + [np.abs(c).ravel() for c in d.details]
    )
    if flat.size == 0:
        return 0.0
    s = flat.sum()
    if s == 0:
        return 0.0
    probs = flat / s
    return _shannon_entropy(probs)


def band_entropies(d: WaveletDecomposition) -> Dict[str, float]:
    """Per-band entropies: approx + per-level details."""
    out: Dict[str, float] = {"approx": _band_entropy(d.approx)}
    for lvl, c in enumerate(d.details, start=1):
        out[f"detail_L{lvl}"] = _band_entropy(c)
    return out


def _band_entropy(c: np.ndarray) -> float:
    a = np.abs(c.ravel())
    if a.size == 0:
        return 0.0
    s = a.sum()
    if s == 0:
        return 0.0
    return _shannon_entropy(a / s)


def entropy_summary(d: WaveletDecomposition) -> Dict[str, float]:
    """Compact entropy dict (global + summary)."""
    return {
        "global_entropy": coefficient_entropy(d),
        "approx_entropy": _band_entropy(d.approx),
        "mean_band_entropy": float(
            np.mean([_band_entropy(c) for c in d.details])
            if d.details else 0.0
        ),
        "max_band_entropy": float(
            max((_band_entropy(c) for c in d.details), default=0.0)
        ),
    }


# --------------------------------------------------------------------------- #
# Higher-order statistics
# --------------------------------------------------------------------------- #

def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 3:
        return 0.0
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def kurtosis(x: np.ndarray, fisher: bool = True) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 4:
        return 0.0
    mu = x.mean()
    sigma = x.std()
    if sigma == 0:
        return 0.0
    k = float(np.mean(((x - mu) / sigma) ** 4))
    return k - 3.0 if fisher else k


def coefficient_statistics(d: WaveletDecomposition) -> Dict[str, float]:
    """Skewness & kurtosis of the combined coefficient distribution."""
    flat = np.concatenate(
        [d.approx.ravel()] + [c.ravel() for c in d.details]
    )
    return {
        "mean": float(flat.mean()) if flat.size else 0.0,
        "variance": float(flat.var()) if flat.size else 0.0,
        "skewness": skewness(flat),
        "kurtosis": kurtosis(flat),
    }


def batch_entropy_summary(decomps: List[WaveletDecomposition]) -> Dict[str, np.ndarray]:
    """Aggregate entropy stats for many decompositions."""
    if not decomps:
        return {}
    keys = list(entropy_summary(decomps[0]).keys())
    keys += ["skewness", "kurtosis", "mean", "variance"]
    out = {k: np.zeros(len(decomps)) for k in keys}
    for i, d in enumerate(decomps):
        s = entropy_summary(d)
        st = coefficient_statistics(d)
        for k, v in s.items():
            out[k][i] = v
        for k, v in st.items():
            out[k][i] = v
    return out
