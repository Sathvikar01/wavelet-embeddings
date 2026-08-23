"""Compute wavelet energies & derived statistics.

For every level ℓ of a multilevel DWT we get a detail coefficient array
cD_ℓ.  The Energy E_ℓ := Σ cD_ℓ², and the corresponding approximation
energy EA := Σ cA².

This module returns scalars and per-level vectors:

  * ``total_energy``           - EA + Σ_ℓ E_ℓ
  * ``approx_energy``          - EA
  * ``detail_energy_per_level``- {ℓ: E_ℓ}
  * ``low_freq_energy``        - EA + energies of the coarsest ℓ ≥ 2
  * ``high_freq_energy``       - energies of ℓ == 1 (finest details)
  * ``energy_ratio``           - low / high  (the "Information Score"
                                          baseline candidate)
  * ``energy_distribution``    - normalised energy share per level
  * ``dominant_level``          - level with max E_ℓ
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from wavelets.base import WaveletDecomposition


def total_energy(d: WaveletDecomposition) -> float:
    return d.energy_approx + sum(d.energy_per_level.values())


def approx_energy(d: WaveletDecomposition) -> float:
    return d.energy_approx


def detail_energy_per_level(d: WaveletDecomposition) -> Dict[int, float]:
    return dict(d.energy_per_level)


def low_freq_energy(d: WaveletDecomposition, low_levels: int | None = None) -> float:
    """Approx + coarsest (highest ℓ) detail energies.

    ``low_levels`` controls how many of the coarsest detail levels we bucket
    as "low-frequency". If ``None`` it buckets half the levels (rounded up).
    """
    n = d.level
    if n <= 1:
        return d.energy_approx
    if low_levels is None:
        low_levels = max(1, n // 2)
    e = d.energy_approx
    # details index 0 = level 1 (finest) so coarsest = highest index;
    # bucket the *coarsest* ``low_levels`` detail levels as low-frequency.
    for i in range(max(1, n - low_levels + 1), n + 1):
        e += d.energy_per_level.get(i, 0.0)
    return e


def high_freq_energy(d: WaveletDecomposition, high_levels: int | None = None) -> float:
    """Fine detail energies.

    ``high_levels`` = number of finest detail levels counted as "high".
    Default: half the levels (rounded down) - the coarse levels go to low.
    """
    n = d.level
    if n <= 1:
        return d.energy_per_level.get(1, 0.0)
    if high_levels is None:
        high_levels = max(1, n // 2)
    e = 0.0
    for i in range(1, min(high_levels, n) + 1):
        e += d.energy_per_level.get(i, 0.0)
    return e


def energy_ratio_low_high(d: WaveletDecomposition) -> float:
    """Information Score candidate: E_low / E_high."""
    lo = low_freq_energy(d)
    hi = high_freq_energy(d)
    if hi == 0.0:
        return float("inf") if lo > 0 else 0.0
    return lo / hi


def energy_distribution(d: WaveletDecomposition) -> Dict[int, float]:
    """Normalised energy share per level (incl. approx at level 0)."""
    tot = total_energy(d)
    if tot == 0:
        return {}
    out: Dict[int, float] = {0: d.energy_approx / tot}
    for lvl, e in d.energy_per_level.items():
        out[lvl] = e / tot
    return out


def dominant_level(d: WaveletDecomposition) -> int:
    """Level (0 = approx) with greatest energy."""
    items = [(0, d.energy_approx)] + list(d.energy_per_level.items())
    return max(items, key=lambda kv: kv[1])[0]


def energy_summary(d: WaveletDecomposition) -> Dict[str, float]:
    """Compact energy summary dict."""
    return {
        "total": total_energy(d),
        "approx": approx_energy(d),
        "low": low_freq_energy(d),
        "high": high_freq_energy(d),
        "ratio_low_high": energy_ratio_low_high(d),
        "dominant_level": float(dominant_level(d)),
    }


def batch_energy_summary(decomps: List[WaveletDecomposition]) -> Dict[str, np.ndarray]:
    """Aggregate energy stats for many decompositions.

    Keys are scalar-arm names from :func:`energy_summary`; values are arrays
    of length len(decomps).
    """
    if not decomps:
        return {}
    keys = list(energy_summary(decomps[0]).keys())
    arr = {k: np.zeros(len(decomps)) for k in keys}
    for i, d in enumerate(decomps):
        s = energy_summary(d)
        for k, v in s.items():
            arr[k][i] = v
    return arr
