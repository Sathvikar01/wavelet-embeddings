"""Wavelet decomposition utilities for transformer embeddings.

This module is the heart of the Wavelet-Analysis-of-Embeddings pipeline.

Key concepts
------------
Every token embedding is treated as a 1-D discrete signal of length ``D``
(e.g. 768 / 768 / 768 for the three target models). Because 768 is not a
power of two we pad symmetrically to the next power of two (1024) before
forward DWT and crop back after reconstruction.  Padding mode is the
common 'periodization' (PyWavelets ``mode='periodization'``) whenever it
works but falls back to symmetric padding with edge zeros.

The supported wavelet families are:

  * Haar           ("haar")
  * Daubechies-4   ("db4")
  * Symlet-4       ("sym4")
  * Coiflet-2      ("coif2")

All public callers go through :class:`WaveletDecomposer` which:

  * takes a single embedding vector (length D)
  * performs full multilevel discrete wavelet transform (DWT)
  * returns a :class:`WaveletDecomposition` bundle containing:
      - ``approx``      : Approximation coefficients cA (last level)
      - ``details``     : List of detail coefficient arrays cD_n..cD_1
      - ``coeffs_flat`` : A unified flat representation  [cA, cD_last, ..., cD_1]
      - ``level``       : max usable level
      - ``wavelet_name``: name string
      - ``energy``     : per-level energy

The energy is computed as ``sum(c**2)`` for each coefficient array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import pywt  # PyWavelets
    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _HAS_PYWT = False


# --------------------------------------------------------------------------- #
# Supported wavelets
# --------------------------------------------------------------------------- #

SUPPORTED_WAVELETS: List[str] = ["haar", "db4", "sym4", "coif2"]

# Map our friendly names to PyWavelets names (they coincide here, but this
# makes it explicit and easy to remap later).
_PYWT_NAME_MAP: Dict[str, str] = {
    "haar": "haar",
    "db4": "db4",
    "sym4": "sym4",
    "coif2": "coif2",
}


def _require_pywt():
    if not _HAS_PYWT:
        raise ImportError(
            "PyWavelets (pywt) is required. Install with: pip install PyWavelets"
        )


# --------------------------------------------------------------------------- #
# Decomposition container
# --------------------------------------------------------------------------- #

@dataclass
class WaveletDecomposition:
    wavelet_name: str
    level: int
    approx: np.ndarray            # آخر مستوى تقریب coeffs (1D array)
    details: List[np.ndarray]     # detail coeffs، index 0 = finest (level 1)، آخر = coarsest
    padded_length: int            # length signal was padded to before DWT
    original_length: int          # D
    energy_per_level: Dict[int, float] = field(default_factory=dict)
    energy_approx: float = 0.0


# --------------------------------------------------------------------------- #
# Core decomposer
# --------------------------------------------------------------------------- #

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pad_signal(x: np.ndarray, target_len: int) -> np.ndarray:
    """Symmetric pad to a target length using edge reflection."""
    if len(x) >= target_len:
        return x[:target_len]
    npad = target_len - len(x)
    left = npad // 2
    right = npad - left
    return np.pad(x, (left, right), mode="reflect")


class WaveletDecomposer:
    """Multilevel DWT wrapper around PyWavelets for 1-D embedding vectors."""

    def __init__(
        self,
        wavelet_name: str,
        level: Optional[int] = None,
        mode: str = "periodization",
    ):
        if wavelet_name not in _PYWT_NAME_MAP:
            raise ValueError(
                f"Unknown wavelet '{wavelet_name}'. Choose from {SUPPORTED_WAVELETS}"
            )
        _require_pywt()
        self.wavelet_name = wavelet_name
        self.pywt_name = _PYWT_NAME_MAP[wavelet_name]
        self.wavelet = pywt.Wavelet(self.pywt_name)
        self.mode = mode
        self.default_level = level

    # ------------------------------------------------------------------ #
    def max_level(self, signal_length: int) -> int:
        """Maximum usable level for a signal of given padded length."""
        return pywt.dwt_max_level(signal_length, self.wavelet.dec_len)

    # ------------------------------------------------------------------ #
    def decompose(self, x: np.ndarray) -> WaveletDecomposition:
        """Run multilevel DWT on a 1-D embedding vector.

        ``x`` must be a 1-D numpy array of length D (the embedding dim).
        """
        if x.ndim != 1:
            raise ValueError(
                "decompose expects a 1-D vector; got shape %r" % (x.shape,)
            )
        original_length = len(x)
        target = _next_pow2(original_length)
        padded = _pad_signal(np.asarray(x, dtype=np.float64), target)
        lvl = self.default_level or self.max_level(target)
        lvl = max(1, min(lvl, self.max_level(target)))
        coeffs = pywt.wavedec(
            padded, self.wavelet, level=lvl, mode=self.mode
        )
        # coeffs = [cA_lvl, cD_lvl, cD_{lvl-1}, ..., cD_1]
        approx = np.asarray(coeffs[0], dtype=np.float64)
        details = [np.asarray(c, dtype=np.float64) for c in coeffs[1:]]
        # Re-order details so index 0 is finest (wavelet level 1) for ease
        details = list(reversed(details))

        # Per-level energy
        energy_per_level: Dict[int, float] = {}
        for l, c in enumerate(details, start=1):
            energy_per_level[l] = float(np.sum(c ** 2))
        energy_approx = float(np.sum(approx ** 2))
        return WaveletDecomposition(
            wavelet_name=self.wavelet_name,
            level=lvl,
            approx=approx,
            details=details,
            padded_length=target,
            original_length=original_length,
            energy_per_level=energy_per_level,
            energy_approx=energy_approx,
        )

    # ------------------------------------------------------------------ #
    def reconstruct(
        self,
        decomposition: WaveletDecomposition,
        threshold_ratio: float = 0.0,
        crop: bool = True,
    ) -> np.ndarray:
        """Inverse transform. Optionally threshold the smallest-magnitude
        detail coefficients to ``threshold_ratio`` fraction of the total count
        (the compression experiment in the project spec).
        """
        # Re-order back to PyWavelets order: [cA, cD_lvl, ..., cD_1]
        details = list(reversed(decomposition.details))
        if 0.0 < threshold_ratio < 1.0:
            details = _threshold_details(details, threshold_ratio)
        coeffs = [decomposition.approx] + details
        rec = pywt.waverec(coeffs, self.wavelet, mode=self.mode)
        if crop and len(rec) >= decomposition.original_length:
            # Trim to original length with a center crop matching padding
            npad = decomposition.padded_length - decomposition.original_length
            left = npad // 2
            rec = rec[left: left + decomposition.original_length]
        return np.asarray(rec[: decomposition.original_length], dtype=np.float64)

    # ------------------------------------------------------------------ #
    def batch_decompose(self, X: np.ndarray) -> List[WaveletDecomposition]:
        """Decompose a batch of embeddings (N, D)."""
        if X.ndim != 2:
            raise ValueError("batch_decompose expects 2-D input (N, D)")
        return [self.decompose(X[i]) for i in range(X.shape[0])]


# --------------------------------------------------------------------------- #
# Helper: threshold detail coefficients (compression experiment)
# --------------------------------------------------------------------------- #

def _threshold_details(
    details: List[np.ndarray], ratio: float
) -> List[np.ndarray]:
    """Zero out the smallest-magnitude portion of detail coefficients.

    ``ratio`` of 0.30 means zero out the 30% smallest magnitude coefficients
    across *all* detail coefficients combined.
    """
    flat_sizes = [d.size for d in details]
    flat = np.concatenate([np.abs(d).ravel() for d in details])
    if flat.size == 0:
        return details
    n_zero = int(np.floor(ratio * flat.size))
    if n_zero == 0:
        return [d.copy() for d in details]
    # Indices of the n_zero smallest magnitudes -> zero them
    idx_sorted = np.argpartition(flat, n_zero - 1)[:n_zero]
    mask = np.ones_like(flat, dtype=bool)
    mask[idx_sorted] = False
    out: List[np.ndarray] = []
    cursor = 0
    for d, sz in zip(details, flat_sizes):
        sub_mask = mask[cursor: cursor + sz]
        new = np.where(sub_mask, d, 0.0)
        out.append(new.reshape(d.shape))
        cursor += sz
    return out


# --------------------------------------------------------------------------- #
# Wavelet similarity (frequency-domain cosine)
# --------------------------------------------------------------------------- #

def wavelet_similarity(a: WaveletDecomposition, b: WaveletDecomposition) -> float:
    """Compare two embeddings with a wavelet-aware cosine.

    We concatenate [approx, cD_1, ..., cD_L] padded/truncated to equal length
    and compute cosine similarity in that coefficient space.
    """
    va = _flatten_coefficients(a)
    vb = _flatten_coefficients(b)
    n = min(len(va), len(vb))
    va, vb = va[:n], vb[:n]
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _flatten_coefficients(d: WaveletDecomposition) -> np.ndarray:
    parts = [d.approx] + d.details
    return np.concatenate([p.ravel() for p in parts])
