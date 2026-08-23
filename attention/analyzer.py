"""Two-dimensional wavelet decomposition & metrics for attention matrices.

For attention matrix ``A`` of shape (T, T):

  * pad symmetrically to a square power-of-two (next_pow2)
  * multilevel 2-D discrete wavelet transform (DWT2): dynamical levels
  * return:

      approx           : last-level approximation coeffs (cA)
      details          : dict[level -> {HH, HL, LH, HH... }] actually:
                         per-level dict ('cAH' (horizontal), 'cAV' (vertical),
                         'cAD' (diagonal))
      level            : max level used
      energy_per_level : per-level total energy (sum of squared coeffs
                          across the three detail subbands)
      energy_approx   : squared sum of approx coeffs.

Metric computations consume a :class:`HeadDecomposition` object.

Per-head numerical metrics
  * total_energy
  * low_freq_energy (approx + coarsest-half detail)
  * high_freq_energy (finest-half detail)
  * energy_ratio_low_high
  * shannon_entropy (over energy-normalised coefficient magnitudes)
  * spectral_entropy (Shannon over per-level energy distribution,
                       base-2)
  * sparsity (Gini coefficient)
  * dominant_frequency_level (level with greatest energy)
  * reconstruction_error at default 30% thresholding (L2)
  * compression_ratio (coefficient retention at 99% energy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pywt
    _HAS_PYWT = True
except ImportError:  # pragma: no cover
    _HAS_PYWT = False


def _require_pywt():
    if not _HAS_PYWT:
        raise ImportError("PyWavelets required: pip install PyWavelets")


# --------------------------------------------------------------------------- #
# Supported wavelets (same names as Phase 1)
# --------------------------------------------------------------------------- #

SUPPORTED_WAVELETS: List[str] = ["haar", "db4", "sym4", "coif2"]
_PYWT_NAME_MAP: Dict[str, str] = {"haar": "haar", "db4": "db4",
                                   "sym4": "sym4", "coif2": "coif2"}


# --------------------------------------------------------------------------- #
# Decomposition container
# --------------------------------------------------------------------------- #

@dataclass
class HeadDecomposition:
    wavelet_name: str
    level: int
    approx: np.ndarray                   # approx coeffs (cA) - last level
    details: Dict[int, Dict[str, np.ndarray]]   # level {'cAH','cAV','cAD'}
    energy_approx: float
    energy_per_level: Dict[int, float]
    padded_length: int                  # next pow2 size used during DWT
    original_shape: Tuple[int, int]
    energy_subband: Dict[int, Dict[str, float]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _pad_to_square_pow2(A: np.ndarray) -> Tuple[np.ndarray, int, int]:
    h, w = A.shape
    n = _next_pow2(max(h, w))
    pad_h = n - h
    pad_w = n - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return (np.pad(A, ((top, bottom), (left, right)),
                   mode="reflect"), top, left)


# --------------------------------------------------------------------------- #
# Decomposer (2D)
# --------------------------------------------------------------------------- #

class AttentionWaveletDecomposer:
    """Brief bridge to PyWavelets for 2D DWT."""

    def __init__(self, wavelet_name: str = "db4",
                 level: Optional[int] = None,
                 mode: str = "periodization"):
        if wavelet_name not in _PYWT_NAME_MAP:
            raise ValueError(f"Unknown wavelet '{wavelet_name}'. "
                             f"Choose from {SUPPORTED_WAVELETS}")
        _require_pywt()
        self.wavelet_name = wavelet_name
        self.pywt_name = _PYWT_NAME_MAP[wavelet_name]
        self.wavelet = pywt.Wavelet(self.pywt_name)
        self.level = level
        self.mode = mode

    # ------------------------------------------------------------------ #
    def decompose(self, A: np.ndarray) -> HeadDecomposition:
        """Run a multilevel 2-D DWT."""
        if A.ndim != 2:
            raise ValueError("Need a 2-D matrix")
        original_shape = A.shape
        n = _next_pow2(max(A.shape))
        padded, _, _ = _pad_to_square_pow2(A)
        lvl = self.level or pywt.dwt_max_level(min(padded.shape),
                                                self.wavelet.dec_len)
        lvl = max(1, lvl)
        coeffs = pywt.wavedec2(padded, self.wavelet, level=lvl,
                                mode=self.mode)
        # coeffs = [cA_lvl, (cAH_lvl, cAV_lvl, cAD_lvl), ...]
        approx = np.asarray(coeffs[0], dtype=np.float64)
        details: Dict[int, Dict[str, np.ndarray]] = {}
        energy_approx = float(np.sum(approx ** 2))
        energy_per_level: Dict[int, float] = {}
        energy_subband: Dict[int, Dict[str, float]] = {}
        for i, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
            d = {"cAH": np.asarray(cH, dtype=np.float64),
                 "cAV": np.asarray(cV, dtype=np.float64),
                 "cAD": np.asarray(cD, dtype=np.float64)}
            details[i] = d
            sub_e = {k: float(np.sum(v ** 2)) for k, v in d.items()}
            energy_subband[i] = sub_e
            energy_per_level[i] = sum(sub_e.values())
        return HeadDecomposition(
            wavelet_name=self.wavelet_name,
            level=lvl, approx=approx, details=details,
            energy_approx=energy_approx,
            energy_per_level=energy_per_level,
            padded_length=n,
            original_shape=original_shape,
            energy_subband=energy_subband,
        )

    # ------------------------------------------------------------------ #
    def reconstruct(
        self, decomp: HeadDecomposition,
        threshold_ratio: float = 0.0,
        threshold_band: str = "details",   # "details" | "all" | "approx"
        crop: bool = True
    ) -> np.ndarray:
        """Inverse 2-D DWT. Optionally zero the smallest-magnitude
        ``threshold_ratio`` fraction of *detail* coefficients across all
        detail subbands (all levels), matching the spec: remove smallest
        fraction across the full set of details."""
        approx = decomp.approx.copy()
        details = []
        for lvl in sorted(decomp.details.keys()):
            d = decomp.details[lvl].copy()
            for k in list(d.keys()):
                d[k] = d[k].copy() if isinstance(d[k], np.ndarray) else d[k]
            details.append((d["cAH"], d["cAV"], d["cAD"]))
        if 0.0 < threshold_ratio < 1.0 and threshold_band in ("details", "all"):
            details = _threshold_2d_details(details, threshold_ratio)
        if threshold_band == "all":
            # also threshold the approximation if requested
            flat = approx.ravel()
            if flat.size:
                k = int(np.floor(threshold_ratio * flat.size))
                if k > 0:
                    idx = np.argpartition(np.abs(flat), k - 1)[:k]
                    flat[idx] = 0
                    approx = flat.reshape(approx.shape)
        coeffs = [approx] + details
        rec = pywt.waverec2(coeffs, self.wavelet, mode=self.mode)
        if crop and rec.shape[0] >= decomp.original_shape[0] \
                  and rec.shape[1] >= decomp.original_shape[1]:
            n = rec.shape[0]
            h, w = decomp.original_shape
            top = (n - h) // 2
            left = (n - w) // 2
            rec = rec[top: top + h, left: left + w]
        return rec.astype(np.float32)


def _threshold_2d_details(details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                          ratio: float):
    """Zero out the smallest-magnitude ratio of all detail subband entries across levels."""
    flattened = []
    sizes = []
    for cH, cV, cD in details:
        for c in (cH, cV, cD):
            flattened.append(np.abs(c).ravel())
            sizes.append(c.size)
    all_mag = np.concatenate(flattened) if flattened else np.array([])
    if all_mag.size == 0:
        return details
    n_zero = int(np.floor(ratio * all_mag.size))
    if n_zero == 0:
        return [(cH.copy(), cV.copy(), cD.copy()) for cH, cV, cD in details]
    idx = np.argpartition(all_mag, n_zero - 1)[:n_zero]
    mask_zero = np.zeros_like(all_mag, dtype=bool)
    mask_zero[idx] = True
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    cursor = 0
    for cH, cV, cD in details:
        triple = []
        for c in (cH, cV, cD):
            sz = c.size
            sub_mask = mask_zero[cursor: cursor + sz].reshape(c.shape)
            new_c = np.where(sub_mask, 0.0, c).astype(c.dtype, copy=False)
            triple.append(new_c)
            cursor += sz
        out.append(tuple(triple))
    return out


# --------------------------------------------------------------------------- #
# Metric computations
# --------------------------------------------------------------------------- #

def _shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _spectral_entropy(energy_per_level: Dict[int, float]) -> float:
    """Shannon entropy over the per-level energy share distribution."""
    tot = sum(energy_per_level.values()) or 1.0
    p = np.array(sorted(energy_per_level.values())) / tot
    return _shannon_entropy(p)


def _gini(magnitudes: np.ndarray) -> float:
    s = np.sort(magnitudes)
    if s.size < 2 or s.sum() == 0:
        return 0.0
    n = s.size
    B = np.cumsum(s)[-1]
    if B == 0:
        return 0.0
    G = (2 * np.sum(np.arange(1, n + 1) * s) / (n * B)) - (n + 1) / n
    return float(G)


def _flatten_decomp_magnitudes(d: HeadDecomposition) -> np.ndarray:
    parts = [np.abs(d.approx).ravel()]
    for lvl, dd in d.details.items():
        for k in ("cAH", "cAV", "cAD"):
            parts.append(np.abs(dd[k]).ravel())
    return np.concatenate(parts)


# --------------------------------------------------------------------------- #
# Per-head metric dictionary
# --------------------------------------------------------------------------- #

METRIC_KEYS: Tuple[str, ...] = (
    "total_energy", "low_freq_energy", "high_freq_energy",
    "energy_ratio_low_high",
    "shannon_entropy", "spectral_entropy",
    "gini_sparsity", "dominant_level",
    "reconstruction_error_30pct", "compression_ratio_99",
    "matrix_dim", "data_volume", "seq_len_padded",
)


def compute_head_metrics(
    A: np.ndarray,
    decomposer: AttentionWaveletDecomposer,
    threshold_ratio_for_error: float = 0.30,
    energy_retention_for_cr: float = 0.99,
) -> Dict[str, float]:
    """Return the full per-head metrics dict for one attention matrix."""
    decomp = decomposer.decompose(A)
    e_total = decomp.energy_approx + sum(decomp.energy_per_level.values())
    # Split the levels half-and-half (coarsest half = low, finest half = high)
    levels = sorted(decomp.energy_per_level.keys())
    n = len(levels)
    if n <= 1:
        low_levels = []
        high_levels = levels
    else:
        half = max(1, n // 2)
        low_levels = levels[n - half:]
        high_levels = levels[: n - half]
    e_low = decomp.energy_approx + sum(decomp.energy_per_level[l]
                                         for l in low_levels)
    e_high = sum(decomp.energy_per_level[l] for l in high_levels)
    er = e_low / max(e_high, 1e-12)

    mags = _flatten_decomp_magnitudes(decomp)
    if mags.size == 0:
        shannon = 0.0
    else:
        s = mags.sum()
        shannon = _shannon_entropy(mags / s if s > 0 else mags)
    spec_ent = _spectral_entropy(decomp.energy_per_level)
    gini = _gini(mags)
    items = [(0, decomp.energy_approx)] + list(decomp.energy_per_level.items())
    dominant_l = max(items, key=lambda kv: kv[1])[0]
    # Reconstruction error at given threshold
    rec = decomposer.reconstruct(decomp, threshold_ratio=threshold_ratio_for_error,
                                  crop=True)
    rec_err = float(np.linalg.norm(A - rec))
    # Compression ratio at 99% energy
    cr = _compression_ratio(decomp, eps=1 - energy_retention_for_cr)
    h, w = A.shape
    return {
        "total_energy": float(e_total),
        "low_freq_energy": float(e_low),
        "high_freq_energy": float(e_high),
        "energy_ratio_low_high": float(er),
        "shannon_entropy": float(shannon),
        "spectral_entropy": float(spec_ent),
        "gini_sparsity": float(gini),
        "dominant_level": int(dominant_l),
        "reconstruction_error_30pct": rec_err,
        "compression_ratio_99": float(cr),
        "matrix_dim": int(h),
        "data_volume": int(h * w),
        "seq_len_padded": int(decomp.padded_length),
    }


def _compression_ratio(decomp: HeadDecomposition, eps: float = 0.01) -> float:
    """Total coefficients / number required to retain (1-eps) of energy."""
    e_arr = (_flatten_decomp_magnitudes(decomp) ** 2)
    if e_arr.size == 0:
        return 0.0
    order = np.argsort(-e_arr)
    cum = np.cumsum(e_arr[order])
    if cum[-1] == 0:
        return 0.0
    target = (1 - eps) * cum[-1]
    n_keep = int(np.searchsorted(cum, target)) + 1
    return float(e_arr.size / max(1, n_keep))


# --------------------------------------------------------------------------- #
# Feature vector for clustering / similarity
# --------------------------------------------------------------------------- #

def head_feature_vector(metrics: Dict[str, float]) -> np.ndarray:
    keys = ("total_energy", "low_freq_energy", "high_freq_energy",
            "energy_ratio_low_high", "shannon_entropy",
            "spectral_entropy", "gini_sparsity",
            "reconstruction_error_30pct", "compression_ratio_99")
    return np.array([metrics[k] for k in keys], dtype=np.float64)


def head_wavelet_flat_coeffs(decomp: HeadDecomposition, max_len: int = 256
                              ) -> np.ndarray:
    """Return concatenation of approx + per-level detail subbands, truncated
    or zero-padded to ``max_len`` so heads differ in seq_len compare.

    We use ``max_len`` = 256 so this is affordable for similarity matrices.
    """
    flat = _flatten_decomp_magnitudes(decomp)
    if flat.size > max_len:
        flat = flat[:max_len]
    elif flat.size < max_len:
        flat = np.pad(flat, (0, max_len - flat.size))
    return flat
