"""Compression experiment & reconstruction-error analytics.

Implements the central compression experiment from the project spec:

  * Zero out 10%, 20%, ..., 50% of the smallest-magnitude detail
    coefficients across all detail bands.
  * Run inverse DWT to reconstruct the embedding.
  * Measure against the *original* (pre-DWT) signal:

      - cosine(original, reconstructed)
      - L2 / relative drift
      - signal-to-noise ratio (SNR)
      - top-k nearest-neighbour preservation (k = 5 / 10 by default)
      - compression ratio achieved

Functions here are independent of the wavelet family used (they just call
the decomposer's ``reconstruct`` method).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from wavelets.base import WaveletDecomposer, WaveletDecomposition


# --------------------------------------------------------------------------- #
# Compression ratios to test
# --------------------------------------------------------------------------- #

DEFAULT_COMPRESSION_RATIOS: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)


# --------------------------------------------------------------------------- #
# Single-vector compression result
# --------------------------------------------------------------------------- #

@dataclass
class CompressionResult:
    ratio: float                       # how many small coeffs were zeroed
    reconstructed: np.ndarray          # (D,)
    cosine: float                      # cos(orig, reconstructed)
    drift: float                       # ||orig - reconstructed||_2
    relative_drift: float              # drift / ||orig||_2
    snr_db: float                      # 20*log10(||orig|| / drift)
    energy_retained: float             # fraction of wavelet energy retained
    compression_ratio: float            # total / n_nonzero (always >= 1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _nonzero_count(d: WaveletDecomposition, ratio: float) -> int:
    """How many wavelet coefficients survive thresholding at ``ratio``.

    Mirrors ``wavelets.base._threshold_details``: the zeroed count is
    ``floor(ratio * n_detail_coeffs)`` taken over the *detail* bands only
    (the approximation coefficients are never touched).
    """
    detail_total = sum(c.size for c in d.details)
    n_zero = min(int(np.floor(ratio * detail_total)), detail_total)
    return d.approx.size + detail_total - n_zero


def _retained_energy_fraction(d: WaveletDecomposition, ratio: float) -> float:
    """Fraction of total wavelet energy retained after zeroing the
    smallest-magnitude ``ratio`` fraction of the detail coefficients.

    Replicates the exact coefficient selection used by
    ``wavelets.base._threshold_details`` (argpartition on magnitudes).
    """
    if not d.details or not (0.0 < ratio < 1.0):
        return 1.0
    flat = np.concatenate([np.abs(c).ravel() for c in d.details])
    n_zero = int(np.floor(ratio * flat.size))
    if n_zero <= 0:
        return 1.0
    idx = np.argpartition(flat, n_zero - 1)[:n_zero]
    zeroed_e = float(np.sum(flat[idx] ** 2))
    total = d.energy_approx + float(np.sum(flat ** 2))
    if total <= 0.0:
        return 1.0
    return max(0.0, 1.0 - zeroed_e / total)


# --------------------------------------------------------------------------- #
# Single embedding
# --------------------------------------------------------------------------- #

def compress_embedding(
    original: np.ndarray,
    decomposer: WaveletDecomposer,
    ratios: Tuple[float, ...] = DEFAULT_COMPRESSION_RATIOS,
) -> List[CompressionResult]:
    """Decompose -> threshold ratios -> reconstruct -> measure."""
    decomp = decomposer.decompose(original)
    norm_orig = np.linalg.norm(original) or 1e-12
    results: List[CompressionResult] = []
    for r in ratios:
        rec = decomposer.reconstruct(decomp, threshold_ratio=r, crop=True)
        # Trim/pad reconstruction to original length
        if rec.shape[0] != original.shape[0]:
            n = min(len(rec), len(original))
            rec_cropped = np.zeros_like(original)
            rec_cropped[:n] = rec[:n]
            rec = rec_cropped
        diff = original - rec
        drift = float(np.linalg.norm(diff))
        results.append(CompressionResult(
            ratio=r,
            reconstructed=rec,
            cosine=_cosine(original, rec),
            drift=drift,
            relative_drift=drift / norm_orig,
            snr_db=20.0 * np.log10(norm_orig / (drift or 1e-12)),
            energy_retained=_retained_energy_fraction(decomp, r),
            compression_ratio=float(sum(len(c) for c in [decomp.approx] + decomp.details) /
                                     max(_nonzero_count(decomp, r), 1)),
        ))
    return results


# --------------------------------------------------------------------------- #
# Batch compression - whole embedding matrix
# --------------------------------------------------------------------------- #

@dataclass
class BatchCompressionStats:
    ratio: float
    mean_cosine: float
    mean_drift: float
    mean_snr_db: float
    mean_relative_drift: float
    mean_compression_ratio: float
    std_cosine: float
    std_snr_db: float


def compress_batch(
    embeddings: np.ndarray,
    decomposer: WaveletDecomposer,
    ratios: Tuple[float, ...] = DEFAULT_COMPRESSION_RATIOS,
) -> Tuple[np.ndarray, Dict[float, BatchCompressionStats]]:
    """Compress a (N, D) embedding matrix.

    Returns
    -------
    reconstructed_matrix : np.ndarray  (len(ratios) maybe)
                          - Actually returns dict mapping ratio -> (N, D) reconstructions
    stats                : per-ratio summary
    """
    n = embeddings.shape[0]
    recs: Dict[float, np.ndarray] = {r: np.zeros_like(embeddings) for r in ratios}
    cos_store: Dict[float, List[float]] = {r: [] for r in ratios}
    drift_store: Dict[float, List[float]] = {r: [] for r in ratios}
    snr_store: Dict[float, List[float]] = {r: [] for r in ratios}
    rel_store: Dict[float, List[float]] = {r: [] for r in ratios}
    cr_store: Dict[float, List[float]] = {r: [] for r in ratios}

    for i in range(n):
        res = compress_embedding(embeddings[i], decomposer, ratios=ratios)
        for cr in res:
            recs[cr.ratio][i] = cr.reconstructed
            cos_store[cr.ratio].append(cr.cosine)
            drift_store[cr.ratio].append(cr.drift)
            snr_store[cr.ratio].append(cr.snr_db)
            rel_store[cr.ratio].append(cr.relative_drift)
            cr_store[cr.ratio].append(cr.compression_ratio)

    stats = {}
    for r in ratios:
        cs = np.asarray(cos_store[r])
        sn = np.asarray(snr_store[r])
        dr = np.asarray(drift_store[r])
        rl = np.asarray(rel_store[r])
        cr = np.asarray(cr_store[r])
        stats[r] = BatchCompressionStats(
            ratio=r,
            mean_cosine=float(cs.mean()) if cs.size else 0.0,
            mean_drift=float(dr.mean()) if dr.size else 0.0,
            mean_snr_db=float(sn.mean()) if sn.size else 0.0,
            mean_relative_drift=float(rl.mean()) if rl.size else 0.0,
            mean_compression_ratio=float(cr.mean()) if cr.size else 0.0,
            std_cosine=float(cs.std()) if cs.size else 0.0,
            std_snr_db=float(sn.std()) if sn.size else 0.0,
        )
    return recs, stats


# --------------------------------------------------------------------------- #
# Neighbor preservation
# --------------------------------------------------------------------------- #

def nearest_neighbors(
    query_vec: np.ndarray,
    candidates: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Indices of the k nearest candidates by cosine similarity."""
    if candidates.shape[0] == 0:
        return np.array([], dtype=int)
    qn = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    cn = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-12)
    sims = cn @ qn
    if k >= sims.size:
        order = np.argsort(-sims)
    else:
        order = np.argpartition(-sims, k)[:k]
        order = order[np.argsort(-sims[order])]
    return order


def neighbor_preservation(
    original_matrix: np.ndarray,
    reconstructed_matrix: np.ndarray,
    k: int = 10,
    sample: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute mean top-k neighbour preservation & neighbour drift.

    Returns
    -------
    preservation : float  (mean Jaccard overlap of top-k sets)
    drift        : float  (mean cosine distance between matched neighbours)
    """
    n = original_matrix.shape[0]
    if sample is None:
        sample = np.arange(n)
    sample = np.asarray(sample, dtype=int)
    overlaps = []
    drifts = []
    for i in sample:
        orig_nn = set(nearest_neighbors(original_matrix[i], np.delete(original_matrix, i, axis=0), k=k).tolist())
        rec_idx = [j for j in range(n) if j != i]
        rec_vec = reconstructed_matrix[i]
        rec_cands = reconstructed_matrix[rec_idx]
        rec_nn_idx = nearest_neighbors(rec_vec, rec_cands, k=k)
        rec_nn = set(np.take(rec_idx, rec_nn_idx).tolist())
        # Jaccard with top-k (excluding self)
        inter = orig_nn & rec_nn
        union = orig_nn | rec_nn
        overlaps.append(len(inter) / max(1, len(union)))
        # Drift = mean cosine distance between original and reconstructed vectors
        # for the matched neighbours
        d = 1.0 - _cosine(original_matrix[i], reconstructed_matrix[i])
        drifts.append(d)
    return float(np.mean(overlaps) if overlaps else 0.0), float(np.mean(drifts) if drifts else 0.0)
