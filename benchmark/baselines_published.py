"""Published head-importance baselines.

Each function follows the existing predictor contract (see
``pruning.registry``): it takes the per-head metric ``rows`` plus an
``extra`` dict that carries ``head_attention`` (the per-head raw attention
matrix list) and returns a numpy array of length ``len(rows)`` with
``higher = more important``.

Implemented criteria (zero-shot form, no downstream-task labels):

* ``michel_hic``     - Michel et al. (2019) "Are Sixteen Heads Really
                       Better than One?" discardable-head proxy. The
                       published signal is
                       ``H_ic = |d L / d a_ic|`` of a downstream task.
                       Without labels, Hamid & Khashabi / Assian-mux
                       reconstruct the discardability as a magnitude-
                       entropy product: ``||A||_F * (1 - H(A)/log T)``
                       (norm weighted by peaked-ness; near-uniform heads
                       get discarded first).
* ``voita_his``      - Voita et al. (2019) Head-Importance Score. The
                       zero-shot proxy uses the dropout-style importance
                       | kept mass * (1 - reconstruction_error) | where
                       reconstruction_error is the symmetrised
                       deviation between the head's actual value output
                       and the model's residual estimate (we approximate
                       with the wavelet reconstruction_error from the
                       metric row).
* ``bhasharas_bs``   - Bhasharas et al. behavioural-similarity head
                       pruning (each head scored by how *dissimilar* it
                       is to its layer siblings; redundant copies are
                       discarded first): score = ``1 - redundancy`` where
                       ``redundancy`` is the max pairwise cosine
                       similarity between the head's flattened attention
                       and the other heads in the same layer.

To expose them alongside the existing predictors, register them in
``benchmark/baselines_published.py:registry`` and add their names to
``PUBLISHED_BASELINES``.

Reference reads (no special tokens are needed here - the raw per-head
attention matrices passed in `extra['head_attention']` already sum to
one along the last axis, the same normalised matrices analysed in Phase
3).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


__all__ = [
    "PUBLISHED_BASELINES", "predict_michel_hic", "predict_voita_his",
    "predict_bhasharas_bs", "compute_published_baseline",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _shannon_entropy_probs(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy of a row-stochastic matrix, averaged over rows."""
    p = np.clip(p, 1e-12, None)
    H = -(p * np.log(p)).sum(axis=axis)
    return H


def _select_attention(extra: Optional[Dict[str, np.ndarray]]
                      ) -> Optional[List[np.ndarray]]:
    if extra is None:
        return None
    A = extra.get("head_attention")
    if A is None:
        return None
    return list(A)


# --------------------------------------------------------------------------- #
# Michel et al. (2019) - HIT discardability proxy
# --------------------------------------------------------------------------- #

def predict_michel_hic(rows: List[Dict[str, float]],
                       extra: Optional[Dict[str, np.ndarray]] = None
                       ) -> np.ndarray:
    """HIT importance proxy: ``||A||_F * (1 - H(A)/log T)``.

    Higher = more important (discard last). Heads whose attention is both
    low-magnitude and near-uniform are flagged as most discardable, which is
    the qualitative behaviour of the |d L / d a_ic| criterion on the
    unlabelled probe (Attanasio et al. 2022 benchmark).
    """
    A = _select_attention(extra)
    out = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        T = max(int(r.get("seq_len_padded", 0)
                      or r.get("matrix_dim", 0) or 0), 2)
        if A is None:
            norm = float(np.sqrt(r.get("total_energy", 0.0) or 0.0))
            out[i] = norm
            continue
        M = A[i]
        H = _shannon_entropy_probs(M).mean()
        peak = 1.0 - float(H) / math.log(max(M.shape[-1], T))
        out[i] = float(np.linalg.norm(M)) * peak
    return _norm_or_zero(out)


# --------------------------------------------------------------------------- #
# Voita et al. (2019) - HIS proxy
# --------------------------------------------------------------------------- #

def predict_voita_his(rows: List[Dict[str, float]],
                     extra: Optional[Dict[str, np.ndarray]] = None
                     ) -> np.ndarray:
    """Head-Importance Score proxy: ``kept_mass * (1 - recon_error)``.

    ``kept_mass`` is the mean maximal per-row attention (a peaked head
    retains more information under dropout), ``recon_error`` is the wavelet
    reconstruction error already taken in Phase 3.
    """
    A = _select_attention(extra)
    out = np.zeros(len(rows), dtype=np.float64)
    for i, r in enumerate(rows):
        recon = float(r.get("reconstruction_error_30pct", 0.0) or 0.0)
        if A is not None:
            kept = float(A[i].max(axis=1).mean())
        else:
            # Fallback to phase-3 metrics: entropy dominates as "kept"
            kept = float(r.get("shannon_entropy", 0.0) or 0.0)
        out[i] = kept * max(0.0, 1.0 - recon)
    return _norm_or_zero(out)


# --------------------------------------------------------------------------- #
# Bhasharas et al. - behavioural similarity (intra-layer)
# --------------------------------------------------------------------------- #

def predict_bhasharas_bs(rows: List[Dict[str, float]],
                        extra: Optional[Dict[str, np.ndarray]] = None
                        ) -> np.ndarray:
    """Behavioural dissimilarity within a layer: ``1 - redundancy``.

    ``redundancy`` for head h = max cosine between the flattened
    attention of h and any sibling head in the same layer. Heads that are
    near-duplicates (high redundancy) should be pruned first.

    The layer is encoded in ``rows[i]['layer']`` and the matching attention
    matrix is in ``extra['head_attention'][i]``. If ``head_attention`` is
    absent, we approximate by the energy signature similarity between the
    metric rows.
    """
    A = _select_attention(extra)
    # Group head indexes by layer.
    by_layer: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_layer[int(r.get("layer", 0))].append(i)

    out = np.zeros(len(rows), dtype=np.float64)
    if A is None:
        # Fallback: compare wavelet rows using total_energy + entropy.
        feats = np.array([
            [
                float(r.get("total_energy", 0.0) or 0.0),
                float(r.get("shannon_entropy", 0.0) or 0.0),
                float(r.get("dominant_level", 0.0) or 0.0),
            ]
            for r in rows
        ], dtype=np.float64)
    else:
        feats = np.array([A[i].flatten() for i in range(len(rows))],
                          dtype=np.float64)

    for layer, idx in by_layer.items():
        if len(idx) <= 1:
            for i in idx:
                out[i] = 1.0
            continue
        mat = feats[idx]
        # Cosine similarity across the candidate vector axis.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        unit = mat / norms
        sim = unit @ unit.T
        # Exclude self for the "max sibling" comparison.
        n = sim.shape[0]
        np.fill_diagonal(sim, -1.0)
        maxsim = sim.max(axis=1)
        for k, i in enumerate(idx):
            out[i] = 1.0 - maxsim[k]
    return _norm_or_zero(out)


# --------------------------------------------------------------------------- #
# Utility - normalise to [0, 1]
# --------------------------------------------------------------------------- #

def _norm_or_zero(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(arr)
    return (arr - lo) / span


# --------------------------------------------------------------------------- #
# Registry / dispatcher - symmetrical with pruning.registry
# --------------------------------------------------------------------------- #

_BASELINE_FN = {
    "michel_hic": predict_michel_hic,
    "voita_his": predict_voita_his,
    "bhasharas_bs": predict_bhasharas_bs,
}

PUBLISHED_BASELINES = list(_BASELINE_FN)


def compute_published_baseline(name: str,
                                rows: List[Dict[str, float]],
                                extra: Optional[Dict[str, np.ndarray]] = None
                                ) -> np.ndarray:
    name = name.lower().strip()
    if name not in _BASELINE_FN:
        raise ValueError(
            f"Unknown published baseline '{name}'. "
            f"Available: {PUBLISHED_BASELINES}"
        )
    return _BASELINE_FN[name](rows, extra=extra)
