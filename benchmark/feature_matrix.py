"""Per-head feature matrix builder for the Phase-6 combined-predictor
experiment.

This module is the single source of truth for which features feed the
ridge-regression importance predictors (``ridge_wavelet_only``,
``ridge_attn_only``, ``ridge_combined``) registered in
``benchmark.baselines_published``.

Design rules
------------

* The feature spec is **frozen**: a tuple of ``(key, source)`` pairs where
  ``source`` is one of:

      - ``"wavelet_row"``  : value comes from ``rows[i][key]`` as produced
                            by ``attention.analyzer.compute_head_metrics``
      - ``"attention"``   : value comes from a per-head attention-derived
                            scalar computed here from
                            ``extra["head_attention"][i]``
      - ``"layer_norm"``  : value comes from ``rows[i]["layer"]`` (treated
                            as a real-valued feature so the ridge can pick
                            up depth-related importance trends)

* **No leakage surface**: the matrix builder is pure -- it consumes the
  ``rows`` + ``extra`` produced for a single cell and returns its own
  ``X`` (standardised in-place using *that cell's* mean/std). The ridge
  weights are *fit* in a separate leave-one-cell-out loop in
  ``benchmark.baselines_published``, never on the same cell. The
  standardisation inside the builder uses the held-out cell's own
  statistics; the fitted ridge sees only the training cells' standardised
  X. (This is the standard "leave-one-site-out"FFE protocol as used in
  multi-site medical-imaging prediction studies.)

* Add a feature by registering it in ``WAVELET_FEATURES`` /
  ``ATTENTION_FEATURES`` and (if a new derived scalar) the corresponding
  computation in ``_attention_scalar``. Do **not** add features outside
  this list -- the ridge predictors must use the spec verbatim so the
  ablation ``ridge_wavelet_only`` vs ``ridge_combined`` is well-defined.

Feature inventory (Phase-6 frozen spec)
---------------------------------------

Wavelet metrics (from ``attention.analyzer.compute_head_metrics``):

  * ``total_energy``               total L2 energy across all levels
  * ``low_freq_energy``            approximation + bottom half of detail
                                   levels (smoother component)
  * ``high_freq_energy``           top half of detail levels (fine component)
  * ``energy_ratio_low_high``      ``e_low / e_high``
  * ``shannon_entropy``            Shannon entropy of the wavelet-coefficient
                                   magnitudes (the original "entropy" term
                                   of the Phase-5 ``wavelet`` predictor)
  * ``spectral_entropy``           Shannon entropy of the per-level energy
                                   distribution (a coarser spectral descriptor
                                   than ``shannon_entropy``)
  * ``gini_sparsity``              Gini coefficient of the wavelet magnitudes
  * ``reconstruction_error_30pct`` ‖A - reconstruct(keep top-30 % coeffs)‖_F
  * ``compression_ratio_99``       total coeffs / number to keep for 99 % E
  * ``dominant_level``             level index with most energy (a small int)

Attention scalars (computed here from raw head attention):

  * ``attention_entropy_true``    mean per-query-row ``-Σ p log p`` over the
                                   softmax-NORMALIZED attention distribution.
                                   Distinct from ``shannon_entropy`` above.

Combined feature vectors
------------------------

* ``ridge_wavelet_only``   uses only the WAVELET_FEATURES
* ``ridge_attn_only``       uses only the attention-derived feature
                            ``attention_entropy_true``
* ``ridge_combined``        uses WAVELET_FEATURES + ATTENTION_FEATURES
                            together; this is the predictor that answers
                            "do wavelet features carry information that
                            attention entropy alone does not?".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Feature spec
# --------------------------------------------------------------------------- #

# Wavelet-coefficient-derived fields produced by attention.analyzer.
# Frozen for Phase-6 so the ablation is well-defined.
WAVELET_FEATURES: Tuple[str, ...] = (
    "total_energy",
    "low_freq_energy",
    "high_freq_energy",
    "energy_ratio_low_high",
    "shannon_entropy",          # wavelet-coefficient entropy (NOT attention)
    "spectral_entropy",
    "gini_sparsity",
    "reconstruction_error_30pct",
    "compression_ratio_99",
    "dominant_level",
)

# Per-head attention-derived scalars computed from extra["head_attention"].
# Each tuple is (key, attention_feature_name_in_module).
ATTENTION_FEATURES: Tuple[str, ...] = ("attention_entropy_true",)


def feature_names(group: str) -> Tuple[str, ...]:
    """Return the ordered feature names for the requested predictor group.

    ``group`` is one of ``"wavelet_only"``, ``"attn_only"`` or
    ``"combined"``. Names are returned in a stable order so that consumers
    on both the fit side and the predict side index the same columns.
    """
    if group == "wavelet_only":
        return WAVELET_FEATURES
    if group == "attn_only":
        return ATTENTION_FEATURES
    if group == "combined":
        return WAVELET_FEATURES + ATTENTION_FEATURES
    raise ValueError(
        f"Unknown feature group '{group}'. "
        "Use one of: 'wavelet_only', 'attn_only', 'combined'."
    )


# --------------------------------------------------------------------------- #
# Attention-derived scalars
# --------------------------------------------------------------------------- #

def _attention_entropy_true(A: np.ndarray) -> float:
    """``H(A) = mean_t(-Σ_u p_{tu} log p_{tu})``.

    Mirrors ``pruning.registry.predict_attention_entropy_true`` exactly
    -- deliberately duplicated rather than imported to keep the feature
    builder self-contained and the spec explicit.
    """
    P = np.clip(np.asarray(A, dtype=np.float64), 1e-12, None)
    P = P / P.sum(axis=-1, keepdims=True)
    H = -(P * np.log(P)).sum(axis=-1)
    return float(H.mean())


def _attention_scalar(key: str, A: np.ndarray) -> float:
    """Dispatch a single attention-derived scalar."""
    if key == "attention_entropy_true":
        return _attention_entropy_true(A)
    raise ValueError(f"Unknown attention scalar '{key}'.")


# --------------------------------------------------------------------------- #
# Per-cell matrix construction
# --------------------------------------------------------------------------- #

def build_feature_matrix(
        rows: Sequence[Dict],
        extra: Optional[Dict[str, np.ndarray]] = None,
        group: str = "combined",
        ) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Return ``(X, feature_names)`` for one cell.

    ``X`` has shape ``(n_heads, n_features)``. Each column is z-scored
    **within this cell only** (subtract mean, divide by std, fall back to
    zeros when std == 0). This is the standardisation protocol required
    by the leave-one-cell-out ridge: the held-out cell carries its own
    scaler, so the fitted ridge -- which only ever saw training cells --
    never has access to the held-out cell's response scale.

    The ridge weights treat each column as informative; the layer index
    is NOT included (deliberately omitted to avoid the model spuriously
    memorising "layer i is generally important" -- we want the predictor
    to be transferable across the architectural differences).
    """
    names = feature_names(group)
    n_heads = len(rows)
    n_feat = len(names)
    X = np.zeros((n_heads, n_feat), dtype=np.float64)
    head_attn = (extra or {}).get("head_attention")
    for j, key in enumerate(names):
        if key in ATTENTION_FEATURES:
            if head_attn is None:
                # No raw attention accessible: zero column (ridge is
                # degenerate on this feature for this cell, but the column
                # remains in-place so the fitted weights from other cells
                # can still be applied). Documented in the Phase-6 note.
                continue
            for i, A in enumerate(head_attn[:n_heads]):
                X[i, j] = _attention_scalar(key, A)
        else:
            for i, r in enumerate(rows[:n_heads]):
                v = r.get(key, 0.0)
                # dominant_level is an int but the ridge expects float.
                X[i, j] = float(v) if v is not None else 0.0
    # Within-cell standardisation.
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-12] = 1.0
    X = (X - mu) / sd
    return X, names
