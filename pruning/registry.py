"""Importance predictors for Phase-4 head pruning.

A *predictor* turns per-head wavelet metrics (energy, entropy, gini,
compression ratio, dominant level, reconstruction error, energy ratio
low/high) into a single `predicted_importance` scalar where **higher =
more important** (i.e. *less* prunable).

A separate `redundancy_score` output reverses the sign for similarity-style
redundancy measures (where higher = more redundant).

Available predictors (each idempotent, callable with ``rows: List[Dict])``):

  * ``wavelet``                weighted combo: wavelet_entropy + gini +
                                0.5 * reconstruction_error + 0.5 * energy_ratio
  * ``wavelet_entropy``        Shannon entropy of the wavelet-coefficient
                                magnitudes (single-term, NEW name).
  * ``attention_entropy``      DEPRECATED alias of ``wavelet_entropy``.
                                Kept for back-compat with Phase-5 result
                                bundles whose ``summary.json`` lists this
                                name under ``predictors``. The Phase-5
                                implementation read the wavelet-coefficient
                                ``shannon_entropy`` field, *not* the
                                attention-distribution entropy -- see the
                                Phase-6 note in ``benchmark/README.md``.
  * ``attention_entropy_true`` Shannon entropy of the *attention
                                distribution* itself (``-sum p log p``
                                averaged over query rows). NEW in Phase 6;
                                required for the "wavelet vs entropy vs
                                combined" headline comparison to be
                                scientifically honest.
  * ``attention_weight``  average attention diagonal mass
  * ``magnitude``         L2 norm of the per-head attention matrix
  * ``random``            uniform sample

All are deterministic (seed-able) and return a numpy array.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Predictor registry
# --------------------------------------------------------------------------- #

PREDICTOR_NAMES: List[str] = [
    "wavelet",
    "wavelet_entropy",
    "attention_entropy",
    "attention_entropy_true",
    "attention_weight",
    "magnitude",
    "random",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _norm(arr: np.ndarray) -> np.ndarray:
    z = arr - arr.min()
    zin = arr.max() - arr.min()
    return z / zin if zin > 0 else np.zeros_like(z)


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #

def predict_wavelet(rows: List[Dict[str, float]],
                     alpha=(1.0, 1.0, 0.5, 0.5)) -> np.ndarray:
    """Composite predicted importance using Phase-3 wavelet metrics.

    Higher = keep the head. We mix:
      * entropy (high -> varied spectrum -> keep)
      * gini sparsity (high -> concentrated spectrum -> keep)
      * reconstruction_error (high -> attention not well-representable
                                 by sparse wavelets -> information-rich -> keep)
      * dominant_level * energy_ratio (taller -> contains frequency
                                          information hard to remove -> keep)

    The weights are exposed in ``alpha`` so the experiment runner can sweep.
    The default weights come from the specification:
      "Wavelet score = low-frequency Energy / high-frequency Energy"
    and entropy.  We additionally add reconstruction-error so heads with
    structurally rich attention are credited.
    """
    e = np.array([float(r.get("shannon_entropy", 0.0)) for r in rows])
    g = np.array([float(r.get("gini_sparsity", 0.0)) for r in rows])
    rec = np.array([float(r.get("reconstruction_error_30pct",
                                   0.0)) for r in rows])
    er = np.array([float(r.get("energy_ratio_low_high", 0.0))
                    for r in rows])
    a, b, c, d = alpha
    score = a * e + b * g + c * rec + d * er
    return score


def predict_wavelet_entropy(rows: List[Dict[str, float]],
                             extra: Optional[Dict[str, np.ndarray]] = None
                             ) -> np.ndarray:
    """Shannon entropy of the per-head wavelet-coefficient magnitudes.

    Higher = more spectrally varied head, predicted more important.
    This is the *wavelet*-domain entropy (computed over the DWT
    magnitudes by ``attention.analyzer.compute_head_metrics``).
    """
    return np.array([float(r.get("shannon_entropy", 0.0)) for r in rows])


# Back-compat alias: the Phase-5 ``attention_entropy`` predictor read the
# same wavelet-coefficient ``shannon_entropy`` field, despite its name
# implying attention-distribution entropy. Preserve the historical name so
# ``summary.json`` Predictor lists stay readable, and route it to the
# correctly-named implementation. The truly-attention-distribution version
# is exposed separately as ``predict_attention_entropy_true`` below.
predict_attention_entropy = predict_wavelet_entropy


def predict_attention_entropy_true(
        rows: List[Dict[str, float]],
        extra: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
    """Shannon entropy of the **attention distribution** itself.

    ``H(A) = mean_t(-sum_u p_{tu} log p_{tu})`` where ``p_{tu}`` is the
    per-query-row softmax-normalised attention mass. Heads with diffuse
    ("uncertain") attention score high.

    Falls back to ``rows[i]['shannon_entropy']`` only when the raw
    attention matrices are unavailable (``extra`` missing
    ``'head_attention'``); in that regime the wavelet-coefficient entropy
    is the only accessible proxy and the caller should label it as such.
    """
    if extra is None or "head_attention" not in extra:
        return np.array([float(r.get("shannon_entropy", 0.0)) for r in rows])
    out: List[float] = []
    for A in extra["head_attention"]:
        # Rows are already softmaxed by attention extraction; clip for safety.
        P = np.clip(np.asarray(A, dtype=np.float64), 1e-12, None)
        P = P / P.sum(axis=-1, keepdims=True)
        H = -(P * np.log(P)).sum(axis=-1)
        out.append(float(H.mean()))
    return np.array(out)


def predict_attention_weight(rows: List[Dict[str, float]],
                              extra: Optional[Dict[str, np.ndarray]] = None
                              ) -> np.ndarray:
    """Average maximum attention mass per query row - heads with concentrated
    attention (high avg attn weight) are predicted to be more important."""
    if extra is None or "head_attention" not in extra:
        return np.zeros(len(rows))
    out = []
    for A in extra["head_attention"]:
        out.append(float(np.mean(A.max(axis=1))))
    return np.array(out)


def predict_magnitude(rows: List[Dict[str, float]],
                      extra: Optional[Dict[str, np.ndarray]] = None
                      ) -> np.ndarray:
    """L2 norm of the head's attention matrix as the importance proxy."""
    if extra is None or "head_attention" not in extra:
        return np.array([float(np.sqrt(r.get("total_energy",
                                                 0.0))) for r in rows])
    out = []
    for A in extra["head_attention"]:
        out.append(float(np.linalg.norm(A)))
    return np.array(out)


def predict_random(rows: List[Dict[str, float]], seed: int = 0
                   ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(len(rows))


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def get_predictor(name: str) -> Callable[..., np.ndarray]:
    name = name.lower().strip()
    if name == "wavelet":
        return predict_wavelet
    if name == "wavelet_entropy":
        return predict_wavelet_entropy
    if name in ("attention_entropy", "attention_entropy_wavelet"):
        # Back-compat alias -- see module docstring. Routes to
        # predict_wavelet_entropy (the wavelet-coefficient entropy) so
        # historical Phase-5 summary.json references remain reproducible.
        return predict_wavelet_entropy
    if name == "attention_entropy_true":
        return predict_attention_entropy_true
    if name == "attention_weight":
        return predict_attention_weight
    if name == "magnitude":
        return predict_magnitude
    if name == "random":
        return predict_random
    raise ValueError(f"Unknown predictor '{name}'. "
                     f"Available: {PREDICTOR_NAMES}")


def compute_predictor(name: str,
                       rows: List[Dict[str, float]],
                       extra: Optional[Dict[str, np.ndarray]] = None,
                       seed: int = 0) -> np.ndarray:
    """Head-importance scores per the named predictor.

    Higher = more important (predicted).
    """
    fn = get_predictor(name)
    if name == "random":
        return fn(rows, seed=seed)
    if name in ("attention_weight", "magnitude",
                "attention_entropy_true"):
        return fn(rows, extra=extra)
    return fn(rows)


__all__ = [
    "PREDICTOR_NAMES", "predict_wavelet", "predict_wavelet_entropy",
    "predict_attention_entropy", "predict_attention_entropy_true",
    "predict_attention_weight", "predict_magnitude", "predict_random",
    "get_predictor", "compute_predictor",
]
