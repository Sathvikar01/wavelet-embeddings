"""Main Phase-4 experiment: does the wavelet score predict redundant heads?

Per ``predictor``:

  * score every (layer, head) with the predictor (wavelet or baseline)
  * treat the *predicted* redundant heads as the lowest ``score`` quantile
  * for each head (single-head ablation), measure cosine_drop / KL /
    attention_drift via :func:``run_model_strike``
  * compute Pearson r, Spearman rho, Kendall tau between the predictor's
    ``predicted_unimportance = -score`` and the *measured* accuracy loss

If wavelet predictor's correlation is hand-over-fist stronger than the
baselines, the Phase-4 hypothesis is supported.

In addition, a *batched* fairness sweep -- we rank heads by a predictor,
ablate the bottom ``p``% of them for ``p in {5, 10, 20, 30, 50}`` and report
the resulting ``kl_div_next_token`` / cosine_drop / attention_drift.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Iterable, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from embeddings.extract import EmbeddingExtractor
from attention import (
    AttentionLoader, AttentionWaveletDecomposer, compute_head_metrics,
)
from pruning.runner import HeadAblator
from pruning.registry import (
    compute_predictor, predict_wavelet, PREDICTOR_NAMES,
)
from evaluation.task_loss import (
    SentenceRun, AblationEffect, run_model, measure_effect,
)


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class HeadValidation:
    layer: int
    head: int
    predicted_importance: float
    predicted_unimportance: float     # = -score, for ranking prunables
    cosine_drop: float
    kl_div_next_token: float
    attention_drift: float


@dataclass
class PredictorValidationReport:
    predictor: str
    pearson_r: float
    spearman_rho: float
    kendall_tau: float
    per_head: List[HeadValidation]
    aggregate_kl: float
    aggregate_cosine: float
    surrogate_for_baseline_difference: float


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _rank_corrs(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (Pearson r, Spearman rho, Kendall tau) of two arrays."""
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr, spearmanr, kendalltau
        r, _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        tau, _ = kendalltau(x, y)
        return float(r), float(rho), float(tau)
    except Exception:
        # Fallback: pure-numpy implementations
        xm = x - x.mean(); ym = y - y.mean()
        r = (xm * ym).sum() / (np.linalg.norm(xm) * np.linalg.norm(ym) + 1e-12)
        # Spearman and Kendall numerically based on ranks
        def rank(a):
            order = np.argsort(np.argsort(a))
            return order.astype(np.float64)
        def spearman(a, b):
            return np.corrcoef(rank(a), rank(b))[0, 1]
        # Kendall simple: use the O(n^2) helper
        def kendall(a, b):
            n = len(a)
            concordant = 0; discordant = 0
            ra = rank(a); rb = rank(b)
            for i in range(n):
                for j in range(i + 1, n):
                    da = ra[j] - ra[i]
                    db = rb[j] - rb[i]
                    if da * db > 0:
                        concordant += 1
                    elif da * db < 0:
                        discordant += 1
            total = n * (n - 1) / 2 if n > 1 else 1
            return (concordant - discordant) / total
        return float(r), float(spearman(x, y)), float(kendall(x, y))


# --------------------------------------------------------------------------- #
# Per-head measurement
# --------------------------------------------------------------------------- #

def _per_head_validation(
    model,
    model_key: str,
    tokenizer,
    sentences: List[str],
    all_heads: List[Tuple[int, int]],
    orig_runs: List[SentenceRun],
    device: Optional[str] = None,
    ablation_mode: str = "zero",
) -> List[AblationEffect]:
    """Return per-head cosine / KL / drift effects when that *single* head
    is ablated via HF ``head_mask``.
    """
    out: List[AblationEffect] = []
    ab = HeadAblator(model, model_key, mode=ablation_mode)
    for L, H in all_heads:
        ab.ablate = {(L, H)}
        hm = ab.head_mask(device=device)
        runs = run_model(model, model_key, sentences, tokenizer,
                          device=device, head_mask=hm)
        eff = measure_effect(orig_runs, runs, n_ablated=1)
        out.append(eff)
    return out


# --------------------------------------------------------------------------- #
# Full validation runner
# --------------------------------------------------------------------------- #

def validate_predictor(
    model,
    model_key: str,
    tokenizer,
    sentences: List[str],
    rows: List[Dict[str, float]],
    extra: Dict[str, np.ndarray],
    predictor_name: str,
    device: Optional[str] = None,
    seed: int = 0,
    ablation_mode: str = "zero",
    cached_effects: Optional[List[AblationEffect]] = None,
) -> PredictorValidationReport:
    """Validate one predictor against measured single-head ablation damage.

    ``cached_effects``: pre-computed per-head ablation effects (from
    :func:`_per_head_validation`). The ablation sweep is predictor-
    independent, so callers validating *several* predictors on the same
    model + sentence set should compute it once and pass it here --
    otherwise every call repeats n_heads x n_sentences forward passes.
    """
    all_heads = [(int(r["layer"]), int(r["head"])) for r in rows]
    # 1. Predicted-importance scores
    scores = compute_predictor(predictor_name, rows, extra=extra, seed=seed)
    # 2. Per-head ablation effect
    if cached_effects is not None:
        single_effects = cached_effects
    else:
        orig_runs = run_model(model, model_key, sentences, tokenizer,
                               device=device)
        single_effects = _per_head_validation(
            model, model_key, tokenizer, sentences, all_heads, orig_runs,
            device=device, ablation_mode=ablation_mode,
        )
    cosine = np.array([e.cosine_drop for e in single_effects])
    kl = np.array([e.kl_div_next_token for e in single_effects])
    drift = np.array([e.attention_drift for e in single_effects])

    # Handle NaN KL (encoder-only models): fall back to cosine drop
    surrogate = np.nan_to_num(kl, nan=cosine)

    per_head: List[HeadValidation] = []
    for i, (L, H) in enumerate(all_heads):
        per_head.append(HeadValidation(
            layer=L, head=H,
            predicted_importance=float(scores[i]),
            predicted_unimportance=float(-scores[i]),
            cosine_drop=float(cosine[i]) if not np.isnan(cosine[i]) else 0.0,
            kl_div_next_token=float(kl[i]) if not np.isnan(kl[i]) else 0.0,
            attention_drift=float(drift[i]),
        ))
    # Correlate lower-importance prediction (ranked highest to be pruned) with
    # higher accuracy-loss empirically observed
    pred_unimp = -scores
    r, rho, tau = _rank_corrs(pred_unimp, surrogate)

    return PredictorValidationReport(
        predictor=predictor_name,
        pearson_r=r, spearman_rho=rho, kendall_tau=tau,
        per_head=per_head,
        aggregate_kl=float(np.mean(np.nan_to_num(kl))),
        aggregate_cosine=float(np.mean(np.nan_to_num(cosine))),
        surrogate_for_baseline_difference=0.0,
    )


# --------------------------------------------------------------------------- #
# Batched ranked ablation across prediction quantiles
# --------------------------------------------------------------------------- #

@dataclass
class AggregateAblationEffect:
    predictor: str
    ratio: float
    cosine_drop: float
    kl_div: float
    attention_drift: float


def ranked_aggregate_pruning(
    model,
    model_key: str,
    tokenizer,
    sentences: List[str],
    rows: List[Dict[str, float]],
    extra: Dict[str, np.ndarray],
    predictor_name: str,
    ratios: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50),
    device: Optional[str] = None,
    seed: int = 0,
    ablation_mode: str = "zero",
) -> List[AggregateAblationEffect]:
    """Rank heads by predicted importance (low importance = pruned first),
    ablate the bottom ``ratio`` fraction for each ``ratio`` and measure the
    aggregate effect.
    """
    all_heads = [(int(r["layer"]), int(r["head"])) for r in rows]
    scores = compute_predictor(predictor_name, rows, extra=extra, seed=seed)
    orig_runs = run_model(model, model_key, sentences, tokenizer,
                           device=device)
    out: List[AggregateAblationEffect] = []
    ab = HeadAblator(model, model_key, mode=ablation_mode)
    for ratio in ratios:
        n_pruned = max(1, int(np.floor(ratio * len(scores))))
        # Lowest importance scores = candidates to prune
        order = np.argsort(scores)             # ascending
        pruned = [all_heads[i] for i in order[:n_pruned]]
        ab.ablate = set(pruned)
        hm = ab.head_mask(device=device)
        runs = run_model(model, model_key, sentences, tokenizer,
                          device=device, head_mask=hm)
        eff = measure_effect(orig_runs, runs, n_ablated=n_pruned)
        out.append(AggregateAblationEffect(
            predictor=predictor_name, ratio=float(ratio),
            cosine_drop=eff.cosine_drop,
            kl_div=eff.kl_div_next_token,
            attention_drift=eff.attention_drift,
        ))
    return out
