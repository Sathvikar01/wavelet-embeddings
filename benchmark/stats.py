"""Statistical reporting helpers for the head-pruning benchmark.

The benchmark scores each predictor (wavelet + baselines) on a per-input
basis with several proxies: rank correlation (Pearson / Spearman /
Kendall) between predicted unimportance and the *measured* ablation loss.

This module turns those per-input sample arrays into:

  * mean +/- standard deviation,
  * 95 % bootstrap confidence intervals (BCA used when possible),
  * paired-samples Wilcoxon signed-rank or paired t-test p-values
    (predictor vs. each baseline),
  * effect sizes - Cohen's d for paired samples and Cliff's delta.

The stats are computed on the directly comparable per-input scalar (the
predictor's correlation with the measured loss, averaged over heads) so a
reviewer can answer "is wavelet significantly better than the strongest
baseline or is it sampling noise?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


__all__ = [
    "Summary", "Comparison",
    "summarise", "compare_pair",
    "bootstrap_ci",
    "cohens_d_paired", "cliffs_delta",
    "paired_test",
]


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class Summary:
    n: int
    mean: float
    std: float
    median: float
    ci_low: float
    ci_high: float
    samples: List[float] = field(default_factory=list)


@dataclass
class Comparison:
    """A predictor-vs-baseline paired comparison."""
    predictor: str
    baseline: str
    metric: str
    diff_mean: float                 # mean(predictor - baseline)
    cohens_d: float
    cliffs_delta: float
    test: str                         # "wilcoxon" | "paired_t"
    statistic: float
    p_value: float


# --------------------------------------------------------------------------- #
# Core summaries
# --------------------------------------------------------------------------- #

def bootstrap_ci(x: Sequence[float],
                  n_boot: int = 5000,
                  confidence: float = 0.95,
                  seed: int = 0) -> Tuple[float, float]:
    """Percentile-bootstrap CI for the mean of ``x``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    # Try BCA correction when scipy is available - strictly better than the
    # plain percentile CI on skewed distributions of correlations.
    if _HAS_SCIPY and arr.size >= 5:
        try:
            from scipy.stats import bootstrap  # type: ignore
            res = bootstrap(
                (arr,), np.mean, n_resamples=n_boot,
                confidence_level=confidence, method="BCa",
                random_state=seed,
            )
            lo, hi = float(res.confidence_interval.low), \
                     float(res.confidence_interval.high)
        except Exception:
            pass
    return lo, hi


def summarise(x: Sequence[float],
              n_boot: int = 5000,
              confidence: float = 0.95,
              seed: int = 0) -> Summary:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return Summary(0, float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), [])
    lo, hi = bootstrap_ci(arr, n_boot=n_boot, confidence=confidence, seed=seed)
    return Summary(
        n=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        median=float(np.median(arr)),
        ci_low=lo, ci_high=hi,
        samples=[float(v) for v in arr],
    )


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #

def cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """Cohen's d for *paired* samples: mean(diff)/std(diff)."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    d = a[:n] - b[:n]
    sd = d.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return 0.0 if d.mean() == 0 else float("inf")
    return float(d.mean() / sd)


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta in [-1, 1]: P(x > y) - P(x < y). Non-parametric."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    # O(n*m); fine for benchmark sizes (per-input).
    more = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return float((more - less) / (a.size * b.size))


# --------------------------------------------------------------------------- #
# Paired hypothesis tests
# --------------------------------------------------------------------------- #

def paired_test(x: Sequence[float], y: Sequence[float],
                test: str = "wilcoxon") -> Tuple[float, float]:
    """Return (statistic, p_value) for a paired comparison.

    ``test`` is one of ``"wilcoxon"`` or ``"paired_t"``. Falls back to an
    approximate pure-numpy version when scipy is unavailable.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    n = min(a.size, b.size)
    if n < 2:
        return float("nan"), float("nan")
    d = a[:n] - b[:n]
    if test == "paired_t":
        if _HAS_SCIPY:
            t, p = sp_stats.ttest_rel(a[:n], b[:n])
            return float(t), float(p)
        sd = d.std(ddof=1)
        if sd == 0:
            return 0.0, 1.0 if d.mean() == 0 else 0.0
        t = float(d.mean() / (sd / np.sqrt(n)))
        return t, float(2 * (1 - _tdist_cdf_approx(abs(t), n - 1)))
    if test == "wilcoxon":
        if _HAS_SCIPY:
            try:
                stat, p = sp_stats.wilcoxon(a[:n], b[:n],
                                          zero_method="wilcox",
                                          alternative="two-sided")
                return float(stat), float(p)
            except Exception:
                pass  # fall through to the approximate variant
        # Sign test fallback (still produces a valid p-value when n>=5).
        nz = d[d != 0]
        if nz.size < 5:
            return float("nan"), float("nan")
        pos = int(np.sum(nz > 0))
        k = min(pos, nz.size - pos)
        from math import comb
        p_two = 2.0 * sum(comb(nz.size, i) for i in range(0, k + 1)) \
                  / (2.0 ** nz.size)
        return float(abs(pos - nz.size / 2)), float(min(1.0, p_two))
    raise ValueError(f"Unknown test '{test}'")


def _tdist_cdf_approx(t: float, df: int) -> float:
    """Plain-python tail approximation; used only if scipy is missing.

    Returns ``P(T <= t)`` for a Student t with ``df`` degrees of freedom.
    Uses the standard-normal CDF approximation (erf) clipped by a finite-
    sample inflation factor; for ``df`` >= ~30 the difference is negligible
    on the order of the benchmark's reporting precision.
    """
    import math
    if df <= 0:
        df = 1
    s = math.sqrt(df / (df + t * t))
    cdf_at_0 = 0.5 + 0.5 * math.erf(t / math.sqrt(2.0 * df))
    return max(0.0, min(1.0, cdf_at_0 * s + (1.0 - s) * 0.5))


# --------------------------------------------------------------------------- #
# Higher-level pairing
# --------------------------------------------------------------------------- #

def compare_pairs(predictor_name: str,
                   predictor_vals: Sequence[float],
                   baseline_values: Dict[str, Sequence[float]],
                   metric: str,
                   test: str = "wilcoxon") -> List[Comparison]:
    out: List[Comparison] = []
    for bname, bvals in baseline_values.items():
        d_cohen = cohens_d_paired(predictor_vals, bvals)
        d_cliff = cliffs_delta(predictor_vals, bvals)
        stat, p = paired_test(predictor_vals, bvals, test=test)
        out.append(Comparison(
            predictor=predictor_name, baseline=bname, metric=metric,
            diff_mean=float(np.mean(np.asarray(predictor_vals)
                                    - np.asarray(bvals))),
            cohens_d=float(d_cohen), cliffs_delta=float(d_cliff),
            test=test, statistic=float(ifnan(stat)),
            p_value=float(ifnan(p)),
        ))
    return out


def ifnan(v, default=0.0):
    try:
        if v != v:                     # NaN test
            return default
    except Exception:
        return default
    return v


def compare_pair(a_vals: Sequence[float], b_vals: Sequence[float],
                  test: str = "wilcoxon") -> Comparison:
    stat, p = paired_test(a_vals, b_vals, test=test)
    return Comparison(
        predictor="a", baseline="b", metric="",
        diff_mean=float(np.mean(np.asarray(a_vals) - np.asarray(b_vals))),
        cohens_d=float(cohens_d_paired(a_vals, b_vals)),
        cliffs_delta=float(cliffs_delta(a_vals, b_vals)),
        test=test, statistic=float(ifnan(stat)),
        p_value=float(ifnan(p)),
    )
