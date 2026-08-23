"""Reproducible head-pruning benchmark runner.

Per (model, dataset, seed) cell:

  1. load ``max_sentences`` cleaned sentences,
  2. run a forward pass with ``output_attentions=True`` and snapshot every
     (layer, head) attention matrix,
  3. compute Phase-3 wavelet metrics for each head,
  4. score every predictor:

       * wavelet               -- existing Phase-4 composite
       * attention_entropy, attention_weight, magnitude, random
                                  -- existing simple baselines
       * michel_hic, voita_his, bhasharas_bs
                                  -- published baselines (this package)

  5. ablate every head individually with the HF ``head_mask`` and record
     the cosine_drop / KL / attention_drift (per input),
  6. correlate predicted-unimportance (-score) with the observed loss to
     produce per-input Pearson / Spearman / Kendall,
  7. collect those per-input scores across seeds into ``BenchResult``.

After the sweep the reporter (``benchmark.report``) aggregates:

  * mean +/- SD across inputs,
  * 95 % bootstrap CIs,
  * paired Wilcoxon and paired-t p-values vs. each baseline,
  * Cohen's d and Cliff's delta effect sizes,
  * per-seed variance flagged as a separate row.

Outputs land under ``--out results/benchmark/`` as CSV + a JSON bundle.
"""

from __future__ import annotations

import json
import os
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from embeddings.extract import MODEL_REGISTRY, EmbeddingExtractor
from attention import (
    AttentionWaveletDecomposer, compute_head_metrics,
)
from attention.extractor import AttentionOutput
from pruning.runner import HeadAblator
from pruning.registry import PREDICTOR_NAMES, compute_predictor
from evaluation.task_loss import run_model, measure_effect
from benchmark.baselines_published import (
    PUBLISHED_BASELINES, compute_published_baseline,
)
from benchmark.datasets import load_sentences, DATASET_NAMES


__all__ = [
    "ALL_PREDICTORS", "BenchRecord", "BenchResult",
    "run_cell", "run_benchmark", "write_results",
    "apply_ridge_looo",
]


# Order: wavelet first, simple baselines, then published baselines, then
# Phase-6 ridge predictors. The ridge predictors are populated by a
# post-pass (``benchmark.ridge_looo.apply_ridge_looo``) after every cell
# in the sweep has finished scoring the cold-start predictors; they
# cannot be scored per-input inside ``run_cell`` because their weights
# are fit leave-one-cell-out. We register them here so the per-cell
# CSV / JSON layout has the columns from the start (filled with NaN / 0
# before the post-pass runs).
from benchmark.ridge_looo import RIDGE_PREDICTORS, apply_ridge_looo  # noqa: E402


# Order: wavelet first, simple baselines, then published baselines, then
# Phase-6 ridge LOOO predictors.
ALL_PREDICTORS = ["wavelet"] + [p for p in PREDICTOR_NAMES if p != "wavelet"] \
                  + PUBLISHED_BASELINES + list(RIDGE_PREDICTORS)


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class BenchRecord:
    """Per-input scores for every predictor (correlation with ablation loss)
    plus the raw ablation effect per head.

    Phase-6 extension: alongside Pearson (the historical scalar), also
    captures Spearman, Kendall, mean cosine-preservation (= ``1 -
    cosine_drop`` per head averaged over heads of the input) and the
    mean per-head pruning loss per input, all indexed the same way as
    ``per_input``. Captured only when ``full_metrics=True`` in
    ``run_cell`` -- default in Phase 6, opt-in for back-compat so the
    Phase-5 reference run is bit-for-bit reproducible. The
    per-input-correlation writers in ``write_results`` peacefully skip
    predictors lacking these extras.
    """
    model: str
    dataset: str
    seed: int
    n_heads: int
    metric_used: str                    # which loss the correlation used
    per_input: Dict[str, List[float]] = field(default_factory=dict)
    # Per-head arrays length n_heads for the first input (kept for offline
    # diagnostic plots / debug).
    per_head_loss: List[float] = field(default_factory=list)
    per_head_score: Dict[str, List[float]] = field(default_factory=dict)
    # Phase-6 LOOO ridge support: per-input raw feature matrices and
    # measured per-head loss. Populated by ``run_cell`` only when
    # ``keep_per_input_features=True`` (set by the Modal driver for the
    # combined-predictor post-pass). Each list entry is one input.
    per_input_features: List[np.ndarray] = field(default_factory=list)
    per_input_loss: List[np.ndarray] = field(default_factory=list)
    # Phase-6 full-metric capture: per-input Spearman / Kendall correlation
    # of predicted-unimportance with the measured loss. Same keying as
    # ``per_input``. Populated in ``run_cell`` only when ``full_metrics=True``.
    per_input_spearman: Dict[str, List[float]] = \
        field(default_factory=dict)
    per_input_kendall: Dict[str, List[float]] = \
        field(default_factory=dict)
    # Per-input scalar summary metrics (per predictor). Populated in
    # ``run_cell`` (and ``apply_ridge_looo`` for the three Phase-6
    # predictors). Lists length n_inputs aligned with per_input[P].
    per_input_cosine_pres: Dict[str, List[float]] = \
        field(default_factory=dict)
    per_input_pruning_loss: Dict[str, List[float]] = \
        field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = {
            "model": self.model, "dataset": self.dataset, "seed": self.seed,
            "n_heads": self.n_heads, "metric_used": self.metric_used,
            "per_input": {k: list(v) for k, v in self.per_input.items()},
        }
        # Per-head feature / loss arrays are not JSON-serialisable; the
        # .npz offline artefact is written by ``benchmark.feature_matrix``
        # consumers (see ``apply_ridge_looo``).
        if self.per_input_spearman:
            d["per_input_spearman"] = {k: list(v) for k, v
                                         in self.per_input_spearman.items()}
        if self.per_input_kendall:
            d["per_input_kendall"] = {k: list(v) for k, v
                                          in self.per_input_kendall.items()}
        if self.per_input_cosine_pres:
            d["per_input_cosine_pres"] = {k: list(v) for k, v in
                                            self.per_input_cosine_pres.items()}
        if self.per_input_pruning_loss:
            d["per_input_pruning_loss"] = {k: list(v) for k, v in
                                          self.per_input_pruning_loss.items()}
        return d


@dataclass
class BenchResult:
    records: List[BenchRecord] = field(default_factory=list)
    # Phase-6 LOOO ridge per-fold learned coefficients (populated by
    # ``apply_ridge_looo`` via ``run_benchmark`` when
    # ``enable_phase6_ridge=True``). ``write_results`` reads this to
    # emit ``feature_importance.csv``. Default-None keeps the dataclass
    # backward-compatible with code that constructs ``BenchResult()``.
    ridge_folds_coeffs: Optional[Dict] = None

    def to_jsonable(self) -> List[Dict]:
        return [r.to_dict() for r in self.records]


# --------------------------------------------------------------------------- #
# Per-sentence head scoring
# --------------------------------------------------------------------------- #

def _extract_attention(ext: EmbeddingExtractor, sentence: str, max_len: int
                       ) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """Forward-pass for one sentence and produce a per-head (layer, head)
    attention list plus the corresponding wavelet metrics list."""
    import torch
    tokenizer = ext.tokenizer
    model = ext.model
    model.eval()
    enc = tokenizer(sentence, return_tensors="pt", truncation=True,
                    max_length=max_len).to(ext.device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    attn = tuple(a.squeeze(0).cpu().numpy().astype(np.float32)
                  for a in out.attentions)   # L x (H, T, T)
    head_list: List[np.ndarray] = []
    rows: List[Dict[str, float]] = []
    decomposer = AttentionWaveletDecomposer("db4")
    for L, layer_attn in enumerate(attn):
        for H in range(layer_attn.shape[0]):
            A = layer_attn[H]
            row = compute_head_metrics(A, decomposer)
            row["layer"] = L
            row["head"] = H
            head_list.append(A)
            rows.append(row)
    return head_list, rows


def _per_head_effects(ext: EmbeddingExtractor, model_key: str, sentences:
                       List[str], all_heads: List[Tuple[int, int]],
                       orig_runs, device) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray]:
    """Ablate every head individually and return per-head cosine_drop,
    kl_div_next_token and attention_drift arrays.

    Uses the architecture-agnostic ``run_model_ablated`` (forward hooks on
    the post-attention Linear) so it works on families whose ``forward``
    drops ``head_mask`` (Llama, DeBERTa). Canonical HF ``head_mask`` is still
    wired through :class:`HeadAblator` for backwards compatibility but is no
    longer used in the hot path here.
    """
    from evaluation.task_loss import run_model_ablated
    model = ext.model
    tokenizer = ext.tokenizer
    cos, kl, drift = [], [], []
    for (L, H) in all_heads:
        runs = run_model_ablated(model, model_key, sentences[:1],
                                  tokenizer, ablate=[(L, H)], device=device)
        eff = measure_effect(orig_runs[:1], runs, n_ablated=1)
        cos.append(eff.cosine_drop)
        kl.append(eff.kl_div_next_token)
        drift.append(eff.attention_drift)
    return (np.asarray(cos, dtype=np.float64),
             np.asarray(kl, dtype=np.float64),
             np.asarray(drift, dtype=np.float64))


# --------------------------------------------------------------------------- #
# Cell execution
# --------------------------------------------------------------------------- #

def _rank_corrs(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan"), float("nan")
    try:
        from scipy.stats import pearsonr, spearmanr, kendalltau
        r, _ = pearsonr(x, y)
        rho, _ = spearmanr(x, y)
        tau, _ = kendalltau(x, y)
        return float(r), float(rho), float(tau)
    except Exception:
        xm = x - x.mean(); ym = y - y.mean()
        denom = (np.linalg.norm(xm) * np.linalg.norm(ym) + 1e-12)
        r = float((xm * ym).sum() / denom)
        rho = float(np.corrcoef(np.argsort(np.argsort(x)),
                                  np.argsort(np.argsort(y)))[0, 1])
        # O(n^2) kendall
        n = x.size; con = 0; dis = 0
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        for i in range(n):
            for j in range(i + 1, n):
                d = (rx[j] - rx[i]) * (ry[j] - ry[i])
                if d > 0: con += 1
                elif d < 0: dis += 1
        t = con + dis if (con + dis) else 1
        return r, rho, (con - dis) / t


def _score_all(rows, head_attn):
    """Run wavelet + simple + published predictors, returning a dict of
    arrays (length n_heads)."""
    scores: Dict[str, np.ndarray] = {}
    extra = {"head_attention": head_attn}
    for name in PREDICTOR_NAMES:
        scores[name] = compute_predictor(name, rows, extra=extra, seed=0)
    for name in PUBLISHED_BASELINES:
        scores[name] = compute_published_baseline(name, rows, extra=extra)
    # Uniform random - baseline reused across calls / no seed dependence for
    # the *correlation* stage: it is purely a noise floor.
    return scores


def run_cell(model_key: str, ds_name: str, seed: int,
               max_sentences: int = 2000, max_len: int = 64,
               cache_dir: Optional[str] = None, metric: str = "cosine_drop",
               device: Optional[str] = None, verbose: bool = False,
               keep_per_input_features: bool = False,
               features_out_dir: Optional[str] = None,
               full_metrics: bool = True) -> BenchRecord:
    """Process a single (model, dataset, seed) cell.

    Used by the Modal app (``run_bench.map``) to parallelise the sweep across
    A10G containers -- one cell per container. Writes its
    ``BenchRecord.per_input`` into the global ``BenchResult`` returned by
    :func:`run_benchmark` (serial) or aggregated by the driver
    (Modal-parallel).

    ``keep_per_input_features``: when True, also stash per-input raw
    per-head wavelet-feature matrices and measured per-head ablation loss
    into the record (``per_input_features`` / ``per_input_loss``). The
    Phase-6 LOOO ridge post-pass consumes these to populate the
    Phase-6 predictors. Adds negligible memory overhead (n_heads ~144,
    ~10 features) but is opt-in to keep non-Phase-6 cells byte-equal.

    ``features_out_dir``: when ``keep_per_input_features`` is also True,
    persists the cell's standardised per-input feature matrix + measured
    loss as ``<features_out_dir>/cell_<model>_<dataset>_<seed>.npz``.
    Pure reproducibility artefact -- not consumed by anything else, but
    required by the Phase-6 spec so reviewers can re-fit the ridge
    offline and reproduce the published ``feature_importance.csv``.

    ``full_metrics``: when True (default), additionally capture per-input
    Spearman, Kendall, mean cosine-preservation and mean pruning loss per
    predictor on the BenchRecord. Phase-5 back-compat note: this only
    adds new dict fields; the existing ``per_input`` key/values are
    unchanged, so the Phase-5 / Phase-6 reference `summary.json` is
    byte-identical for the Pearson-only comparisons.
    """
    spec = MODEL_REGISTRY[model_key]
    if verbose:
        print(f"[benchmark] loading model {model_key} ({spec.label})")
    ext = EmbeddingExtractor(model_key, cache_dir=cache_dir, device=device)
    ext.load()
    dev = device or ext.device
    sentences = load_sentences(ds_name, max_sentences=max_sentences)
    if verbose:
        print(f"  dataset={ds_name} sentences={len(sentences)}")
    rng = np.random.default_rng(seed)
    if max_sentences and len(sentences) > max_sentences:
        # Per-seed random subsample without replacement -- different
        # seeds take genuinely disjoint 150-sentence slices so the
        # seed-variance report measures real sampling variance and not
        # a deterministic-stride artefact.
        idx = rng.choice(len(sentences), size=max_sentences,
                          replace=False)
        idx.sort()
        sel = [sentences[i] for i in idx]
    else:
        sel = sentences
    rec = BenchRecord(model=model_key, dataset=ds_name, seed=seed,
                       n_heads=0, metric_used=metric)
    for pname in ALL_PREDICTORS:
        rec.per_input[pname] = []
    for snum, sent in enumerate(sel):
        try:
            head_attn, rows = _extract_attention(ext, sent, max_len)
        except Exception as e:
            if verbose:
                print(f"    skip sentence {snum}: {e}")
            continue
        all_heads = [(int(r["layer"]), int(r["head"])) for r in rows]
        rec.n_heads = len(all_heads)
        scores = _score_all(rows, head_attn)
        orig_runs = run_model(ext.model, model_key, [sent],
                              ext.tokenizer, device=dev)
        cos, kl, drift = _per_head_effects(
            ext, model_key, [sent], all_heads, orig_runs, dev
        )
        if metric in ("kl_div_next_token",):
            loss = kl
            if np.all(np.isnan(loss)) or loss.size == 0:
                loss = cos
        elif metric == "attention_drift":
            loss = drift
        else:
            loss = cos
        loss = np.nan_to_num(loss, nan=0.0)
        rec.per_head_loss = [float(v) for v in loss]
        # Per-input scalar summary metrics (predictor-independent):
        # cosine_pres = 1 - mean(cosine_drop). measures representation
        # preservation under ablation; high = preserved. pruning_loss
        # here = the per-input mean of the chosen ablation-loss metric
        # (the same value reported per-head above), so a reviewer can
        # read the magnitude bar without recombining per-head arrays.
        cos_pres_input = float(np.mean(1.0 - loss)) if loss.size else 0.0
        loss_mean_input = float(np.mean(loss)) if loss.size else 0.0
        if keep_per_input_features:
            # Build the combined-group feature matrix for this input.
            # The other groups (wavelet_only, attn_only) are sub-matrices
            # of this one; the ridge post-pass slices on column name.
            from benchmark.feature_matrix import build_feature_matrix
            X, _ = build_feature_matrix(
                rows, {"head_attention": head_attn}, group="combined")
            rec.per_input_features.append(X)
            rec.per_input_loss.append(np.asarray(loss, dtype=np.float64))
        for pname, arr in scores.items():
            rec.per_head_score.setdefault(
                pname, [float(v) for v in arr])
            pred_unimp = -arr.astype(np.float64)
            r, rho, tau = _rank_corrs(pred_unimp, loss)
            rec.per_input[pname].append(
                float(r if np.isfinite(r) else 0.0))
            if full_metrics:
                rec.per_input_spearman.setdefault(pname, []).append(
                    float(rho if np.isfinite(rho) else 0.0))
                rec.per_input_kendall.setdefault(pname, []).append(
                    float(tau if np.isfinite(tau) else 0.0))
                # Same input-level metric for every predictor (it is the
                # input's measured representation-preservation / loss,
                # independent of who predicts). Emitted per-predictor so
                # ``predictor_metrics.csv`` joins cleanly on
                # (model, dataset, seed, predictor).
                rec.per_input_cosine_pres.setdefault(pname, []).append(
                    cos_pres_input)
                rec.per_input_pruning_loss.setdefault(pname, []).append(
                    loss_mean_input)
        if verbose and (snum + 1) % 100 == 0:
            print(f"    {snum + 1}/{len(sel)} sentences done")

    # Phase-6 reproducibility artefact: persist the within-cell
    # standardised per-input feature matrix + measured losses to a
    # .npz file when requested. Mirrors the Modal cells/ path layout
    # so lazy reviews can re-run the LOOO ridge offline and verify
    # ``feature_importance.csv`` independently.
    if keep_per_input_features and features_out_dir and rec.per_input_features:
        try:
            os.makedirs(features_out_dir, exist_ok=True)
            out_path = os.path.join(
                features_out_dir,
                f"cell_{model_key.replace('/', '_')}_{ds_name}_{seed}.npz")
            Xin = np.stack(rec.per_input_features, axis=0)
            yin = np.stack(rec.per_input_loss, axis=0)
            np.savez_compressed(out_path, X=Xin, y=yin,
                                  model=np.array([rec.model]),
                                  dataset=np.array([rec.dataset]),
                                  seed=np.array([rec.seed]),
                                  metric_used=np.array([rec.metric_used]))
            if verbose:
                print(f"    persisted {out_path} "
                      f"(shape={Xin.shape})", flush=True)
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"    feature npz persist warning: {e}", flush=True)

    if verbose:
        print(f"  seed={seed} model={model_key} dataset={ds_name} "
              f"heads={rec.n_heads} "
              f"inputs={len(rec.per_input.get('wavelet', []))}")
    ext.close()
    return rec


def run_benchmark(models: List[str], datasets: List[str],
                   seeds: Sequence[int],
                   max_sentences: int = 2000,
                   max_len: int = 64,
                   cache_dir: Optional[str] = None,
                   metric: str = "cosine_drop",
                   device: Optional[str] = None,
                   verbose: bool = True,
                   enable_phase6_ridge: bool = True,
                   ridge_alpha: float = 1.0,
                   features_out_dir: Optional[str] = None,
                   full_metrics: bool = True) -> BenchResult:
    """Run the full benchmark sweep and return per-input correlations.

    Serial implementation.  For parallel execution across Modal GPUs use
    :func:`run_cell` directly via the platform's `.starmap` mechanism (see
    ``benchmark/modal_app.py``).

    ``enable_phase6_ridge``: when True, after every cell has been scored,
    run the leave-one-cell-out ridge post-pass to populate the three
    Phase-6 predictors (``wavelet_only_model``, ``attention_only_model``,
    ``wavelet_feature_model``). Cells are scored with
    ``keep_per_input_features=True`` so their per-input feature matrices
    + measured losses are available for the LOOO loop. Disable only when
    reproducing the Phase-5 reference run bit-for-bit.

    ``features_out_dir``: passed to ``run_cell`` so each cell deposits its
    standardised per-input feature matrix + measured loss as a
    ``cell_<model>_<dataset>_<seed>.npz`` next to the per-cell JSON.
    Required by the Phase-6 spec's "save all intermediate feature matrices
    for reproducibility" clause.

    ``full_metrics``: passed straight through to ``run_cell``; the spec
    demands Spearman + Kendall + cosine preservation + mean pruning loss
    per predictor per cell so this defaults True.
    """
    result = BenchResult()
    for model_key in models:
        for ds_name in datasets:
            for seed in seeds:
                rec = run_cell(model_key, ds_name, seed,
                                max_sentences=max_sentences,
                                max_len=max_len, cache_dir=cache_dir,
                                metric=metric, device=device, verbose=verbose,
                                keep_per_input_features=enable_phase6_ridge,
                                features_out_dir=features_out_dir,
                                full_metrics=full_metrics)
                if rec.n_heads:
                    result.records.append(rec)
    if enable_phase6_ridge:
        coeffs_art = apply_ridge_looo(result, alpha=ridge_alpha,
                                       verbose=verbose,
                                       return_coeffs=True)
        # Stash the per-fold learned coefficients on the BenchResult so
        # ``write_results`` can emit ``feature_importance.csv``.
        result.ridge_folds_coeffs = coeffs_art
    return result


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #

def write_results(result: BenchResult, out_dir: str,
                   test: str = "wilcoxon") -> Dict:
    """Persist per-records CSV + aggregated stats JSON; returns the aggregate
    document.

    Accepts either a :class:`BenchResult` (serial path) or a list of plain
    dicts (the JSON-serialisable form used by the Modal parallel driver).
    """
    from benchmark.stats import summarise, compare_pair

    os.makedirs(out_dir, exist_ok=True)

    # Normalise to a list of record-like dicts.
    if isinstance(result, BenchResult):
        records = result.records
    else:
        records = list(result)
    record_iter = records

    def _get(r, k):
        if isinstance(r, dict):
            return r.get(k)
        return getattr(r, k)

    def rec_model(r): return _get(r, "model")
    def rec_dataset(r): return _get(r, "dataset")
    def rec_seed(r): return _get(r, "seed")
    def rec_metric(r): return _get(r, "metric_used")
    def rec_items(r, key):
        if isinstance(r, dict):
            return r.get(key, {})
        return getattr(r, key, {})

    def _safe_stdhood(xs, ddof=1):
        # Wraps np.std suppressing the ddof>0 warning when n<=1.
        arr = list(xs) if not isinstance(xs, np.ndarray) else xs
        if arr is None or len(arr) <= ddof:
            return float("nan")
        with np.errstate(all="ignore"):
            return float(np.std(arr, ddof=ddof))

    # Per-input CSV: one row per (model, dataset, seed, predictor).
    csv_path = os.path.join(out_dir, "per_input_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "seed", "predictor",
                     "metric", "n", "mean", "std", "ci_low", "ci_high"])
        aggregate: Dict[Tuple[str, str, str], Dict[str, List[float]]] = {}
        for rec in record_iter:
            for pname, vals in rec_items(rec, "per_input").items():
                if not vals:
                    continue
                agg_key = (rec_model(rec), rec_dataset(rec), rec_metric(rec))
                aggregate.setdefault(agg_key, {})[pname] = vals
                sm = summarise(vals)
                w.writerow([rec_model(rec), rec_dataset(rec),
                              rec_seed(rec), pname,
                              rec_metric(rec), sm.n, f"{sm.mean:.4f}",
                              f"{sm.std:.4f}", f"{sm.ci_low:.4f}",
                              f"{sm.ci_high:.4f}"])

    # Aggregated cross-seed CSV: collapse seeds for a model+dataset.
    agg_path = os.path.join(out_dir, "aggregate.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "metric", "predictor",
                     "seeds", "n", "mean", "std",
                     "ci_low", "ci_high"])
        for (model, ds, metric_used), by_pred in aggregate.items():
            for pname, vals in by_pred.items():
                sm = summarise(vals)
                seeds = sorted({rec_seed(r) for r in record_iter
                                  if rec_model(r) == model
                                  and rec_dataset(r) == ds})
                w.writerow([model, ds, metric_used, pname,
                              ";".join(map(str, seeds)), sm.n,
                              f"{sm.mean:.4f}", f"{sm.std:.4f}",
                              f"{sm.ci_low:.4f}", f"{sm.ci_high:.4f}"])

    # Comparison document (wavelet vs every baseline).
    comparisons: List[Dict] = []
    for (model, ds, metric_used), by_pred in aggregate.items():
        wval = by_pred.get("wavelet")
        if not wval:
            continue
        for bname, bvals in by_pred.items():
            if bname == "wavelet" or len(bvals) < 3:
                continue
            comp = compare_pair(wval, bvals, test=test)
            comparisons.append({
                "model": model, "dataset": ds, "metric": metric_used,
                "predictor": "wavelet", "baseline": bname,
                "diff_mean": round(comp.diff_mean, 4),
                "cohens_d": round(comp.cohens_d, 4),
                "cliffs_delta": round(comp.cliffs_delta, 4),
                "test": comp.test,
                "statistic": round(comp.statistic, 4),
                "p_value": round(comp.p_value, 6),
            })

    # Phase-6 additional comparisons: wavelet_entropy vs everything
    # else (including the Phase-6 ridge predictors and
    # attention_entropy_true). This block answers the scientifically-
    # loaded question "does the combined (wavelet + attention-entropy)
    # ridge regression lift predictive power over the strongest single
    # feature (wavelet-entropy)?" -- on a per-cell statistical basis.
    for (model, ds, metric_used), by_pred in aggregate.items():
        we_val = by_pred.get("wavelet_entropy")
        if not we_val:
            continue
        for bname, bvals in by_pred.items():
            if bname in ("wavelet", "wavelet_entropy") or len(bvals) < 3:
                continue
            comp = compare_pair(we_val, bvals, test=test)
            comparisons.append({
                "model": model, "dataset": ds, "metric": metric_used,
                "predictor": "wavelet_entropy", "baseline": bname,
                "diff_mean": round(comp.diff_mean, 4),
                "cohens_d": round(comp.cohens_d, 4),
                "cliffs_delta": round(comp.cliffs_delta, 4),
                "test": comp.test,
                "statistic": round(comp.statistic, 4),
                "p_value": round(comp.p_value, 6),
            })

    # Phase-6 headline comparison: wavelet_feature_model vs EVERY other
    # predictor (including the ablation models wavelet_only_model and
    # attention_only_model). This is the comparison block reviewers
    # will read to decide whether the combined ridge lifted predictive
    # power over the A / B ablation baselines per cell.
    for (model, ds, metric_used), by_pred in aggregate.items():
        wfm_val = by_pred.get("wavelet_feature_model")
        if not wfm_val:
            continue
        for bname, bvals in by_pred.items():
            if bname == "wavelet_feature_model" or len(bvals) < 3:
                continue
            comp = compare_pair(wfm_val, bvals, test=test)
            comparisons.append({
                "model": model, "dataset": ds, "metric": metric_used,
                "predictor": "wavelet_feature_model", "baseline": bname,
                "diff_mean": round(comp.diff_mean, 4),
                "cohens_d": round(comp.cohens_d, 4),
                "cliffs_delta": round(comp.cliffs_delta, 4),
                "test": comp.test,
                "statistic": round(comp.statistic, 4),
                "p_value": round(comp.p_value, 6),
            })

    # Variance-of-seeds report: per (model, dataset, predictor) report
    # std of seed-mean. Smaller => robust across seeds.
    var_path = os.path.join(out_dir, "seed_variance.csv")
    seed_means: Dict[Tuple[str, str, str, str], List[float]] = {}
    for rec in record_iter:
        for pname, vals in rec_items(rec, "per_input").items():
            if not vals:
                continue
            key = (rec_model(rec), rec_dataset(rec), rec_metric(rec), pname)
            seed_means.setdefault(key, []).append(float(np.mean(vals)))
    with open(var_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "metric", "predictor",
                     "n_seeds", "seed_mean_of_means", "seed_std"])
        for key, mns in seed_means.items():
            model, ds, metric_used, pname = key
            w.writerow([model, ds, metric_used, pname,
                          len(mns), f"{np.mean(mns):.4f}",
                          f"{_safe_stdhood(mns, ddof=1):.4f}"])

    # Phase-6 predictor_metrics.csv: one row per (model, dataset, seed,
    # predictor) carrying all the spec-required per-predictor summary
    # metrics (Pearson, Spearman, Kendall, cosine preservation, pruning
    # loss) each with mean, std, 95% bootstrap CI. Predictors that did
    # not capture Spearman/Kendall (e.g. barren Phase-5 cells with
    # full_metrics=False) emit NaN rows so the file shape is stable.
    pm_path = os.path.join(out_dir, "predictor_metrics.csv")
    with open(pm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "dataset", "seed", "predictor",
            "n_inputs",
            "pearson_mean", "pearson_std", "pearson_ci_low",
            "pearson_ci_high",
            "spearman_mean", "spearman_std", "spearman_ci_low",
            "spearman_ci_high",
            "kendall_mean", "kendall_std", "kendall_ci_low",
            "kendall_ci_high",
            "cosine_preservation_mean", "cosine_preservation_std",
            "pruning_loss_mean", "pruning_loss_std",
            "pruning_loss_ci_low", "pruning_loss_ci_high",
        ])
        for rec in record_iter:
            model = rec_model(rec)
            ds = rec_dataset(rec)
            seed = rec_seed(rec)
            metric_used = rec_metric(rec)
            per_input_dict = rec_items(rec, "per_input")
            spe_dict = rec_items(rec, "per_input_spearman")
            tau_dict = rec_items(rec, "per_input_kendall")
            pres_dict = rec_items(rec, "per_input_cosine_pres")
            loss_dict = rec_items(rec, "per_input_pruning_loss")
            for pname, pearson_vals in per_input_dict.items():
                if not pearson_vals:
                    continue
                pe_sm = summarise(pearson_vals)
                spe_vals = spe_dict.get(pname, []) if isinstance(
                    spe_dict, dict) else spe_dict
                tau_vals = tau_dict.get(pname, []) if isinstance(
                    tau_dict, dict) else tau_dict
                pres_vals = pres_dict.get(pname, []) if isinstance(
                    pres_dict, dict) else pres_dict
                loss_vals = loss_dict.get(pname, []) if isinstance(
                    loss_dict, dict) else loss_dict
                spe_sm = summarise(spe_vals) if spe_vals else None
                tau_sm = summarise(tau_vals) if tau_vals else None
                rows_for_cell = pearson_vals  # any of them; same length
                # If any extras were captured, they should match length.
                pres_mean = (float(np.mean(pres_vals)) if pres_vals
                              else float("nan"))
                pres_std = _safe_stdhood(pres_vals, ddof=1) if len(
                    pres_vals) > 1 else 0.0
                loss_sm = summarise(loss_vals) if loss_vals else None
                w.writerow([
                    model, ds, seed, pname, pe_sm.n,
                    f"{pe_sm.mean:.4f}", f"{pe_sm.std:.4f}",
                    f"{pe_sm.ci_low:.4f}", f"{pe_sm.ci_high:.4f}",
                    f"{spe_sm.mean:.4f}" if spe_sm else "nan",
                    f"{spe_sm.std:.4f}" if spe_sm else "nan",
                    f"{spe_sm.ci_low:.4f}" if spe_sm else "nan",
                    f"{spe_sm.ci_high:.4f}" if spe_sm else "nan",
                    f"{tau_sm.mean:.4f}" if tau_sm else "nan",
                    f"{tau_sm.std:.4f}" if tau_sm else "nan",
                    f"{tau_sm.ci_low:.4f}" if tau_sm else "nan",
                    f"{tau_sm.ci_high:.4f}" if tau_sm else "nan",
                    f"{pres_mean:.4f}", f"{pres_std:.4f}",
                    f"{loss_sm.mean:.4f}" if loss_sm else "nan",
                    f"{loss_sm.std:.4f}" if loss_sm else "nan",
                    f"{loss_sm.ci_low:.4f}" if loss_sm else "nan",
                    f"{loss_sm.ci_high:.4f}" if loss_sm else "nan",
                ])

    # Phase-6 correlation_table.csv: one row per (model, dataset,
    # predictor) summarised over seeds. Spec asks for a single
    # "predictor x metric" pivot; this is the cross-seed aggregate
    # equivalent of predictor_metrics.csv.
    ct_path = os.path.join(out_dir, "correlation_table.csv")
    ct_aggregate: Dict[Tuple[str, str, str, str], List[float]] = {}
    ct_spearman: Dict[Tuple[str, str, str, str], List[float]] = {}
    ct_kendall: Dict[Tuple[str, str, str, str], List[float]] = {}
    for rec in record_iter:
        model = rec_model(rec)
        ds = rec_dataset(rec)
        metric_used = rec_metric(rec)
        for pname, vals in rec_items(rec, "per_input").items():
            if not vals:
                continue
            k = (model, ds, metric_used, pname)
            ct_aggregate.setdefault(k, []).extend([float(v) for v in vals])
        for pname, vals in rec_items(rec, "per_input_spearman").items():
            if not vals:
                continue
            k = (model, ds, rec_metric(rec), pname)
            ct_spearman.setdefault(k, []).extend([float(v) for v in vals])
        for pname, vals in rec_items(rec, "per_input_kendall").items():
            if not vals:
                continue
            k = (model, ds, rec_metric(rec), pname)
            ct_kendall.setdefault(k, []).extend([float(v) for v in vals])
    with open(ct_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "metric", "predictor",
                     "n_seeds", "pearson_mean", "pearson_std",
                     "spearman_mean", "spearman_std",
                     "kendall_mean", "kendall_std"])
        keys = set(ct_aggregate) | set(ct_spearman) | set(ct_kendall)
        for k in sorted(keys):
            model, ds, metric_used, pname = k
            pear = ct_aggregate.get(k, [])
            spe = ct_spearman.get(k, [])
            tau = ct_kendall.get(k, [])
            n_seeds = len({rec_seed(r) for r in record_iter
                            if rec_model(r) == model
                            and rec_dataset(r) == ds})

            def _safe_std(xs):
                if len(xs) > 1:
                    with np.errstate(all="ignore"):
                        return float(np.std(xs, ddof=1))
                return float("nan")
            w.writerow([
                model, ds, metric_used, pname, n_seeds,
                f"{np.mean(pear):.4f}" if pear else "nan",
                _safe_std(pear),
                f"{np.mean(spe):.4f}" if spe else "nan",
                _safe_std(spe),
                f"{np.mean(tau):.4f}" if tau else "nan",
                _safe_std(tau),
            ])

    # Phase-6 comparisons.csv: exactly the same comparison rows as
    # summary.json's ``comparisons`` list, but as a flat CSV for
    # spreadsheet-friendly review. The existing ``comparisons``
    # block remains in ``summary.json`` for backwards compatibility
    # with Phase-5 tooling that expects to read it from there.
    cmp_path = os.path.join(out_dir, "comparisons.csv")
    with open(cmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "metric", "predictor", "baseline",
                     "diff_mean", "cohens_d", "cliffs_delta", "test",
                     "statistic", "p_value"])
        for c in comparisons:
            w.writerow([c["model"], c["dataset"], c["metric"],
                          c["predictor"], c["baseline"],
                          c["diff_mean"], c["cohens_d"],
                          c["cliffs_delta"], c["test"],
                          c["statistic"], c["p_value"]])

    # Phase-6 feature_importance.csv: per-fold learned ridge coefficients
    # + cross-fold mean / std / abs-mean rank per feature. Source: the
    # ``ridge_folds_coeffs`` artefact on ``BenchResult`` (serial path)
    # or the ``_ridge_folds_coeffs`` key on the last dict (Modal path).
    fi_path = os.path.join(out_dir, "feature_importance.csv")
    folds_coeffs = None
    if isinstance(result, BenchResult):
        folds_coeffs = getattr(result, "ridge_folds_coeffs", None)
    if folds_coeffs is None and isinstance(records, list) and records:
        # Modal-aggregator-path dict survived on the last record.
        folds_coeffs = records[-1].get("_ridge_folds_coeffs") if isinstance(
            records[-1], dict) else None
    with open(fi_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["predictor", "feature", "fold", "coefficient",
                     "abs_coefficient", "mean_abs_coeff",
                     "std_abs_coeff", "rank_by_abs_coeff"])
        if folds_coeffs and "folds" in folds_coeffs:
            # Aggregate per (predictor, feature) across folds for the
            # mean / std / rank columns.
            per_pred_feat: Dict[Tuple[str, str], List[float]] = {}
            for fold in folds_coeffs["folds"]:
                pred = fold["predictor"]
                for fname, coef in fold["coefficients"].items():
                    per_pred_feat.setdefault((pred, fname), []).append(coef)
            for fold in folds_coeffs["folds"]:
                pred = fold["predictor"]
                for fname, coef in fold["coefficients"].items():
                    absc = abs(coef)
                    all_coefs = per_pred_feat[(pred, fname)]
                    all_abs = [abs(c) for c in all_coefs]
                    mean_abs = float(np.mean(all_abs))
                    std_abs = float(_safe_stdhood(all_abs, ddof=1) if len(
                        all_abs) > 1 else 0.0)
                    # Per-feature rank within this predictor's feature
                    # set, by cross-fold mean abs coefficient.
                    pred_feats = sorted({k[1] for k in per_pred_feat
                                            if k[0] == pred})
                    pred_mean_abs = {fn: float(np.mean(
                        [abs(c) for c in per_pred_feat[(pred, fn)]]))
                        for fn in pred_feats}
                    ranked = sorted(pred_mean_abs.items(),
                                     key=lambda kv: -kv[1])
                    rank_map = {fn: i + 1 for i, (fn, _) in enumerate(
                        ranked)}
                    rank = rank_map[fname]
                    w.writerow([
                        pred, fname,
                        f"{fold['model']}_{fold['dataset']}_seed{fold['seed']}",
                        f"{coef:.6f}", f"{absc:.6f}",
                        f"{mean_abs:.6f}", f"{std_abs:.6f}", rank,
                    ])

    doc = {
        "n_cells": len(record_iter),
        "predictors": ALL_PREDICTORS,
        "per_input_csv": csv_path,
        "aggregate_csv": agg_path,
        "predictor_metrics_csv": pm_path,
        "correlation_table_csv": ct_path,
        "comparisons_csv": cmp_path,
        "feature_importance_csv": fi_path,
        "comparisons": comparisons,
        "seed_variance_csv": var_path,
    }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return doc
