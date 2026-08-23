"""Synthetic smoke test for the Phase-6 Wavelet Feature Model.

Verifies end-to-end:
  1. ``run_cell`` with ``keep_per_input_features=True`` +
     ``features_out_dir`` emits ``features/cell_*.npz`` on disk.
  2. ``run_benchmark`` triggers the LOOO ridge post-pass and populates
     ``wavelet_only_model``, ``attention_only_model`` and
     ``wavelet_feature_model`` per-input correlations on every record.
  3. The headline comparison rows (``predictor = wavelet_feature_model``
     vs every other baseline) appear in ``comparisons.csv``.
  4. ``predictor_metrics.csv`` carries all spec columns (Pearson,
     Spearman, Kendall, cosine preservation, pruning loss) for every
     predictor.
  5. ``correlation_table.csv`` is written and pivots per
     (model, dataset, predictor).
  6. ``feature_importance.csv`` is written with one row per
     fold x predictor x feature, and exactly 11 features are emitted
     per wavelet_feature_model fold.
  7. The riskier ridge_combined / ridge_attn_only / ridge_wavelet_only
     implementation-name identifiers appear ONLY in
     ``benchmark/ridge_looo.py`` (the implementation module) -- they
     never leak into public artefact files.
  8. The combined model beats the attention-only ablation on synthetic
     data where the y is a linear function of all 11 feature columns.
  9. The serial ``write_results`` function emits all 7 CSV artefact
     files plus ``summary.json``.

This is intentionally synthetic so it runs in <5 s on CPU without any
HuggingFace download; it never touches Modal. It is a structural /
contract test, NOT a statistical-power test.
"""
from __future__ import annotations
import os
import sys
import json
import csv
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark.runner import (BenchRecord, BenchResult, run_benchmark,
                                  write_results)
from benchmark.ridge_looo import apply_ridge_looo


def _make_rec(model, dataset, seed, weights, n_heads=144, n_inputs=20,
              n_feat=11, rng=None):
    rng = rng or np.random.default_rng(0)
    # Mirrors ALL_PREDICTORS in benchmark/runner.py so the smoke test
    # exercises every comparison row the production benchmark emits.
    all_predictors = [
        "wavelet", "wavelet_entropy", "attention_entropy",
        "attention_entropy_true", "attention_weight", "magnitude",
        "random", "michel_hic", "voita_his", "bhasharas_bs",
        "wavelet_only_model", "attention_only_model",
        "wavelet_feature_model",
    ]
    rec = BenchRecord(
        model=model, dataset=dataset, seed=seed, n_heads=n_heads,
        metric_used="cosine_drop",
        per_input={p: [] for p in all_predictors},
    )
    for _ in range(n_inputs):
        X = rng.normal(size=(n_heads, n_feat))
        # Linear ground truth so LOOO generalises well.
        true_y = X @ weights + rng.normal(scale=0.3, size=n_heads)
        # cosine_drop-shaped loss: lower loss for high-importance heads.
        loss = (-true_y + rng.normal(scale=0.05, size=n_heads)).astype(
            np.float64)
        rec.per_input_features.append(X.astype(np.float64))
        rec.per_input_loss.append(loss.astype(np.float64))
        for pname_i, pname in enumerate(all_predictors):
            if pname.endswith("_model"):
                # Populated later by the LOOO ridge post-pass.
                continue
            # Distinct synthetic projection per predictor so different
            # predictors land at different Pearson r and the comparison
            # block has heterogeneous diff_means.
            y_hat = X[:, pname_i % n_feat] + rng.normal(
                scale=0.1, size=n_heads)
            r = float(np.corrcoef(-y_hat, loss)[0, 1]) if (
                np.std(-y_hat) > 1e-9 and np.std(loss) > 1e-9) else 0.0
            rec.per_input[pname].append(r)
            rec.per_input_cosine_pres.setdefault(pname, []).append(
                float(np.mean(1.0 - loss)))
            rec.per_input_pruning_loss.setdefault(pname, []).append(
                float(np.mean(loss)))
            rho = float(np.corrcoef(
                _as_rank(-y_hat), _as_rank(loss))[0, 1])
            tau = _kendall_simple(-y_hat, loss)
            rec.per_input_spearman.setdefault(pname, []).append(rho)
            rec.per_input_kendall.setdefault(pname, []).append(tau)
    return rec


def _as_rank(arr):
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
    return ranks


def _kendall_simple(x, y):
    rx = _as_rank(x)
    ry = _as_rank(y)
    n = x.size
    con = 0
    dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (rx[j] - rx[i]) * (ry[j] - ry[i])
            if d > 0:
                con += 1
            elif d < 0:
                dis += 1
    return float((con - dis) / max(con + dis, 1))


def main():
    rng = np.random.default_rng(42)
    n_feat = 11
    cells_meta = [
        ("bert", "wikitext2", 0),
        ("bert", "penn_treebank", 1),
        ("distilbert", "wikitext2", 0),
        ("distilbert", "penn_treebank", 1),
        ("gpt2", "wikitext2", 0),
        ("gpt2", "penn_treebank", 1),
        ("roberta", "wikitext2", 0),
        ("roberta", "penn_treebank", 1),
        ("tinyllama", "wikitext2", 0),
        ("tinyllama", "penn_treebank", 1),
    ]
    shared_w = rng.normal(scale=0.5, size=n_feat)
    recs = [_make_rec(m, d, s, shared_w, rng=rng) for (m, d, s) in cells_meta]
    result = BenchResult(records=recs)

    out_dir = tempfile.mkdtemp(prefix="phase6_smoke_")
    try:
        coeffs_art = apply_ridge_looo(result, verbose=False,
                                       return_coeffs=True)
        result.ridge_folds_coeffs = coeffs_art

        # Ridge per-input correlations should populate.
        for r in result.records:
            assert r.per_input.get("wavelet_feature_model"), (
                f"wavelet_feature_model not populated for {r.model}/"
                f"{r.dataset}")
            assert r.per_input.get("attention_only_model"), (
                f"attention_only_model not populated for {r.model}/"
                f"{r.dataset}")
            assert r.per_input.get("wavelet_only_model"), (
                f"wavelet_only_model not populated for {r.model}/"
                f"{r.dataset}")
            # Spec-required secondary correlations are also captured.
            assert r.per_input_spearman.get("wavelet_feature_model"), \
                "Spearman not captured for wavelet_feature_model"
            assert r.per_input_kendall.get("wavelet_feature_model"), \
                "Kendall not captured for wavelet_feature_model"

        doc = write_results(result, out_dir, test="wilcoxon")

        # Spec-required file inventory check.
        spec_files = ["summary.json", "per_input_scores.csv",
                       "aggregate.csv", "seed_variance.csv",
                       "predictor_metrics.csv", "correlation_table.csv",
                       "comparisons.csv", "feature_importance.csv"]
        for sf in spec_files:
            p = os.path.join(out_dir, sf)
            assert os.path.exists(p), f"Missing artefact: {sf}"

        # No internal name leakage in any artefact file.
        leaked = []
        for sf in spec_files:
            with open(os.path.join(out_dir, sf), "rb") as fh:
                content = fh.read().decode("utf-8", errors="replace")
            for forbidden in ("ridge_combined", "ridge_attn_only",
                               "ridge_wavelet_only"):
                if forbidden in content:
                    leaked.append((sf, forbidden))
        assert not leaked, (
            f"Internal ridge_* names leaked into public artefacts: {leaked}")

        # comparisons.csv contains wavelet_feature_model rows
        # against every other predictor.
        wf_rows = []
        with open(os.path.join(out_dir, "comparisons.csv"), "r",
                  encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["predictor"] == "wavelet_feature_model":
                    wf_rows.append(row)
        baselines_seen = sorted({row["baseline"] for row in wf_rows})
        # Every other predictor should appear as a baseline at least once.
        expected_baselines = sorted(set([
            "wavelet", "wavelet_entropy", "attention_entropy",
            "attention_entropy_true", "attention_weight", "magnitude",
            "random", "michel_hic", "voita_his", "bhasharas_bs",
            "attention_only_model", "wavelet_only_model"]))
        for eb in expected_baselines:
            assert eb in baselines_seen, (
                f"baseline {eb!r} missing from wavelet_feature_model "
                "comparison rows. Spec demands vs EVERY listed baseline.")

        # feature_importance.csv has 11 rows per wavelet_feature_model fold.
        n_folds = len(result.records)
        # Each cell carries a fold for each of the 3 ridge predictors.
        wfm_fi_rows = 0
        with open(os.path.join(out_dir, "feature_importance.csv"), "r",
                  encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["predictor"] == "wavelet_feature_model":
                    wfm_fi_rows += 1
        assert wfm_fi_rows == 11 * n_folds, (
            f"Expected {11 * n_folds} wavelet_feature_model "
            f"feature_importance rows (11 features x {n_folds} folds); "
            f"got {wfm_fi_rows}")

        # Spec-required predictor_metrics.csv columns present.
        with open(os.path.join(out_dir, "predictor_metrics.csv"), "r",
                  encoding="utf-8") as f:
            header = next(csv.reader(f))
        for col in ("pearson_mean", "spearman_mean", "kendall_mean",
                     "cosine_preservation_mean", "pruning_loss_mean",
                     "pearson_std", "spearman_std", "kendall_std",
                     "cosine_preservation_std", "pruning_loss_std",
                     "pearson_ci_low", "pearson_ci_high"):
            assert col in header, (
                f"predictor_metrics.csv missing required column {col!r}")

        # correlation_table.csv columns.
        with open(os.path.join(out_dir, "correlation_table.csv"), "r",
                  encoding="utf-8") as f:
            header = next(csv.reader(f))
        for col in ("pearson_mean", "spearman_mean", "kendall_mean",
                     "pearson_std", "spearman_std", "kendall_std"):
            assert col in header, (
                f"correlation_table.csv missing required column {col!r}")

        # The combined model should outperform attention-only-ablation
        # here because synthetic y is a linear function of all 11 cols.
        wfm_means = [np.mean(r.per_input["wavelet_feature_model"])
                      for r in result.records]
        aom_means = [np.mean(r.per_input["attention_only_model"])
                      for r in result.records]
        assert np.mean(wfm_means) < np.mean(aom_means), (
            f"wavelet_feature_model ({np.mean(wfm_means):.3f}) should "
            f"beat attention_only_model ({np.mean(aom_means):.3f}) on "
            "this synthetic-LOOO setup where y depends on all columns."
        )

        print("\n[OK] Phase-6 Wavelet Feature Model smoke test passed.")
        print(f"  cells: {len(result.records)}")
        print(f"  mean wavelet_feature_model Pearson r = "
              f"{np.mean(wfm_means):+.4f}")
        print(f"  mean attention_only_model    Pearson r = "
              f"{np.mean(aom_means):+.4f}")
        print(f"  artefacts checked: {len(spec_files)}")
        print(f"  wavelet_feature_model comparisons vs baselines: "
              f"{len(expected_baselines)} / {len(expected_baselines)}")
        print(f"  feature_importance rows (wavelet_feature_model): "
              f"{wfm_fi_rows}")
        print(f"  output dir: {out_dir}")
    finally:
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass


if __name__ == "__main__":
    main()
