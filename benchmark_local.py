"""Crash-resumable local Phase-5+6 benchmark driver.

Each (model, dataset, seed) cell runs in its OWN foreground python
process and is pickled to ``<out>/records/cell_<model>_<ds>_<seed>.pkl``
as soon as it finishes -- a kill mid-sweep never loses completed cells.
After every expected cell exists, one final pass pickles everything into
a BenchResult, applies the Phase-6 ridge LOOO post-pass and writes the
canonical CSV/JSON bundle.

Usage:
  python benchmark_local.py --models bert-base distilbert gpt2 \
      --datasets wikitext2 --seeds 0 1 2 --max-sentences 12
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

from benchmark.runner import (
    ALL_PREDICTORS, BenchRecord, BenchResult, run_cell, write_results,
    apply_ridge_looo,
)


def cell_path(out: str, model: str, ds: str, seed: int) -> str:
    return os.path.join(out, "records", f"cell_{model}_{ds}_{seed}.pkl")


def run_missing(models, datasets, seeds, max_sentences, max_len, out) -> None:
    rec_dir = os.path.join(out, "records")
    os.makedirs(rec_dir, exist_ok=True)
    feats = os.path.join(out, "cells")
    os.makedirs(feats, exist_ok=True)
    todo = [(m, d, s) for m in models for d in datasets for s in seeds
             if not os.path.isfile(cell_path(out, m, d, s))]
    for m, d, s in todo:
        print(f"[cell] {m}/{d}/seed={s} ...", flush=True)
        rec = run_cell(m, d, s, max_sentences=max_sentences, max_len=max_len,
                       metric="cosine_drop", verbose=True,
                       keep_per_input_features=True, features_out_dir=feats)
        with open(cell_path(out, m, d, s), "wb") as f:
            pickle.dump(rec, f)
        print(f"[cell] saved {cell_path(out, m, d, s)} "
              f"(n_heads={rec.n_heads})", flush=True)


def finalize(out: str, test: str = "wilcoxon") -> None:
    result = BenchResult()
    for fn in sorted(os.listdir(os.path.join(out, "records"))):
        if not fn.endswith(".pkl"):
            continue
        with open(os.path.join(out, "records", fn), "rb") as f:
            rec = pickle.load(f)
        if isinstance(rec, BenchRecord) and rec.n_heads:
            result.records.append(rec)
    coeffs_art = apply_ridge_looo(result, alpha=1.0, verbose=True,
                                   return_coeffs=True)
    result.ridge_folds_coeffs = coeffs_art
    doc = write_results(result, out, test=test)
    print("\n[benchmark] done.")
    print(" cells:", doc["n_cells"])
    print(" summary.json:", os.path.join(out, "summary.json"))
    n_win = sum(1 for c in doc["comparisons"] if c["diff_mean"] < 0)
    print(f" comparisons: {len(doc['comparisons'])} "
          f"(predictor-side better in {n_win})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--max-sentences", type=int, default=12)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--out", default=os.path.join("results", "benchmark"))
    p.add_argument("--stage", choices=["run", "finalize", "all"],
                    default="all")
    args = p.parse_args()

    if args.stage in ("run", "all"):
        run_missing(args.models, args.datasets, args.seeds,
                     args.max_sentences, args.max_length, args.out)
    if args.stage in ("finalize", "all"):
        finalize(args.out)


if __name__ == "__main__":
    main()
