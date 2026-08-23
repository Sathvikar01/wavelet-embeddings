"""End-to-end evaluation driver.

Runs the full pipeline on the curated eval set (data/eval_sentences.json):

  python run_eval.py phase3-extract           # attention snapshots, 3 models x 12 sentences
  python run_eval.py phase3-analyze           # wavelet analysis of every snapshot x 4 wavelets
  python run_eval.py phase4                   # dedicated snapshots + pruning validation
  python run_eval.py benchmark-local          # local Phase-5+6 benchmark with ridge LOOO
  python run_eval.py all                      # everything above in order
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import main as M


def _fix_w(w: str) -> str:
    return M._fix_wavelet_name(w)


def _sentences() -> list:
    with open(os.path.join(os.path.dirname(__file__),
                            "data", "eval_sentences.json"),
               encoding="utf-8") as f:
        return json.load(f)["sentences"]


def _ns(**kw) -> argparse.Namespace:
    base = dict(models=list(M.DEFAULT_MODELS),
                wavelets=list(M.DEFAULT_WAVELETS),
                sample=2000, out=None, data_dir=None,
                anchors=None, sentences=None, max_length=64,
                src_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


def phase3_extract(args) -> None:
    sents = _sentences()
    print(f"[eval] phase3-extract: {len(sents)} sentences x {args.models}")
    M.cmd_attention_extract(_ns(models=args.models, sentences=sents,
                                  max_length=args.max_length))


def phase3_analyze(args) -> None:
    print(f"[eval] phase3-analyze: wavelets={args.wavelets}")
    M.cmd_attention_analyze(_ns(models=args.models, wavelets=args.wavelets,
                                  max_length=args.max_length))


def _pruning_complete(model: str, wavelets) -> bool:
    """True when every wavelet's predictor_correlations.csv already exists."""
    root = os.path.join(M._default_results_dir(), "pruning")
    snap_dirs = [d for d in os.listdir(root)
                  if d.startswith(model + "_")]
    if not snap_dirs:
        return False
    snap = sorted(snap_dirs)[0]
    missing = [w for w in wavelets if not os.path.isfile(os.path.join(
        root, snap, "wavelet_" + w, "predictor_correlations.csv"))]
    return not missing


def phase4(args) -> None:
    """Dedicated pruning snapshots: one per model (first eval sentence),
    analysed against the FULL eval sentence set."""
    from attention.extractor import extract_and_save

    sents = _sentences()
    src_root = os.path.join(M._default_results_dir(), "attention_pruning",
                             "_sentences")
    os.makedirs(src_root, exist_ok=True)
    probe = sents[0]
    todo_models = []
    for m in args.models:
        sub_dir = os.path.join(
            src_root,
            f"{m}_00_" + probe[:30].replace(" ", "_").replace("?", "")
              .replace(",", "").replace(".", ""))
        if not os.path.isdir(sub_dir):
            print(f"[eval] phase4 snapshot extract: {m} <- {probe!r}")
            extract_and_save(m, probe, sub_dir, max_length=args.max_length)
        else:
            print(f"[eval] phase4 snapshot exists: {sub_dir}")
        if not _pruning_complete(m, [_fix_w(w) for w in args.wavelets]):
            todo_models.append(m)
        else:
            print(f"[eval] phase4 already complete for {m}; skipping")
    if not todo_models:
        print("[eval] phase4 nothing to do.")
        return
    M.cmd_pruning_analyze(_ns(models=todo_models, wavelets=args.wavelets,
                                sentences=sents, max_length=args.max_length,
                                src_dir=src_root))


def benchmark_local(args) -> None:
    """Local Phase-5 + Phase-6 (ridge LOOO) benchmark on CPU."""
    from benchmark.runner import run_benchmark, write_results

    out_root = args.out or os.path.join(M._default_results_dir(), "benchmark")
    os.makedirs(out_root, exist_ok=True)
    feats = os.path.join(out_root, "cells")
    result = run_benchmark(
        models=args.models, datasets=args.datasets, seeds=args.seeds,
        max_sentences=args.max_sentences, max_len=args.max_length,
        metric="cosine_drop", verbose=True,
        enable_phase6_ridge=True, features_out_dir=feats,
    )
    doc = write_results(result, out_root, test="wilcoxon")
    print("\n[eval] local benchmark done.")
    print(" cells:", doc["n_cells"])
    print(" summary.json:", os.path.join(out_root, "summary.json"))
    n_win = sum(1 for c in doc["comparisons"] if c["diff_mean"] < 0)
    print(f" comparisons: {len(doc['comparisons'])} "
          f"(wavelet-side better in {n_win})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["phase3-extract", "phase3-analyze",
                                      "phase4", "benchmark-local", "all"])
    p.add_argument("--models", nargs="+", default=list(M.DEFAULT_MODELS))
    p.add_argument("--wavelets", nargs="+", default=list(M.DEFAULT_WAVELETS))
    p.add_argument("--datasets", nargs="+", default=["wikitext2"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--max-sentences", type=int, default=12)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    stages = {
        "phase3-extract": phase3_extract,
        "phase3-analyze": phase3_analyze,
        "phase4": phase4,
        "benchmark-local": benchmark_local,
    }
    if args.stage == "all":
        for fn in stages.values():
            fn(args)
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()
