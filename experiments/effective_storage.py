"""Recompute EXACT effective storage for compression-shootout results.

The nominal budget grid used by ``experiments/dim_reduction.py`` charges
payload bytes only. This pass adds every metadata / amortised-shared-
parameter term so that methods are comparable at equal *total* storage:

  pca/rp      : d*4 payload + (D*d*4)/V projection + (D*4)/V mean vector
                (V = vocabulary size; projection/mean amortised over it)
  *_f32 sparse: nnz*(4+2)                      value + uint16 index
  *_i8 sparse : nnz*(1+2) + 2                  value + index + fp16 scale
  pq          : m + (256*D*4)/V                code + codebook amortised

Quality-vs-storage curves are then interpolated at matched EFFECTIVE
byte targets so the comparison is genuinely equal-storage.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = {   # model -> (vocab, dim)
    "bert-base": (30522, 768), "distilbert": (30522, 768),
    "gpt2": (50257, 768), "roberta-base": (50265, 768),
    "deberta-base": (50265, 768), "tinyllama": (32000, 2048),
}


def effective_bytes(model: str, method: str, param: str) -> float:
    V, D = META[model]
    m = re.search(r"(\d+)", param or "")
    k = int(m.group(1)) if m else 0
    if method == "pca":
        return k * 4 + (D * k * 4) / V + (D * 4) / V
    if method == "rand_proj":
        return k * 4 + (D * k * 4) / V
    if method.endswith("_f32") and not method.startswith("dct"):
        return k * 6.0
    if method == "dct_f32":
        return k * 6.0
    if method == "fft_c64":
        return k * 10.0
    if method.endswith("int8") or method.endswith("_i8"):
        return k * 3 + 2.0
    if method == "pq":
        return k + (256 * D * 4) / V
    raise ValueError(method)


def load_rows(models):
    rows = []
    for m in models:
        p = os.path.join(ROOT, "results", "dim_reduction",
                          f"{m}_compression_shootout.csv")
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["model"] = m
                r["eff_bytes"] = effective_bytes(m, r["method"],
                                                  r.get("param", ""))
                rows.append(r)
    return rows


def interp_at(rows_by_method, target):
    """Interpolate quality at an effective-bytes target, PER MODEL, then
    average over models that have coverage."""
    out = {}
    for meth, pts in rows_by_method.items():
        # pts entries carry no model tag here; group externally instead.
        pass
    return out


def interp_model(rows, target):
    """rows: list of (eff_bytes, cos, jac) for ONE model+method."""
    by_knob = defaultdict(list)
    for b, c, j in rows:
        by_knob[b].append((c, j))
    xs = sorted(by_knob)
    if len(xs) < 2 or not (xs[0] <= target <= xs[-1]):
        return None
    cs = [np.mean([t[0] for t in by_knob[x]]) for x in xs]
    js = [np.mean([t[1] for t in by_knob[x]]) for x in xs]
    lg = np.log(np.array(xs, dtype=float))
    lt = np.log(target)
    return (float(np.interp(lt, lg, cs)), float(np.interp(lt, lg, js)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(META))
    p.add_argument("--targets", nargs="+", type=int,
                   default=[64, 128, 256, 512, 1024])
    args = p.parse_args()

    rows = load_rows(args.models)

    # write corrected per-row CSVs
    out_dir = os.path.join(ROOT, "results", "dim_reduction")
    fields = ["model", "method", "param", "eff_bytes", "cosine",
               "jaccard10"]
    with open(os.path.join(out_dir, "all_effective_bytes.csv"), "w",
               newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([r["model"], r["method"], r.get("param"),
                         f"{r['eff_bytes']:.1f}", r["cosine"],
                         r["jaccard10"]])

    print("=== POOLED AT MATCHED EFFECTIVE BYTES/TOKEN "
          "(cosine / neighbour-Jaccard@10) ===")
    print(f"{'B_eff':>7} " + " ".join(f"{mm:>13}" for mm in (
        "pca", "rand_proj", "topk_i8", "wav_i8", "pq")))
    table_lines = []
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["method"]][r["model"]].append(
            (r["eff_bytes"], float(r["cosine"]), float(r["jaccard10"])))
    for T in args.targets:
        cells = []
        for meth in ("pca", "rand_proj", "topk_i8", "wav_i8", "pq"):
            per_model = [interp_model(by[meth][mdl], T)
                          for mdl in by[meth]]
            vals = [v for v in per_model if v]
            if not vals:
                cells.append("--")
            else:
                c = float(np.mean([v[0] for v in vals]))
                j = float(np.mean([v[1] for v in vals]))
                cov = len(vals)
                cells.append(f"{c:.3f}/{j:.2f}({cov})")
        print(f"{T:>7} " + " ".join(c.rjust(15) for c in cells))
        table_lines.append([T] + cells)

    with open(os.path.join(out_dir, "pooled_effective_table.csv"), "w",
               newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eff_bytes", "pca", "rand_proj", "topk_i8",
                     "wav_i8", "pq"])
        w.writerows(table_lines)
    print("\nsaved all_effective_bytes.csv + pooled_effective_table.csv "
          "(n_models with coverage in parentheses)")


if __name__ == "__main__":
    main()
