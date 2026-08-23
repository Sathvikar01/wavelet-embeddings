"""Generate paper/gen_table.tex at equal EFFECTIVE storage budgets.

Uses experiments.effective_storage.effective_bytes (payload + metadata +
amortised shared parameters) and interpolates each method's
quality-vs-storage curve per model. Infeasible cells become '--'.
"""

import csv
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.effective_storage import effective_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["roberta-base", "deberta-base", "tinyllama"]
LABELS = {"roberta-base": "RoBERTa-base", "deberta-base": "DeBERTa-base",
           "tinyllama": "TinyLlama-2048"}
BUDGETS = [128, 256, 512]
METHODS = ["pca", "topk_i8", "wav_i8", "pq"]
COLS = {"pca": "PCA", "topk_i8": "TopK-int8", "wav_i8": "Wav-int8",
         "pq": "PQ"}


def load(model):
    p = os.path.join(ROOT, "results", "dim_reduction",
                      f"{model}_compression_shootout.csv")
    pts = defaultdict(list)
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            eff = effective_bytes(model, r["method"], r.get("param", ""))
            pts[r["method"]].append(
                (eff, float(r["cosine"]), float(r["jaccard10"])))
    return pts


def interp(points, target):
    """points: list of (eff, cos, jac). None when not covered."""
    xs = sorted({round(p[0], 1) for p in points})
    if len(xs) < 2 or not (xs[0] <= target <= xs[-1]):
        return None
    by = defaultdict(list)
    for e, c, j in points:
        by[round(e, 1)].append((c, j))
    cs = [np.mean([t[0] for t in by[x]]) for x in xs]
    js = [np.mean([t[1] for t in by[x]]) for x in xs]
    lg = np.log(np.array(xs, dtype=float))
    lt = np.log(target)
    return (float(np.interp(lt, lg, cs)), float(np.interp(lt, lg, js)))


def fmt(v):
    return "--" if v is None else f"{v[0]:.2f}/{v[1]:.2f}"


def main() -> None:
    data = {m: load(m) for m in MODELS}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Generalisation of the compression ranking to unseen",
        "architectures at equal \\emph{effective} storage budgets",
        "(cosine / top-10 neighbour overlap; bytes/token including all",
        "metadata and amortised shared parameters). Best sparse method",
        "per row bolded; -- = method cannot fit the budget.}",
        "\\label{tab:gen}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & $B_{\\mathrm{eff}}$ & PCA & TopK-int8 & Wav-int8 & PQ \\\\",
        "\\midrule",
    ]
    for m in MODELS:
        d = data[m]
        curves = {meth: d[meth] for meth in METHODS if meth in d}
        first = True
        for B in BUDGETS:
            vals = {meth: interp(curves[meth], B) if meth in curves else None
                     for meth in METHODS}
            sparse_best = max(
                ((v[1], k) for k, v in vals.items()
                  if v is not None and k in ("topk_i8", "wav_i8")),
                default=None)
            cells = []
            for meth in METHODS:
                s = fmt(vals[meth])
                if sparse_best and vals[meth] is not None \
                        and sparse_best[1] == meth:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
            model_cell = LABELS[m] if first else ""
            lines.append(f"{model_cell} & {B} & " + " & ".join(cells)
                          + " \\\\")
            first = False
        lines.append("\\midrule" if m != MODELS[-1] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]

    out_path = os.path.join(ROOT, "paper", "gen_table.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
