"""Leave-one-MODEL-out ridge evaluation (harsher than leave-one-cell-out).

For each held-out model, the ridge is fitted on every cell of the OTHER
models (all datasets/seeds) and scores the held-out model's inputs.
Compares against the leave-one-cell-out numbers to show that transfer is
cross-architecture rather than within-cell memorisation.

Reads the per-cell feature matrices persisted under
``results/benchmark/cells/cell_<model>_<ds>_<seed>.npz``.
"""

from __future__ import annotations

import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.feature_matrix import feature_names        # noqa: E402
from benchmark.ridge_looo import (                        # noqa: E402
    _column_subset, _fit_ridge, _safe_pearson,
)


def load_cells(root):
    cells = []
    for p in sorted(glob.glob(os.path.join(root, "cell_*.npz"))):
        d = np.load(p, allow_pickle=False)
        cells.append(dict(model=str(d["model"][0]),
                           dataset=str(d["dataset"][0]),
                           seed=int(d["seed"][0]),
                           X=d["X"], y=d["y"]))
    return cells


GROUP_KEY = {"combined": "combined", "wavelet_only_model": "wavelet_only",
              "attention_only_model": "attn_only",
              "wavelet_feature_model": "combined"}


def evaluate(group="combined", alpha=1.0, root=None):
    root = root or os.path.join("results", "benchmark", "cells")
    cells = load_cells(root)
    full_names = feature_names(GROUP_KEY[group])
    models = sorted({c["model"] for c in cells})
    out = {}
    for held in models:
        train = [c for c in cells if c["model"] != held]
        test = [c for c in cells if c["model"] == held]
        Xs, ys = [], []
        for c in train:
            Xf = c["X"].reshape(-1, c["X"].shape[-1])
            sub = _column_subset(Xf, full_names, GROUP_KEY[group])
            Xs.append(sub)
            ys.append(np.concatenate([yy for yy in c["y"]]))
        X_train = np.concatenate(Xs, axis=0)
        y_train = np.concatenate(ys)
        w, b, _ = _fit_ridge(X_train, y_train, alpha)
        rs = []
        for c in test:
            Xf = c["X"].reshape(-1, c["X"].shape[-1])
            Xh = _column_subset(Xf, full_names, GROUP_KEY[group])
            nh = c["y"].shape[1]
            for inp in range(c["y"].shape[0]):
                y_hat = Xh[inp * nh:(inp + 1) * nh] @ w + b
                rs.append(_safe_pearson(
                    -y_hat, np.asarray(c["y"][inp], dtype=float)))
        out[held] = (float(np.nanmean(rs)), float(np.nanstd(rs)), len(rs))
    return out


def main() -> None:
    for group in ("wavelet_only_model", "attention_only_model",
                   "wavelet_feature_model"):
        res = evaluate(group)
        print(f"\n== leave-one-model-out ({group}) ==")
        for m, (mean, std, n) in sorted(res.items()):
            print(f"  held-out {m:<12} mean r = {mean:+.4f} "
                  f"(sd {std:.4f}, n={n})")


if __name__ == "__main__":
    main()
