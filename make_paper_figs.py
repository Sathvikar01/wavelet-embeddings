"""Generate publication-quality figures for the IEEE paper.

Reads result CSVs produced by the pipeline and writes vector PDFs into
paper/figs/.
"""

from __future__ import annotations

import os
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
})

STYLE = {
    "pca":        dict(color="#888888", ls="--", marker="o", label="PCA"),
    "rand_proj":  dict(color="#bbbbbb", ls=":",  marker="v",
                        label="Rand. proj."),
    "topk_i8":    dict(color="#1f77b4", ls="-.", marker="s",
                        label="Top-k sparse + int8"),
    "wav_i8":     dict(color="#d62728", ls="--", marker="D",
                        label="Wavelet sparse + int8"),
    "dct_f32":    dict(color="#ff7f0e", ls="-", marker="x",
                        label="DCT sparse (fp32)"),
    "pq":         dict(color="#2ca02c", ls="-",  marker="^",
                        label="Product quantization"),
}


def fig_compression() -> None:
    """Pooled cosine & neighbour-Jaccard vs EFFECTIVE bytes/token."""
    path = os.path.join(ROOT, "results", "dim_reduction",
                         "all_effective_bytes.csv")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    budgets = sorted({float(r["eff_bytes"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for metric, ax, ylab in (("cosine", axes[0], "Cosine to original"),
                              ("jaccard10", axes[1],
                               "Top-10 neighbour overlap")):
        for meth, st in STYLE.items():
            xs, ys = [], []
            for B in budgets:
                sel = [float(r[metric]) for r in rows
                        if r["method"] == meth
                        and abs(float(r["eff_bytes"]) - B) < 1e-6]
                if sel:
                    xs.append(B)
                    ys.append(float(np.mean(sel)))
            ax.plot(xs, ys, **st)
        ax.set_xlabel("Effective storage budget (bytes / token)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256, 512])
        ax.set_xticklabels(["64", "128", "256", "512"])
    axes[1].set_ylim(0, 1.0)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compression_shootout.pdf"),
                 bbox_inches="tight")
    plt.close(fig)


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fig_pruning_sweep() -> None:
    """Batched bottom-p% ablation damage, db4, encoder models."""
    snaps = {
        "BERT-base": ("bert-base_00_The_bank_approved_the_loan_for",
                       "#1f77b4"),
        "DistilBERT": ("distilbert_00_The_bank_approved_the_loan_for",
                        "#d62728"),
    }
    preds = [("wavelet", "Wavelet composite", "#1f77b4"),
             ("attention_entropy_true", "Attn. entropy (true)", "#2ca02c"),
             ("magnitude", "Frobenius mass", "#7f7f7f"),
             ("attention_weight", "Max-column mass", "#9467bd"),
             ("random", "Random", "#cccccc")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, (title, (snap, _), ) in zip(axes, snaps.items()):
        path = os.path.join(ROOT, "results", "pruning", snap,
                             "wavelet_db4", "aggregate_pruning.csv")
        data = defaultdict(list)
        for r in _read_csv(path):
            data[r["predictor"]].append(
                (float(r["ratio"]), float(r["cosine_drop"])))
        for pid, label, color in preds:
            pts = sorted(data.get(pid, []))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    marker="o", ms=3, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Pruned fraction of heads")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Sentence-embedding cosine drop")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pruning_sweep.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_layer_progression() -> None:
    path = os.path.join(ROOT, "results", "attention_analysis",
                         "distilbert_00_The_bank_approved_the_loan_for",
                         "wavelet_db4", "layer_progression",
                         "layer_progression.csv")
    rows = _read_csv(path)
    # column names discovered at runtime (layer, *_mean columns)
    cols = {k.lower(): k for k in rows[0].keys()}
    layer_col = cols.get("layer") or list(rows[0].keys())[0]
    def col(*names):
        for n in names:
            for k, v in cols.items():
                if n in k:
                    return v
        return None
    layers = [int(float(r[layer_col])) for r in rows]
    ratio = [float(r[col("ratio_lh")]) for r in rows]
    gini = [float(r[col("gini")]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(3.4, 2.3))
    ax1.plot(layers, ratio, marker="o", ms=3, color="#d62728",
              label="Low-/high-freq. energy ratio")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Low/high-frequency energy ratio", color="#d62728")
    ax2 = ax1.twinx()
    ax2.plot(layers, gini, marker="s", ms=3, color="#1f77b4",
              label="Gini sparsity")
    ax2.set_ylabel("Gini coefficient of coeff. magnitudes", color="#1f77b4")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "layer_progression.pdf"),
                 bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_compression()
    fig_pruning_sweep()
    try:
        fig_layer_progression()
    except Exception as e:
        print(f"[figs] layer progression skipped: {e}")
    print("[figs] written to", OUT)
