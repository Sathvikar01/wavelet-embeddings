"""Phase-3 reconstruction experiment for attention heads.

For every (layer, head):

  * Apply thresholds 10/20/30/40/50% on smallest detail coefficients
  * Reconstruct the attention matrix
  * Measure:
      - PSNR (peak signal-to-noise ratio)
      - cosine similarity (original vs reconstructed, flatten & cosine)
      - KL divergence (attention_* distributions -- uses normalized
        probabilities of attention rows, KL(orig || rec))
      - attention drift (L2)
      - top-k token ranking preservation (mean kendall tau approximation
        based on shared top-k sets across query rows)

Also computes a per-(model, wavelet) summary CSV of mean PSNR vs ratio curve.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from attention import (
    AttentionLoader, AttentionWaveletDecomposer, HeadMatrix,
)
from attention.analyzer import HeadDecomposition


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_COMPRESSION_RATIOS: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb)) if na and nb else 0.0


def _psnr(mse: float, peak: float) -> float:
    if mse <= 0 or peak <= 0:
        return float("inf")
    return float(20.0 * np.log10(peak / np.sqrt(mse)))


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) row-averaged with eps-smoothing on rows."""
    p = p.astype(np.float64) + 1e-12
    q = q.astype(np.float64) + 1e-12
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    kls = np.sum(p * np.log(p / q), axis=1)
    return float(np.mean(kls))


def _topk_preservation(orig: np.ndarray, rec: np.ndarray, k: int = 5) -> float:
    """Mean Jaccard overlap of top-k attended-to keys across queries."""
    if orig.shape != rec.shape:
        return 0.0
    T = orig.shape[0]
    k = min(k, T)
    out = []
    for i in range(T):
        a = set(np.argpartition(-orig[i], k - 1)[:k].tolist())
        b = set(np.argpartition(-rec[i], k - 1)[:k].tolist())
        u = a | b
        if not u:
            out.append(0.0)
        else:
            out.append(len(a & b) / len(u))
    return float(np.mean(out))


# --------------------------------------------------------------------------- #
# Per-head results
# --------------------------------------------------------------------------- #

@dataclass
class HeadCompressionReport:
    layer: int
    head: int
    ratios: List[float]
    psnrs: List[float]
    cosines: List[float]
    kls: List[float]
    drifts: List[float]
    topk_preservations: List[float]


# --------------------------------------------------------------------------- #
# Per-head runner
# --------------------------------------------------------------------------- #

def compress_head_attention(
    head: HeadMatrix,
    decomposer: AttentionWaveletDecomposer,
    ratios: Sequence[float] = DEFAULT_COMPRESSION_RATIOS,
    k_top: int = 5,
) -> HeadCompressionReport:
    A = head.normalized
    d = decomposer.decompose(A)
    peak = float(np.abs(A).max() or 1.0)
    psnrs, cosines, kls, drifts, topks = [], [], [], [], []
    for r in ratios:
        rec = decomposer.reconstruct(d, threshold_ratio=r, crop=True)
        diff = A - rec
        mse = float(np.mean(diff ** 2))
        psnrs.append(_psnr(mse, peak))
        cosines.append(_cosine(A, rec))
        kls.append(_kl_div(A, rec))
        drifts.append(float(np.linalg.norm(diff)))
        topks.append(_topk_preservation(A, rec, k=k_top))
    return HeadCompressionReport(
        layer=head.layer, head=head.head,
        ratios=list(ratios),
        psnrs=psnrs, cosines=cosines, kls=kls,
        drifts=drifts, topk_preservations=topks,
    )


# --------------------------------------------------------------------------- #
# Batch - whole model
# --------------------------------------------------------------------------- #

def compress_all_heads(
    loader: AttentionLoader,
    decomposer_factory,
    ratios: Sequence[float] = DEFAULT_COMPRESSION_RATIOS,
    save_dir: Optional[str] = None,
) -> List[HeadCompressionReport]:
    dec = decomposer_factory()
    out: List[HeadCompressionReport] = []
    for L in range(loader.n_layers):
        for H in range(loader.n_heads):
            head = loader.load_head(L, H)
            out.append(compress_head_attention(head, dec, ratios=ratios))
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import csv
        with open(os.path.join(save_dir, "compression_results.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w_ = csv.writer(f)
            w_.writerow([
                "layer", "head", "ratio",
                "psnr_db", "cosine", "kl", "drift", "topk_preservation",
            ])
            for r in out:
                for k in range(len(r.ratios)):
                    w_.writerow([
                        r.layer, r.head, f"{r.ratios[k]:.2f}",
                        f"{r.psnrs[k]:.3f}", f"{r.cosines[k]:.4f}",
                        f"{r.kls[k]:.5f}", f"{r.drifts[k]:.5f}",
                        f"{r.topk_preservations[k]:.4f}",
                    ])
        # Per-model curve plotting
        ratios = list(ratios)
        mean_psnr = np.zeros(len(ratios))
        mean_cosine = np.zeros(len(ratios))
        mean_topk = np.zeros(len(ratios))
        for r in out:
            mean_psnr += np.asarray(r.psnrs)
            mean_cosine += np.asarray(r.cosines)
            mean_topk += np.asarray(r.topk_preservations)
        n = max(1, len(out))
        mean_psnr /= n
        mean_cosine /= n
        mean_topk /= n
        # PSNR curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ratios, mean_psnr, marker="o", color="tab:blue", label="PSNR (dB)")
        ax.set_xlabel("Compression ratio")
        ax.set_ylabel("Mean PSNR (dB)")
        ax.set_title("Attention compression robustness (per-head avg)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "compression_psnr_curve.png"),
                    dpi=140, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, "compression_psnr_curve.pdf"),
                    bbox_inches="tight")
        plt.close(fig)
        # Cosine & topk preservation
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ratios, mean_cosine, marker="s", color="tab:green",
                label="Cosine(orig, rec)")
        ax.plot(ratios, mean_topk, marker="^", color="tab:red",
                label="Top-k attention preservation")
        ax.set_xlabel("Compression ratio")
        ax.set_ylabel("Mean")
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "compression_cosine_topk.png"),
                    dpi=140, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, "compression_cosine_topk.pdf"),
                    bbox_inches="tight")
        plt.close(fig)
    return out
