"""Publication-quality heatmap visualisations for Phase-3 attention matrices.

Functions:
  * ``original_attention_heatmap``       - imshow of the raw attention matrix
  * ``wavelet_coefficient_heatmap``       - imshow of approx + 3 detail bands
  * ``energy_spectrum_bars``              - stacked bar of energy per band/level
  * ``coefficient_histogram``             - histogram of wavelet coeff magnitudes
  * ``reconstruction_comparison``         - side-by-side original vs reconstruction

All save as both PNG and PDF.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from attention import HeadDecomposition
from attention.loader import HeadMatrix


def _save(fig, save_dir: str, basename: str):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, basename + ".png"),
                dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, basename + ".pdf"),
                bbox_inches="tight")
    plt.close(fig)


def original_attention_heatmap(
    head: HeadMatrix,
    tokens: List[str],
    save_dir: str,
    basename: str = "attention_heatmap",
    title: str = "Attention matrix",
):
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.55),
                                    max(6, len(tokens) * 0.55)))
    norm = mcolors.LogNorm(
        vmin=max(head.normalized.min(), 1e-12),
        vmax=head.normalized.max() or 1.0,
    )
    im = ax.imshow(head.normalized, aspect="auto", cmap="viridis",
                   norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    short = [t[:8] for t in tokens]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize="8")
    ax.set_yticklabels(short, fontsize="8")
    ax.set_title(f"{title}\n(layer {head.layer}, head {head.head})")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    _save(fig, save_dir, basename)


def wavelet_coefficient_heatmap(
    decomposition: HeadDecomposition,
    save_dir: str,
    basename: str = "wavelet_coefficients",
    title: str = "Wavelet coefficient magnitudes",
):
    """Subband layout: approx top-left, then a 4-by-level grid."""
    approx = np.abs(decomposition.approx)
    levels = sorted(decomposition.details.keys())
    n_lev = max(1, len(levels))
    n_row = n_lev + 1
    n_col = 4
    fig, axes = plt.subplots(n_row, n_col,
                              figsize=(3.5 * n_col, 3 * n_row))
    if n_row == 0:
        return
    # Approx spans full first row (across 4 cols when only 1 col tensor)
    ax = axes[0, 0] if n_row > 1 else axes[0]
    ax.imshow(approx, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_title("cA (approx)")
    for col_spine in axes[0, 1:] if n_row > 1 else []:
        col_spine.axis("off")
    # Per-level: cAH, cAV, cAD
    for i, lvl in enumerate(levels):
        d = decomposition.details[lvl]
        for j, k in enumerate(("cAH", "cAV", "cAD")):
            ax = axes[i + 1, j]
            ax.imshow(np.abs(d[k]), aspect="auto", cmap="magma",
                      interpolation="nearest")
            ax.set_title(f"{k} (L{lvl})")
        axes[i + 1, 3].axis("off")
    fig.suptitle(f"{title}\n({decomposition.wavelet_name}, level {decomposition.level})")
    fig.tight_layout()
    _save(fig, save_dir, basename)


def energy_spectrum_bars(
    decomposition: HeadDecomposition,
    save_dir: str,
    basename: str = "energy_spectrum",
):
    """Grouped bars per (level, band)."""
    levels = sorted(decomposition.energy_subband.keys())
    bands = ("cAH", "cAV", "cAD")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(levels))
    width = 0.27
    cmap = plt.cm.tab10(np.linspace(0, 1, len(bands)))
    for i, b in enumerate(bands):
        ys = [decomposition.energy_subband[l][b] for l in levels]
        ax.bar(x + (i - 1) * width, ys, width, label=b, color=cmap[i])
    # Approx as separate bar at x=-1
    if decomposition.energy_approx:
        ax.bar([-1], decomposition.energy_approx, width, label="cA",
               color="darkred")
    ax.set_xticks(np.append([-1], x))
    ax.set_xticklabels([f"L{l}" for l in [-1] + levels])
    ax.set_ylabel("Energy")
    ax.set_title(f"Per-band wavelet energy\n({decomposition.wavelet_name})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, basename)


def coefficient_histogram(
    decomposition: HeadDecomposition,
    save_dir: str,
    basename: str = "coefficient_histogram",
):
    parts = [decomposition.approx.ravel()]
    for lvl in decomposition.details:
        for k in ("cAH", "cAV", "cAD"):
            parts.append(decomposition.details[lvl][k].ravel())
    flat = np.concatenate([np.abs(p).ravel() for p in parts])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if flat.size:
        ax.hist(flat, bins=80, color="teal")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_yscale("log")
    ax.set_xlabel("|wavelet coefficient| (symlog)")
    ax.set_ylabel("Counts (log)")
    ax.set_title(f"Coefficient magnitudes histogram "
                 f"({decomposition.wavelet_name})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, save_dir, basename)


def reconstruction_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_dir: str,
    basename: str = "reconstruction_comparison",
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, lbl in zip(axes, [original, reconstructed],
                              ["Original", "Reconstructed (30% threshold)"]):
        v = np.abs(mat).max() or 1.0
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       interpolation="nearest", vmin=0, vmax=v)
        ax.set_title(lbl)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, save_dir, basename)
