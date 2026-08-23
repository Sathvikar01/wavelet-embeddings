"""Heatmap visualisations of wavelet coefficients & per-level energies.

Functions
---------
* ``coefficient_heatmap``   - heatmaps of approx + details for a single token
* ``energy_distribution``   - bar chart of energy per band
* ``tokens_energy_matrix``  - imshow across many tokens x levels
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from wavelets.base import WaveletDecomposition


def _to_rgba_coeffs(coeffs: List[np.ndarray], ax) -> None:
    """Helper - normalised imshow of a 2-D coefficient matrix.

    Pieces of different lengths get them zero-padded to the same length.
    """
    max_len = max(len(c) for c in coeffs)
    arr = np.zeros((len(coeffs), max_len))
    for i, c in enumerate(coeffs):
        arr[i, : len(c)] = c
    im = ax.imshow(arr, aspect="auto", cmap="viridis", interpolation="nearest")
    return im


def coefficient_heatmap(
    decompositions: List[WaveletDecomposition],
    tokens: List[str],
    save_path: Optional[str] = None,
    cmap: str = "RdBu_r",
    show: bool = False,
):
    """Plot one row per token showing flat concatenated wavelet coefficients."""
    fig, ax = plt.subplots(
        len(decompositions), 1, figsize=(14, 0.6 * len(decompositions)),
        squeeze=False,
    )
    flat = []
    max_len = 0
    for d in decompositions:
        parts = [d.approx.ravel()] + [c.ravel() for c in d.details]
        cat = np.concatenate(parts)
        flat.append(cat)
        max_len = max(max_len, cat.size)
    arr = np.zeros((len(flat), max_len))
    for i, f in enumerate(flat):
        arr[i, : f.size] = f
    arr = np.clip(arr, -np.percentile(np.abs(arr), 95),
                  np.percentile(np.abs(arr), 95))
    vmin = -np.max(np.abs(arr)) or -1
    vmax = np.max(np.abs(arr)) or 1
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    im = ax[0, 0].imshow(arr, aspect="auto", cmap=cmap, norm=norm,
                         interpolation="nearest")
    ax[0, 0].set_yticks(range(len(tokens)))
    ax[0, 0].set_yticklabels(tokens)
    ax[0, 0].set_xlabel("Wavelet coefficient index")
    ax[0, 0].set_title("Wavelet coefficient heatmap")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.025, pad=0.02)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def energy_distribution_bar(
    decompositions: List[WaveletDecomposition],
    tokens: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Stacked bar chart of normalised energy share across levels."""
    from analysis.energy import energy_distribution

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.85
    rows = []
    for d in decompositions:
        ed = energy_distribution(d)
        row = [ed.get(k, 0.0) for k in sorted(ed.keys())]
        rows.append(row)
    if not rows:
        plt.close(fig)
        return save_path
    max_n = max(len(r) for r in rows)
    arr = np.zeros((len(rows), max_n))
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    x = np.arange(len(tokens))
    bottom = np.zeros(len(tokens))
    levels = sorted(energy_distribution(decompositions[0]).keys())
    cmap = plt.cm.viridis(np.linspace(0, 1, max_n))
    for lvl in range(max_n):
        ax.bar(x, arr[:, lvl], width, bottom=bottom, color=cmap[lvl],
               label=f"L{lvl}")
        bottom += arr[:, lvl]
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_ylabel("Normalised energy share")
    ax.set_title("Per-level energy distribution")
    ax.legend(loc="upper right", ncol=max_n // 3 + 1, fontsize="small")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def tokens_energy_heatmap(
    decompositions: List[WaveletDecomposition],
    tokens: List[str],
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Matrix heatmap of energies (rows=tokens, cols=levels)."""
    from analysis.energy import energy_distribution

    rows = []
    max_n_levels = 1
    for d in decompositions:
        ed = energy_distribution(d)
        row = [ed.get(k, 0.0) for k in sorted(ed.keys())]
        max_n_levels = max(max_n_levels, len(row))
        rows.append(row)
    arr = np.zeros((len(rows), max_n_levels))
    for i, r in enumerate(rows):
        arr[i, : len(r)] = r
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(rows) + 1.5))
    im = ax.imshow(arr, aspect="auto", cmap="magma", interpolation="nearest")
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xticks(range(max_n_levels))
    ax.set_xticklabels(["approx"] + [f"D{j}" for j in range(max_n_levels - 1)],
                       rotation=45, ha="right")
    ax.set_title("Per-level energy heatmap across tokens")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path
