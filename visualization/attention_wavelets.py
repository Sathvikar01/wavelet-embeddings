"""Per-head wavelet coefficient image stacks & frequency-distribution plots.

A "stack" is a single figure per attention head showing, side by side:

  * the original (row-normalised) attention matrix,
  * each 2-D DWT subband at every level (cAH / cAV / cAD), normalised to
    its own max magnitude,
  * the per-level energy share as a slim bar column.

This makes the multiscale structure that
``attention.analyzer.compute_head_metrics`` summarises numerically visible
for individual heads.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from attention import (
    AttentionLoader, AttentionWaveletDecomposer, HeadDecomposition,
)


def plot_head_wavelet_stack(
    A: np.ndarray,
    decomp: HeadDecomposition,
    title: str = "",
    save_dir: Optional[str] = None,
    basename: str = "wavelet_stack",
    dpi: int = 160,
) -> Optional[str]:
    """Render original attention + all DWT subbands + energy share."""
    levels = sorted(decomp.details.keys())
    n_sub = 3 * len(levels) + 1  # + original
    n_bars = len(levels)
    width_ratios = [1.6] * n_sub + [0.35] * n_bars
    fig, axes = plt.subplots(
        1, n_sub + n_bars,
        gridspec_kw={"width_ratios": width_ratios},
        figsize=(1.5 * (n_sub + n_bars), 2.4),
    )
    ims = []
    im = axes[0].imshow(A, cmap="viridis", aspect="equal")
    axes[0].set_title("attention", fontsize=8)
    ims.append((axes[0], im))
    col = 1
    for lvl in levels:
        for key in ("cAH", "cAV", "cAD"):
            c = np.abs(decomp.details[lvl][key])
            c = c / c.max() if c.max() > 0 else c
            im = axes[col].imshow(c, cmap="magma", aspect="equal")
            axes[col].set_title(f"L{lvl}-{key}", fontsize=7)
            ims.append((axes[col], im))
            col += 1
    tot = decomp.energy_approx + sum(decomp.energy_per_level.values())
    shares = ([decomp.energy_approx / tot if tot else 0.0]
              + [decomp.energy_per_level[l] / tot if tot else 0.0
                 for l in levels])
    labels = ["cA"] + [f"L{l}" for l in levels]
    axes[-n_bars].barh(range(len(shares)), shares,
                        color=plt.cm.viridis(np.linspace(0.15, 0.9,
                                                          len(shares))))
    axes[-n_bars].set_yticks(range(len(shares)), labels=labels, fontsize=7)
    for ax in axes[1:-n_bars]:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes:
        for s in ax.spines.values():
            s.set_visible(False)
    if title:
        fig.suptitle(title, fontsize=9)
    out_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{basename}.png")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_stacks_for_heads(
    loader: AttentionLoader,
    decomposer_factory,
    heads: Sequence[tuple],
    save_dir: str,
) -> list:
    """Save a wavelet stack figure for each ``(layer, head)`` in ``heads``."""
    paths = []
    dec = decomposer_factory()
    for L, H in heads:
        head = loader.load_head(L, H)
        d = dec.decompose(head.normalized)
        p = plot_head_wavelet_stack(
            head.normalized, d,
            title=f"{getattr(loader, 'model_key', '')} layer{L} head{H}",
            save_dir=os.path.join(save_dir, f"layer{L}_head{H}"),
            basename="wavelet_stack",
        )
        if p:
            paths.append(p)
    return paths
