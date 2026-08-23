"""Phase-2 visualisations: meaning-evolution heatmaps & per-sense spectra.

Functions
---------
* ``delta_heatmap``   - difference between two contextual vectors' wavelet coeffs
* ``per_sense_spectrum_bars`` - bar chart of energy per sense & band
* ``meaning_evolution_plot`` - 2D projection of contextual vectors colored by sense
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from wavelets import make_decomposer, wavelet_similarity, WaveletDecomposer
from wavelets.base import WaveletDecomposition
from contextual.loader import LoadedAnchorContext
from analysis.energy import energy_distribution, total_energy


def _flatten_coeffs(d: WaveletDecomposition) -> np.ndarray:
    parts = [d.approx.ravel()] + [c.ravel() for c in d.details]
    return np.concatenate(parts)


def delta_heatmap(
    decomps_a: List[WaveletDecomposition],
    decomps_b: List[WaveletDecomposition],
    title: str,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Heatmap of |cA_i - cA_j| and |cD_i - cD_j| flattened across levels."""
    assert len(decomps_a) == len(decomps_b)
    flat_a = [_flatten_coeffs(d) for d in decomps_a]
    flat_b = [_flatten_coeffs(d) for d in decomps_b]
    max_len = max(len(f) for f in flat_a + flat_b)
    arr = np.zeros((len(flat_a), max_len))
    for i, (a, b) in enumerate(zip(flat_a, flat_b)):
        n = min(len(a), len(b))
        arr[i, :n] = np.abs(a[:n] - b[:n])
    vmax = np.nanpercentile(arr, 95) or 1.0
    fig, ax = plt.subplots(figsize=(14, 0.5 * len(flat_a) + 1.2))
    im = ax.imshow(arr, aspect="auto", cmap="magma",
                   interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_ylabel("pair index")
    ax.set_xlabel("Wavelet coefficient index (approx | details)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def per_sense_spectrum_bars(
    ctx: LoadedAnchorContext,
    decomposer_factory,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Stacked bar chart: per (sense) - normalised energy in each band."""
    dec = decomposer_factory()
    decomps = dec.batch_decompose(ctx.vectors)
    senses = ctx.senses
    sense_set = sorted(set(senses))

    # Build per-sense mean energy distribution
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / max(1, len(sense_set))
    x = np.arange(len(sense_set))

    # Compute per-sense averaged energy distribution
    pdf_rows = []
    for s in sense_set:
        idxs = [i for i, ss in enumerate(senses) if ss == s]
        if not idxs:
            continue
        agg = None
        for k in idxs:
            ed = energy_distribution(decomps[k])
            if agg is None:
                agg = np.array([ed.get(j, 0.0) for j in sorted(ed.keys())])
            else:
                arr2 = np.array([ed.get(j, 0.0) for j in sorted(ed.keys())])
                if arr2.shape[0] > agg.shape[0]:
                    arr2 = arr2[: agg.shape[0]]
                elif arr2.shape[0] < agg.shape[0]:
                    pad = np.zeros(agg.shape[0])
                    pad[: arr2.shape[0]] = arr2
                    arr2 = pad
                agg = agg + arr2
        agg = agg / max(1, len(idxs))
        pdf_rows.append((s, agg))

    max_bands = max(r.shape[0] for _, r in pdf_rows) if pdf_rows else 0
    cmap = plt.cm.viridis(np.linspace(0, 1, max_bands))
    for i, (s, agg) in enumerate(pdf_rows):
        bottom = np.zeros(1)
        for b in range(agg.shape[0]):
            ax.bar(i, agg[b], width, bottom=bottom[0] if bottom.size else 0.0,
                   color=cmap[b])
            if bottom.size == 0:
                bottom = np.array([agg[b]])
            else:
                bottom[0] += agg[b]
    ax.set_xticks(range(len(pdf_rows)))
    ax.set_xticklabels([s for s, _ in pdf_rows], rotation=20, ha="right")
    ax.set_ylabel("Normalised energy share")
    ax.set_title(f"Per-sense wavelet spectrum ({ctx.anchor}, {dec.wavelet_name})")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def meaning_evolution_projection(
    ctx: LoadedAnchorContext,
    method: str = "tsne",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """2D scatter of contextual vectors coloured by sense."""
    from visualization.tsne import embed_2d, scatter_2d
    if ctx.n < 3:
        return None
    coords = embed_2d(ctx.vectors, method=method)
    title = title or f"Contextual embedding projection: '{ctx.anchor}' ({method})"
    scatter_2d(coords, ctx.senses, title=title, save_path=save_path, show=show)
