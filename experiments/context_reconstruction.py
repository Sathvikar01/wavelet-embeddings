"""Phase-2 contextual reconstruction experiment.

For every contextual vector we ask:

  * How robust is each sense's spectrum against wavelet compression?
  * Do high-frequency (sparse, "detail heavy") senses collapse faster?

We measure per-(anchor, sense):
  * mean cosine(orig, rec) across compression ratios
  * mean SNR
  * mean swap rate: how often the nearest-neighbour (sense-classified) flips
    after compression.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from analysis.compression import compress_embedding, DEFAULT_COMPRESSION_RATIOS
from contextual.loader import LoadedAnchorContext
from wavelets import make_decomposer, WaveletDecomposer


# --------------------------------------------------------------------------- #
# Per-vector compression results, aggregated per sense
# --------------------------------------------------------------------------- #

@dataclass
class SenseCompressionReport:
    sense: str
    n: int
    mean_cosines: List[float]            # per ratio
    mean_snrs: List[float]
    mean_drifts: List[float]
    inverse_swap_rate: List[float]


@dataclass
class AnchorCompressionReport:
    anchor: str
    wavelet: str
    ratios: List[float]
    per_sense: Dict[str, SenseCompressionReport]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_contextual_compression(
    ctx: LoadedAnchorContext,
    decomposer_factory,
    ratios: Sequence[float] = DEFAULT_COMPRESSION_RATIOS,
    save_dir: Optional[str] = None,
) -> AnchorCompressionReport:
    dec = decomposer_factory()
    n = len(ctx.vectors)
    senses = ctx.senses

    # Pre-compute峠 all reconstructions
    recs_per_ratio: Dict[float, List[np.ndarray]] = {r: [] for r in ratios}
    for i in range(n):
        res = compress_embedding(ctx.vectors[i], dec, ratios=ratios)
        for cr in res:
            recs_per_ratio[cr.ratio].append(cr.reconstructed)

    # Per-sense aggregation
    sense_set = sorted(set(senses))
    per_sense: Dict[str, SenseCompressionReport] = {}

    for s in sense_set:
        idxs = [i for i, ss in enumerate(senses) if ss == s]
        if not idxs:
            continue
        cosines, snrs, drifts = zip(*[
            (np.zeros(len(ratios)), np.zeros(len(ratios)), np.zeros(len(ratios)))
        ])  # placeholder shapes
        cos_mean = np.zeros(len(ratios))
        snr_mean = np.zeros(len(ratios))
        dri_mean = np.zeros(len(ratios))
        swap_rate = np.zeros(len(ratios))
        for ri, r in enumerate(ratios):
            cs, ss_, drs = [], [], []
            for i in idxs:
                rec = recs_per_ratio[r][i]
                cs.append(_cosine(ctx.vectors[i], rec))
                drift = float(np.linalg.norm(ctx.vectors[i] - rec))
                ss_.append(20 * np.log10((np.linalg.norm(ctx.vectors[i]) or 1e-12)
                                          / (drift or 1e-12)))
                drs.append(drift)
            cos_mean[ri] = float(np.mean(cs))
            snr_mean[ri] = float(np.mean(ss_))
            dri_mean[ri] = float(np.mean(drs))
            # Inverse swap rate: fraction of within-sense nearest neighbours
            # still recovered from the compressed matrix.
            if len(idxs) >= 2:
                sub_orig = ctx.vectors[idxs]
                sub_rec = np.stack([recs_per_ratio[r][i] for i in idxs])
                orig_nn = _nearest_within(sub_orig)
                rec_nn = _nearest_within(sub_rec)
                swap_rate[ri] = float(np.mean(orig_nn == rec_nn))
            else:
                swap_rate[ri] = 1.0
        per_sense[s] = SenseCompressionReport(
            sense=s, n=len(idxs),
            mean_cosines=cos_mean.tolist(),
            mean_snrs=snr_mean.tolist(),
            mean_drifts=dri_mean.tolist(),
            inverse_swap_rate=swap_rate.tolist(),
        )

    rep = AnchorCompressionReport(
        anchor=ctx.anchor, wavelet=dec.wavelet_name,
        ratios=list(ratios), per_sense=per_sense,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import csv
        with open(os.path.join(save_dir, "contextual_compression.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["anchor", "wavelet", "sense", "n",
                        "ratio", "cos", "snr_db", "drift", "inv_swap_rate"])
            for s, sr in per_sense.items():
                for k in range(len(ratios)):
                    w.writerow([rep.anchor, rep.wavelet, s, sr.n,
                                f"{rep.ratios[k]:.2f}",
                                f"{sr.mean_cosines[k]:.4f}",
                                f"{sr.mean_snrs[k]:.3f}",
                                f"{sr.mean_drifts[k]:.4f}",
                                f"{sr.inverse_swap_rate[k]:.3f}"])
        # Plot cosine curves per sense
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for s, sr in per_sense.items():
            ax.plot(rep.ratios, sr.mean_cosines, marker="o", label=f"{s} (n={sr.n})")
        ax.set_xlabel("Compression ratio (smallest details zeroed)")
        ax.set_ylabel("Mean cosine(orig, reconstructed)")
        ax.set_title(f"Contextual compression by sense ({rep.anchor}, {rep.wavelet})")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "compression_per_sense.png"), dpi=140)
        plt.close(fig)
    return rep


def _nearest_within(mat: np.ndarray) -> np.ndarray:
    """For each row, return the index (relative to mat) of nearest other row."""
    n = mat.shape[0]
    out = np.zeros(n, dtype=int)
    norms = np.linalg.norm(mat, axis=1)
    if (norms == 0).any() or n < 2:
        return out
    normed = mat / norms[:, None]
    sim = normed @ normed.T
    np.fill_diagonal(sim, -np.inf)
    return sim.argmax(axis=1)
