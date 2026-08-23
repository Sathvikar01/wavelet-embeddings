"""Compare wavelet signatures across token categories (Nouns vs Verbs vs ...).

For each category present in the probe set we compute:

  * mean energy (approx / low / high)
  * mean entropy
  * mean skewness / kurtosis
  * mean compression ratio
  * distribution of energy across levels

Also computes neighbour analysis on the king/queen/man/woman/apple/orange
tokens using *wavelet similarity* vs *cosine similarity*.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import itertools

from embeddings.loader import EmbeddingLoader, NEIGHBOR_TOKENS
from wavelets.base import WaveletDecomposer, wavelet_similarity
from analysis.energy import (
    batch_energy_summary,
    energy_summary,
    energy_distribution,
)
from analysis.entropy import batch_entropy_summary
from analysis.sparsity import batch_sparsity_summary
from analysis.compression import compress_embedding


# --------------------------------------------------------------------------- #
# Category comparison
# --------------------------------------------------------------------------- #

@dataclass
class CategoryStats:
    category: str
    n_tokens: int
    mean_total_energy: float
    mean_approx_energy: float
    mean_low_freq_energy: float
    mean_high_freq_energy: float
    mean_energy_ratio_low_high: float
    mean_entropy: float
    mean_skewness: float
    mean_kurtosis: float
    mean_gini: float
    mean_compression_ratio: float
    mean_information_score: float           # entropy * energy (normalised)


def _information_score(entropy: float, energy: float) -> float:
    """A simple Information Score: entropy × log(energy)."""
    return float(entropy * np.log1p(max(energy, 0.0)))


def run_token_comparison(
    loader: EmbeddingLoader,
    decomposer_factory,
    probe_tokens: Optional[List[str]] = None,
    save_dir: str | None = None,
) -> Dict[str, CategoryStats]:
    """Per-category comparison."""
    if probe_tokens is None:
        from embeddings.loader import DEFAULT_PROBE_TOKENS
        probe_tokens = DEFAULT_PROBE_TOKENS
    decomposer = decomposer_factory()

    indexed = loader.get_many(probe_tokens)
    if not indexed:
        return {}
    X = np.stack([ie.vector for ie in indexed])
    decomps = decomposer.batch_decompose(X)

    e_sum = batch_energy_summary(decomps)
    ent_sum = batch_entropy_summary(decomps)
    sp_sum = batch_sparsity_summary(decomps)
    # Per-token information score
    info_scores = [
        _information_score(e, ev)
        for e, ev in zip(ent_sum["global_entropy"], e_sum["total"])
    ]

    by_cat: Dict[str, List[int]] = defaultdict(list)
    for i, ie in enumerate(indexed):
        by_cat[ie.category].append(i)

    out: Dict[str, CategoryStats] = {}
    for cat, idxs in by_cat.items():
        e_total = float(np.mean(e_sum["total"][idxs]))
        e_approx = float(np.mean(e_sum["approx"][idxs]))
        e_low = float(np.mean(e_sum["low"][idxs]))
        e_high = float(np.mean(e_sum["high"][idxs]))
        out[cat] = CategoryStats(
            category=cat,
            n_tokens=len(idxs),
            mean_total_energy=e_total,
            mean_approx_energy=e_approx,
            mean_low_freq_energy=e_low,
            mean_high_freq_energy=e_high,
            mean_energy_ratio_low_high=float(np.mean(e_sum["ratio_low_high"][idxs])),
            mean_entropy=float(np.mean(ent_sum["global_entropy"][idxs])),
            mean_skewness=float(np.mean(ent_sum["skewness"][idxs])),
            mean_kurtosis=float(np.mean(ent_sum["kurtosis"][idxs])),
            mean_gini=float(np.mean(sp_sum["gini"][idxs])),
            mean_compression_ratio=float(np.mean(sp_sum["compression_ratio"][idxs])),
            mean_information_score=float(np.mean([info_scores[i] for i in idxs])),
        )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import csv
        with open(os.path.join(save_dir, "token_category_comparison.csv"),
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "category", "n", "E_total", "E_approx", "E_low", "E_high",
                "E_low/high", "entropy", "skewness", "kurtosis", "gini",
                "compression_ratio", "info_score",
            ])
            for c, s in sorted(out.items(), key=lambda kv: -kv[1].mean_total_energy):
                writer.writerow([
                    c, s.n_tokens,
                    f"{s.mean_total_energy:.4f}", f"{s.mean_approx_energy:.4f}",
                    f"{s.mean_low_freq_energy:.4f}", f"{s.mean_high_freq_energy:.4f}",
                    f"{s.mean_energy_ratio_low_high:.3f}", f"{s.mean_entropy:.4f}",
                    f"{s.mean_skewness:.3f}", f"{s.mean_kurtosis:.3f}",
                    f"{s.mean_gini:.3f}", f"{s.mean_compression_ratio:.3f}",
                    f"{s.mean_information_score:.3f}",
                ])
        # Bar chart of category energy ratio
        fig, ax = plt.subplots(figsize=(10, 5))
        cats = sorted(out.keys())
        ax.bar(cats, [out[c].mean_energy_ratio_low_high for c in cats])
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylabel("E_low / E_high")
        ax.set_title(f"Low/high frequency energy ratio per token category "
                     f"({decomposer.wavelet_name})")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "category_energy_ratio.png"), dpi=140)
        plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Neighbour analysis (wavelet vs cosine similarity)
# --------------------------------------------------------------------------- #

def run_neighbour_analysis(
    loader: EmbeddingLoader,
    decomposer: WaveletDecomposer,
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """For every pair of tokens in the neighbour probe set, compute both
    cosine similarity and wavelet similarity.
    """
    if tokens is None:
        tokens = NEIGHBOR_TOKENS
    tokens_present = [t for t in tokens if t in loader.tokens]
    if len(tokens_present) < 2:
        return {}
    vecs = np.stack([loader.embeddings[loader.index_of(t)] for t in tokens_present])

    # Cosine similarity
    def cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

    # Pre-decompose
    decomps = decomposer.batch_decompose(vecs)

    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for i, j in itertools.combinations(range(len(tokens_present)), 2):
        a = tokens_present[i]
        b = tokens_present[j]
        cs = cos(vecs[i], vecs[j])
        ws = wavelet_similarity(decomps[i], decomps[j])
        out[(a, b)] = {"cosine": cs, "wavelet": ws}

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        import csv
        with open(save_path.replace(".png", ".csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token_a", "token_b", "cosine_sim", "wavelet_sim"])
            for (a, b), s in out.items():
                writer.writerow([a, b, f"{s['cosine']:.4f}", f"{s['wavelet']:.4f}"])
        # Scatter
        fig, ax = plt.subplots(figsize=(7, 6))
        xs = [s["cosine"] for s in out.values()]
        ys = [s["wavelet"] for s in out.values()]
        ax.scatter(xs, ys, alpha=0.7)
        for (a, b), s in out.items():
            ax.annotate(f"{a}-{b}", (s["cosine"], s["wavelet"]), fontsize=7)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Wavelet similarity")
        ax.set_title(f"Semantic neighbour similarity comparison "
                     f"({decomposer.wavelet_name})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
    return out
