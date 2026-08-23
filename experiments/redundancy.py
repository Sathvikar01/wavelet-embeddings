"""Detect candidate redundant heads per the Phase-3 specification rules.

A head (A) is "redundant" relative to another head (B) when *all* thresholds
hold between them:

  * wavelet_similarity > thr_wav       (default 0.85)
  * entropy difference        < thr_e (default 0.25)
  * energy difference         < thr_E (default 0.15 * mean energy)
  * attention overlap         > thr_attn (top-k token overlap, default 0.60)

For every head we aggregate:

  * redundancy_score = mean(max wavelet sim) over heads above thresholds
  * redundancy partners = list(L, H)

We produce ``head_rank.csv`` listing every (layer, head) sorted descending
by redundancy_score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from attention import AttentionLoader


# --------------------------------------------------------------------------- #
# Data class
# --------------------------------------------------------------------------- #

@dataclass
class HeadRedundancy:
    layer: int
    head: int
    redundancy_score: float
    entropy: float
    energy: float
    sparsity: float
    compression_ratio: float
    partners: List[Tuple[int, int]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _top_k_overlap(A: np.ndarray, B: np.ndarray, k: int = 5,
                    per_query: str = "mean") -> float:
    """Mean fraction of shared top-k attended-to tokens between two heads.

    For every query-position, compute the set of top-attended keys and measure
    Jaccard overlap, then average across query positions.
    """
    if A.shape != B.shape:
        return 0.0
    T = A.shape[0]
    k = min(k, T)
    overlaps = []
    for i in range(T):
        a_top = set(np.argpartition(-A[i], k - 1)[:k].tolist())
        b_top = set(np.argpartition(-B[i], k - 1)[:k].tolist())
        union = a_top | b_top
        if not union:
            overlap = 0.0
        else:
            overlap = len(a_top & b_top) / len(union)
        overlaps.append(overlap)
    if not overlaps:
        return 0.0
    return float(np.mean(overlaps) if per_query == "mean" else np.max(overlaps))


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #

def compute_redundancy(
    loader: AttentionLoader,
    similarity_npy_path: Optional[str],
    rows: List[Dict[str, float]],
    wavelet_thr: float = 0.85,
    entropy_thr: float = 0.25,
    energy_relative_thr: float = 0.15,
    attention_overlap_thr: float = 0.60,
    top_k_attn: int = 5,
    save_dir: Optional[str] = None,
) -> List[HeadRedundancy]:
    """Compute redundancy score per head and optionally save CSV.

    ``similarity_npy_path`` should point to a ``similarity_wavelet.npy`` file
    produced by :mod:`experiments.head_similarity`. If absent, redundancy is
    computed using only the metrics CSV row-based proxy.
    """
    if not rows:
        return []
    n_total = loader.n_layers * loader.n_heads
    if similarity_npy_path and os.path.exists(similarity_npy_path):
        ws = np.load(similarity_npy_path)
    else:
        # Fallback: identity matrix -> no wavelet-based redundancy
        ws = np.eye(n_total, dtype=np.float32)
    # Build a metric dict
    metrics_map: Dict[Tuple[int, int], Dict[str, float]] = {
        (int(r["layer"]), int(r["head"])): r for r in rows
    }
    # Compute attention overlap matrix on demand
    mean_energy = float(np.mean([r["total_energy"] for r in rows]) or 1.0)
    redundancy: List[HeadRedundancy] = []
    for i, row_i in enumerate(rows):
        Li, Hi = int(row_i["layer"]), int(row_i["head"])
        max_score = 0.0
        partners: List[Tuple[int, int]] = []
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            Lj, Hj = int(row_j["layer"]), int(row_j["head"])
            sim = float(ws[i, j])
            dE = abs(row_i["total_energy"] - row_j["total_energy"])
            de = abs(row_i["shannon_entropy"] - row_j["shannon_entropy"])
            if (sim > wavelet_thr and dE < energy_relative_thr * mean_energy
                    and de < entropy_thr):
                # Compute attention-overlap lazily
                head_j = loader.load_head(Lj, Hj)
                head_i = loader.load_head(Li, Hi)
                ov = _top_k_overlap(head_i.normalized, head_j.normalized,
                                    k=top_k_attn)
                if ov >= attention_overlap_thr:
                    partner_score = (sim + ov) / 2.0
                    if partner_score > max_score:
                        max_score = partner_score
                    partners.append((Lj, Hj))
        redundancy.append(HeadRedundancy(
            layer=Li, head=Hi, redundancy_score=max_score,
            entropy=row_i["shannon_entropy"],
            energy=row_i["total_energy"],
            sparsity=row_i["gini_sparsity"],
            compression_ratio=row_i["compression_ratio_99"],
            partners=partners,
        ))
    # Sort by redundancy_score descending
    redundancy.sort(key=lambda r: -r.redundancy_score)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import csv
        with open(os.path.join(save_dir, "head_rank.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "layer", "head", "redundancy_score",
                "entropy", "energy", "sparsity", "compression_ratio",
                "n_partners",
            ])
            for r in redundancy:
                writer.writerow([
                    r.layer, r.head,
                    f"{r.redundancy_score:.4f}",
                    f"{r.entropy:.4f}", f"{r.energy:.5f}",
                    f"{r.sparsity:.4f}", f"{r.compression_ratio:.4f}",
                    len(r.partners),
                ])
    return redundancy
