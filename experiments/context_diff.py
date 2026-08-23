"""Phase-2 core experiment: **wavelet spectrum delta across senses**.

For one (model, anchor) we have many contextual vectors spanning 2+ senses.
We ask:

  * How different are the wavelet spectra between senses?
  * Between occurrences within the same sense?
  * Which bands (approx vs details) carry the most inter-sense signal?

We compute, for every pair of contextual vectors:

  * cosine similarity (baseline)
  * **wavelet similarity** (concatenated coefficients)
  * per-level energy difference ``|E_i^A - E_i^B|``
  * energy-distance aggregations: ``d_low``, ``d_high``

Aggregates:

  * ``inter_sense_similarity`` -- mean sim between vectors that share a sense
  * ``cross_sense_similarity`` -- mean sim between vectors of different senses
  * ``separation`` -- inter_sense - cross_sense (positive = good)
  * ``mean_band_energy``      -- per (sense, level) energy summary
  * ``wavelet_spectrum_per_sense`` -- mean approx + per-level detail energies
                                       per sense

These directly answer the Phase-2 research question: *"How does the wavelet
spectrum change with context?  Do different meanings occupy different
frequency spectra?"*
"""

from __future__ import annotations

import itertools
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wavelets import make_decomposer, wavelet_similarity, WaveletDecomposer
from analysis.energy import (energy_summary, low_freq_energy,
                              high_freq_energy, total_energy)
from analysis.entropy import coefficient_entropy
from analysis.sparsity import gini_coefficient
from contextual.loader import LoadedAnchorContext


# --------------------------------------------------------------------------- #
# Per-anchor report
# --------------------------------------------------------------------------- #

@dataclass
class SpectrumDeltaReport:
    anchor: str
    wavelet: str
    n_examples: int
    senses: List[str]
    inter_sense_cosine: float            # within-sense cosine
    cross_sense_cosine: float             # across-sense cosine
    cosine_separation: float              # inter - cross

    inter_sense_wavelet: float            # within-sense wavelet sim
    cross_sense_wavelet: float             # across-sense wavelet sim
    wavelet_separation: float              # inter - cross

    inter_sense_low_energy: float
    cross_sense_low_energy: float         # mean abs delta of low-frequency energy
    inter_sense_high_energy: float
    cross_sense_high_energy: float

    # Per-sense spectrum summary
    senses_mean_total_energy: Dict[str, float] = field(default_factory=dict)
    senses_mean_low_energy: Dict[str, float] = field(default_factory=dict)
    senses_mean_high_energy: Dict[str, float] = field(default_factory=dict)
    senses_mean_entropy: Dict[str, float] = field(default_factory=dict)
    senses_mean_gini: Dict[str, float] = field(default_factory=dict)
    senses_mean_low_high_ratio: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #

def run_spectrum_delta(
    ctx: LoadedAnchorContext,
    decomposer_factory,
    save_dir: Optional[str] = None,
) -> SpectrumDeltaReport:
    """Compute inter/sense vs cross-sense similarity metrics for one anchor."""
    dec = decomposer_factory()
    decomps = dec.batch_decompose(ctx.vectors)

    n = len(decomps)
    senses = ctx.senses

    # Per-vector summaries
    total_e = np.array([total_energy(d) for d in decomps])
    low_e = np.array([low_freq_energy(d) for d in decomps])
    high_e = np.array([high_freq_energy(d) for d in decomps])
    ent = np.array([coefficient_entropy(d) for d in decomps])
    gini = np.array([gini_coefficient(d) for d in decomps])
    ratio = np.array([low_freq_energy(d) / max(high_freq_energy(d), 1e-12)
                      for d in decomps])

    # Pairwise similarity
    pairs_same_cos, pairs_diff_cos = [], []
    pairs_same_wav, pairs_diff_wav = [], []
    d_low_same, d_low_diff = [], []
    d_high_same, d_high_diff = [], []

    for i, j in itertools.combinations(range(n), 2):
        cs = _cosine(ctx.vectors[i], ctx.vectors[j])
        ws = wavelet_similarity(decomps[i], decomps[j])
        same = senses[i] == senses[j]
        (pairs_same_cos if same else pairs_diff_cos).append(cs)
        (pairs_same_wav if same else pairs_diff_wav).append(ws)
        d_lo = abs(low_e[i] - low_e[j])
        d_hi = abs(high_e[i] - high_e[j])
        (d_low_same if same else d_low_diff).append(d_lo)
        (d_high_same if same else d_high_diff).append(d_hi)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    # Per-sense aggregates
    sense_set = sorted(set(senses))
    se_te, se_le, se_he, se_e, se_g, se_r = {}, {}, {}, {}, {}, {}
    for s in sense_set:
        idxs = [i for i, ss in enumerate(senses) if ss == s]
        if not idxs:
            continue
        se_te[s] = float(np.mean(total_e[idxs]))
        se_le[s] = float(np.mean(low_e[idxs]))
        se_he[s] = float(np.mean(high_e[idxs]))
        se_e[s] = float(np.mean(ent[idxs]))
        se_g[s] = float(np.mean(gini[idxs]))
        se_r[s] = float(np.mean(ratio[idxs]))

    report = SpectrumDeltaReport(
        anchor=ctx.anchor,
        wavelet=dec.wavelet_name,
        n_examples=n,
        senses=sense_set,
        inter_sense_cosine=_mean(pairs_same_cos),
        cross_sense_cosine=_mean(pairs_diff_cos),
        cosine_separation=_mean(pairs_same_cos) - _mean(pairs_diff_cos),
        inter_sense_wavelet=_mean(pairs_same_wav),
        cross_sense_wavelet=_mean(pairs_diff_wav),
        wavelet_separation=_mean(pairs_same_wav) - _mean(pairs_diff_wav),
        inter_sense_low_energy=_mean(d_low_same),
        cross_sense_low_energy=_mean(d_low_diff),
        inter_sense_high_energy=_mean(d_high_same),
        cross_sense_high_energy=_mean(d_high_diff),
        senses_mean_total_energy=se_te,
        senses_mean_low_energy=se_le,
        senses_mean_high_energy=se_he,
        senses_mean_entropy=se_e,
        senses_mean_gini=se_g,
        senses_mean_low_high_ratio=se_r,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        import csv
        # --- Main summary CSV ---
        with open(os.path.join(save_dir, f"spectrum_delta.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "anchor", "wavelet", "n", "n_senses",
                "inter_cos", "cross_cos", "delta_cos",
                "inter_wav", "cross_wav", "delta_wav",
                "inter_dLow", "cross_dLow", "inter_dHigh", "cross_dHigh",
            ])
            writer.writerow([
                report.anchor, report.wavelet, report.n_examples, len(report.senses),
                f"{report.inter_sense_cosine:.4f}",
                f"{report.cross_sense_cosine:.4f}",
                f"{report.cosine_separation:.4f}",
                f"{report.inter_sense_wavelet:.4f}",
                f"{report.cross_sense_wavelet:.4f}",
                f"{report.wavelet_separation:.4f}",
                f"{report.inter_sense_low_energy:.4f}",
                f"{report.cross_sense_low_energy:.4f}",
                f"{report.inter_sense_high_energy:.4f}",
                f"{report.cross_sense_high_energy:.4f}",
            ])
        # --- Per-sense spectrum CSV ---
        with open(os.path.join(save_dir, f"per_sense_spectrum.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sense", "n_examples", "E_total", "E_low", "E_high",
                "entropy", "gini", "E_low/high",
            ])
            for s in sense_set:
                idxs = [i for i, ss in enumerate(senses) if ss == s]
                writer.writerow([
                    s, len(idxs),
                    f"{se_te[s]:.4f}", f"{se_le[s]:.4f}", f"{se_he[s]:.4f}",
                    f"{se_e[s]:.4f}", f"{se_g[s]:.4f}", f"{se_r[s]:.4f}",
                ])
        # --- Pairwise similarity CSV ---
        with open(os.path.join(save_dir, f"pairwise_similarities.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["i", "j", "sense_i", "sense_j", "same_sense",
                              "cosine", "wavelet", "dE_low", "dE_high"])
            for i, j in itertools.combinations(range(n), 2):
                writer.writerow([
                    i, j, senses[i], senses[j],
                    "Y" if senses[i] == senses[j] else "N",
                    f"{_cosine(ctx.vectors[i], ctx.vectors[j]):.4f}",
                    f"{wavelet_similarity(decomps[i], decomps[j]):.4f}",
                    f"{abs(low_e[i]-low_e[j]):.4f}",
                    f"{abs(high_e[i]-high_e[j]):.4f}",
                ])
    return report


# --------------------------------------------------------------------------- #
# Across-anchors aggregator
# --------------------------------------------------------------------------- #

def run_spectrum_delta_for_anchors(
    anchors_data: Dict[str, LoadedAnchorContext],
    decomposer_factory,
    save_root: Optional[str] = None,
) -> Dict[str, SpectrumDeltaReport]:
    reports: Dict[str, SpectrumDeltaReport] = {}
    for anchor, ctx in anchors_data.items():
        save_dir = os.path.join(save_root, anchor) if save_root else None
        reports[anchor] = run_spectrum_delta(ctx, decomposer_factory, save_dir=save_dir)
    # Aggregate summary
    if save_root:
        import csv
        os.makedirs(save_root, exist_ok=True)
        with open(os.path.join(save_root, "spectrum_delta_summary.csv"), "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "anchor", "wavelet", "n_examples", "n_senses",
                "cosine_separation", "wavelet_separation",
                "inter_wav", "cross_wav",
            ])
            for a, r in reports.items():
                writer.writerow([
                    a, r.wavelet, r.n_examples, len(r.senses),
                    f"{r.cosine_separation:.4f}",
                    f"{r.wavelet_separation:.4f}",
                    f"{r.inter_sense_wavelet:.4f}",
                    f"{r.cross_sense_wavelet:.4f}",
                ])
    return reports
