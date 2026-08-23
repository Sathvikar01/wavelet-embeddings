"""Aggregate Phase-3 head metrics across all snapshots.

Walks ``results/attention_analysis/<snapshot>/wavelet_<w>/head_metrics.csv``
and writes two artefacts under ``results/attention_analysis/``:

  * phase3_summary_by_wavelet.csv  - mean of every metric per
    (model, wavelet), pooled over all sentences of that model.
  * phase3_summary_by_sentence.csv - per (snapshot, wavelet) means, so
    sentence-level spread stays visible.
"""

from __future__ import annotations

import os
import re
import glob

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results",
                     "attention_analysis")

METRICS = ["total_energy", "low_freq_energy", "high_freq_energy",
           "energy_ratio_low_high", "shannon_entropy", "spectral_entropy",
           "gini_sparsity", "reconstruction_error_30pct",
           "compression_ratio_99"]


def _parse_snapshot(name: str):
    m = re.match(r"([A-Za-z0-9\-\.]+)_(\d{2})_", name)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def main() -> None:
    rows_sentence, rows_model = [], []
    for csv_path in glob.glob(os.path.join(ROOT, "*", "wavelet_*",
                                            "head_metrics.csv")):
        snap_dir = os.path.basename(os.path.dirname(
            os.path.dirname(csv_path)))
        wavelet = os.path.basename(os.path.dirname(csv_path)).replace(
            "wavelet_", "")
        model, sid = _parse_snapshot(snap_dir)
        if model is None:
            continue
        df = pd.read_csv(csv_path)
        means = df[METRICS].mean()
        row = {"model": model, "sentence_id": sid, "wavelet": wavelet,
               "snapshot": snap_dir, **{k: round(float(v), 5)
                                         for k, v in means.items()}}
        rows_sentence.append(row)
        rows_model.append(row)

    by_sentence = pd.DataFrame(rows_sentence).sort_values(
        ["model", "sentence_id", "wavelet"])
    by_sentence.to_csv(os.path.join(ROOT, "phase3_summary_by_sentence.csv"),
                       index=False, encoding="utf-8")

    agg = by_sentence.groupby(["model", "wavelet"])[METRICS].agg(
        ["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.round(5).reset_index()
    agg.to_csv(os.path.join(ROOT, "phase3_summary_by_wavelet.csv"),
               index=False, encoding="utf-8")
    print(f"snapshots pooled: {by_sentence['snapshot'].nunique()}, "
          f"rows: {len(by_sentence)}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
