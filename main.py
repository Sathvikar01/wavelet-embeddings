"""Wavelet Analysis of Transformer Embeddings -- entry point.

Phase 1: static embedding wavelet analysis
Phase 2: contextual-embedding wavelet analysis (per-token hidden states
         across contexts, comparing spectra across meanings).
Phase 3: attention-matrix wavelet analysis (per-(layer, head) multiscale
         signatures, redundancy detection, head clustering, layer-progression,
         comparison across models).
Phase 4: predictive pruning -- do Phase-3 wavelet metrics predict
         redundant heads better than baselines (entropy / weight /
         magnitude / random)? Measures per-head ablation effect on the
         forward pass and computes rank correlations.

Usage
-----
    # Phase 1
    python main.py extract --model bert-base
    python main.py analyze --models bert-base distilbert gpt2 --wavelet db4
    python main.py all --models bert-base distilbert gpt2 \
        --wavelet db4 --sample 1000 --out results

    # Phase 2
    python main.py context-extract --models bert-base distilbert gpt2
    python main.py context-analyze --models bert-base --wavelet db4
    python main.py context-all --models bert-base distilbert gpt2 \
        --wavelet db4 --anchors bank plant match

    # Phase 3
    python main.py attention-extract --models bert-base distilbert gpt2
    python main.py attention-analyze --models bert-base --wavelet db4
    python main.py attention-all --models bert-base distilbert gpt2 \
        --wavelet db4

    # Phase 4
    python main.py pruning-analyze --models distilbert --wavelet db4
    python main.py pruning-all --models bert-base distilbert gpt2 \
        --wavelet db4

Each stage persists its outputs to ``results/<model>/<wavelet>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

# Make the project importable regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# Helper factories
# --------------------------------------------------------------------------- #

DEFAULT_MODELS = ["bert-base", "distilbert", "gpt2"]
DEFAULT_WAVELETS = ["haar", "db4", "sym4", "coif2"]


def _fix_wavelet_name(name: str) -> str:
    """Accept common synonyms used on the command line."""
    name = name.lower().strip()
    aliases = {
        "symlet": "sym4", "sym": "sym4", "sym8": "sym4",
        "coiflet": "coif2", "coif": "coif2",
        "daubechies": "db4", "db": "db4",
    }
    return aliases.get(name, name)


def _default_results_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --------------------------------------------------------------------------- #
# Stage: extract embeddings
# --------------------------------------------------------------------------- #

def cmd_extract(args: argparse.Namespace):
    from embeddings.extract import extract_and_save, MODEL_REGISTRY

    models = args.models or DEFAULT_MODELS
    out_dir = os.path.join(args.data_dir or _default_results_dir(), "embeddings")
    for m in models:
        if m not in MODEL_REGISTRY:
            print(f"[skip] Unknown model '{m}'. Available: {list(MODEL_REGISTRY)}")
            continue
        print(f"[extract] {m} -> {out_dir}")
        path = extract_and_save(m, out_dir)
        print(f"    saved: {path}")


def _loaders_for_models(models: List[str], data_dir: str) -> Dict[str, "EmbeddingLoader"]:
    from embeddings.loader import EmbeddingLoader
    out: Dict[str, "EmbeddingLoader"] = {}
    base = os.path.join(data_dir, "embeddings") if not data_dir.endswith("embeddings") else data_dir
    for m in models:
        path = os.path.join(base, f"{m}_embeddings.npz")
        if not os.path.exists(path):
            print(f"[warn] Missing embeddings file: {path}; call `extract` first.")
            continue
        out[m] = EmbeddingLoader(path)
    return out


# --------------------------------------------------------------------------- #
# Stage: analysis
# --------------------------------------------------------------------------- #

def cmd_analyze(args: argparse.Namespace):
    models = args.models or DEFAULT_MODELS
    wavelets = args.wavelets or DEFAULT_WAVELETS
    wavelets = [_fix_wavelet_name(w) for w in wavelets]
    out_root = args.out or _default_results_dir()
    data_dir = args.data_dir or _default_results_dir()

    loaders = _loaders_for_models(models, data_dir)
    if not loaders:
        print("No embeddings found. Run `python main.py extract` first.")
        return

    import json
    from wavelets import make_decomposer
    from experiments.compare_models import run_model_comparison
    from experiments.compare_tokens import (
        run_token_comparison, run_neighbour_analysis,
    )
    from experiments.reconstruction import run_reconstruction_experiment
    from visualization.heatmaps import (
        coefficient_heatmap, energy_distribution_bar, tokens_energy_heatmap,
    )
    from visualization.spectrum import wavelet_spectrum
    from visualization.tsne import before_after
    from analysis.compression import compress_batch, DEFAULT_COMPRESSION_RATIOS

    for wname in wavelets:
        print(f"\n=== Wavelet: {wname} ===")
        # Per-wavelet per-results subdir:
        # results/<model>/<wavelet>/
        wdir_root = os.path.join(out_root, "wavelet_" + wname)
        os.makedirs(wdir_root, exist_ok=True)

        # 1. Model comparison
        print("[exp] compare_models ...")
        model_stats = run_model_comparison(
            loaders,
            decomposer_factory=lambda w=wname: make_decomposer(w),
            sample_size=args.sample,
            save_dir=os.path.join(wdir_root, "compare_models"),
        )
        for name, s in model_stats.items():
            print(f"   {name}: E={s.mean_energy:.2f}, S={s.mean_skewness:.3f}, "
                  f"K={s.mean_kurtosis:.3f}, Gini={s.mean_gini:.3f}, "
                  f"CR={s.mean_compression_ratio:.2f}")

        # 2. Token-category comparison + neighbour analysis on each model
        for name, loader in loaders.items():
            print(f"[exp] compare_tokens on {name} ...")
            wdir_model = os.path.join(wdir_root, name)
            os.makedirs(wdir_model, exist_ok=True)
            cat_stats = run_token_comparison(
                loader,
                decomposer_factory=lambda w=wname: make_decomposer(w),
                save_dir=wdir_model,
            )
            for c, s in cat_stats.items():
                print(f"   {c:<10} n={s.n_tokens:3d} E_total={s.mean_total_energy:8.2f} "
                      f"E_low/high={s.mean_energy_ratio_low_high:6.2f} "
                      f"entropy={s.mean_entropy:5.3f} info_score={s.mean_information_score:6.2f}")

            # Neighbour analysis (king vs queen vs man vs woman ...)
            print(f"[exp] neighbour_analysis on {name} ...")
            run_neighbour_analysis(
                loader, make_decomposer(wname),
                save_path=os.path.join(wdir_model, "neighbour_analysis.png"),
            )

            # 3. Reconstruction experiment on a small probe set
            print(f"[exp] reconstruction on {name} ...")
            run_reconstruction_experiment(
                loader,
                decomposer_factory=lambda w=wname: make_decomposer(w),
                save_dir=os.path.join(wdir_model, "reconstruction"),
            )

            # 4. Visualisations: coefficient heatmap + per-token energy + spectrum
            print(f"[exp] visualisations on {name} ...")
            from embeddings.loader import DEFAULT_PROBE_TOKENS
            tok_present = [t for t in DEFAULT_PROBE_TOKENS if t in loader.tokens][:20]
            if tok_present:
                vecs = np.stack([loader.embeddings[loader.index_of(t)] for t in tok_present])
                dec = make_decomposer(wname)
                decomps = dec.batch_decompose(vecs)
                coefficient_heatmap(
                    decomps, tok_present,
                    save_path=os.path.join(wdir_model, "coefficient_heatmap.png"),
                )
                energy_distribution_bar(
                    decomps, tok_present,
                    save_path=os.path.join(wdir_model, "energy_distribution.png"),
                )
                tokens_energy_heatmap(
                    decomps, tok_present,
                    save_path=os.path.join(wdir_model, "tokens_energy_heatmap.png"),
                )
                # Spectrum for the first probe token only
                wavelet_spectrum(
                    decomps[0], title=f"{tok_present[0]} | {wname} | {name}",
                    save_path=os.path.join(wdir_model, f"spectrum_{tok_present[0]}.png"),
                )

            # 5. t-SNE / PCA before-after compression (sample 300 tokens)
            print(f"[exp] t-SNE before/after on {name} ...")
            rng = np.random.default_rng(0)
            n = min(300, loader.vocab_size)
            idx = rng.choice(loader.vocab_size, size=n, replace=False)
            X = loader.embeddings[idx]
            recs, _ = compress_batch(
                X, make_decomposer(wname),
                ratios=(0.30,),     # use 30% compression for the figure
            )
            before_after(
                X, recs[0.30],
                labels=[loader.tokens[i] for i in idx],
                method="tsne",
                save_path=os.path.join(wdir_model, "tsne_before_after.png"),
            )
            before_after(
                X, recs[0.30],
                labels=[loader.tokens[i] for i in idx],
                method="pca",
                save_path=os.path.join(wdir_model, "pca_before_after.png"),
            )

            # 6. Per-token information ranking (the end-goal metric)
            print(f"[exp] information ranking on {name} ...")
            from analysis.energy import total_energy
            from analysis.entropy import coefficient_entropy
            rank_indices = rng.choice(
                loader.vocab_size, size=min(args.sample, loader.vocab_size),
                replace=False,
            )
            dec = make_decomposer(wname)
            rows = []
            for i in rank_indices:
                d = dec.decompose(loader.embeddings[i])
                e = total_energy(d)
                h = coefficient_entropy(d)
                rows.append((loader.tokens[i], i, float(e), float(h),
                             float(h * np.log1p(e))))
            rows.sort(key=lambda r: -r[-1])
            with open(os.path.join(wdir_model, "information_ranking.csv"), "w",
                      encoding="utf-8") as f:
                f.write("token,index,energy,entropy,info_score\n")
                for t, i, e, h, score in rows[:200]:
                    f.write(f"{t},{i},{e:.4f},{h:.4f},{score:.4f}\n")

    print("\nDone. Results written to:", out_root)


# --------------------------------------------------------------------------- #
# Single-end entry
# --------------------------------------------------------------------------- #

def cmd_all(args: argparse.Namespace):
    cmd_extract(args)
    cmd_analyze(args)


# --------------------------------------------------------------------------- #
# Phase 2: Contextual embeddings
# --------------------------------------------------------------------------- #

def cmd_context_extract(args: argparse.Namespace):
    """Pull per-token contextual vectors for every (model, anchor) pair."""
    from contextual.data import POLYSEMY_DATASET, DEFAULT_ANCHORS
    from contextual.extractor import extract_and_save_anchor

    models = args.models or DEFAULT_MODELS
    anchors = args.anchors or DEFAULT_ANCHORS
    out_dir = os.path.join(args.data_dir or _default_results_dir(), "contexts")
    os.makedirs(out_dir, exist_ok=True)

    for m in models:
        for anchor in anchors:
            examples = POLYSEMY_DATASET.get(anchor)
            if not examples:
                print(f"[skip] No dataset entries for anchor '{anchor}'")
                continue
            print(f"[context-extract] model={m} anchor={anchor} "
                  f"n_examples={len(examples)}")
            path = extract_and_save_anchor(m, anchor, examples, out_dir)
            if path is None:
                print("    WARN: no anchor found in any sentence")
            else:
                print(f"    saved: {path}")


def cmd_context_analyze(args: argparse.Namespace):
    """Run all Phase-2 experiments on the saved contextual .npz files."""
    from contextual.data import POLYSEMY_DATASET, DEFAULT_ANCHORS, DEFAULT_PAIRS
    from contextual.loader import load_all_anchors
    from experiments.context_diff import run_spectrum_delta_for_anchors
    from experiments.context_reconstruction import run_contextual_compression
    from visualization.context_heatmaps import (
        delta_heatmap, per_sense_spectrum_bars, meaning_evolution_projection,
    )
    from wavelets import make_decomposer

    models = args.models or DEFAULT_MODELS
    wavelets = args.wavelets or DEFAULT_WAVELETS
    wavelets = [_fix_wavelet_name(w) for w in wavelets]
    anchors = args.anchors or DEFAULT_ANCHORS
    out_root = args.out or _default_results_dir()
    data_dir = args.data_dir or _default_results_dir()
    contexts_dir = os.path.join(data_dir, "contexts")

    for m in models:
        anchors_data = load_all_anchors(m, anchors, contexts_dir)
        if not anchors_data:
            print(f"[context-analyze] No contextual .npz found for model '{m}' "
                  f"under {contexts_dir}. Run `context extract` first.")
            continue
        print(f"[context-analyze] {m}: loaded {len(anchors_data)} anchors "
              f"(anchors: {list(anchors_data)})")
        for wname in wavelets:
            print(f"\n  --- Wavelet: {wname} ---")
            wdir_root = os.path.join(out_root, "contextual",
                                     m, "wavelet_" + wname)

            # 1. Spectrum delta across meanings
            print("  [exp] spectrum_delta_for_anchors ...")
            reports = run_spectrum_delta_for_anchors(
                anchors_data,
                decomposer_factory=lambda w=wname: make_decomposer(w),
                save_root=wdir_root,
            )
            for a, r in reports.items():
                print(f"    {a:<8}  senses={len(r.senses)} "
                      f"cos_sep={r.cosine_separation:+.4f} "
                      f"wav_sep={r.wavelet_separation:+.4f} "
                      f"inter_wav={r.inter_sense_wavelet:.4f} "
                      f"cross_wav={r.cross_sense_wavelet:.4f}")

            # 2. Contextual compression per sense for each anchor
            for a, ctx in anchors_data.items():
                wdir_a = os.path.join(wdir_root, a)
                print(f"  [exp] contextual_compression anchor={a} ...")
                run_contextual_compression(
                    ctx,
                    decomposer_factory=lambda w=wname: make_decomposer(w),
                    save_dir=os.path.join(wdir_a, "compression"),
                )

                # 3. Per-sense spectrum bar chart
                print(f"  [viz] per_sense_spectrum_bars a={a} ...")
                per_sense_spectrum_bars(
                    ctx,
                    decomposer_factory=lambda w=wname: make_decomposer(w),
                    save_path=os.path.join(wdir_a, "per_sense_spectrum.png"),
                )

                # 4. Meaning evolution projection (t-SNE)
                print(f"  [viz] meaning_evolution_projection a={a} ...")
                meaning_evolution_projection(
                    ctx, method="tsne",
                    title=f"Meaning evolution: '{a}' ({m}, {wname})",
                    save_path=os.path.join(wdir_a, "meaning_evolution_tsne.png"),
                )
                meaning_evolution_projection(
                    ctx, method="pca",
                    title=f"Meaning evolution: '{a}' ({m}, {wname})",
                    save_path=os.path.join(wdir_a, "meaning_evolution_pca.png"),
                )

                # 5. Delta-heatmaps for hardcoded contrast pairs
                decomposer = make_decomposer(wname)
                # Take the first example of two different senses
                from collections import defaultdict
                by_sense = defaultdict(list)
                for i, s in enumerate(ctx.senses):
                    by_sense[s].append(i)
                if len(by_sense) >= 2:
                    sense_a, sense_b = sorted(by_sense)[:2]
                    idxs_a = by_sense[sense_a]
                    idxs_b = by_sense[sense_b]
                    decs_a = [decomposer.decompose(ctx.vectors[i]) for i in idxs_a]
                    decs_b = [decomposer.decompose(ctx.vectors[i]) for i in idxs_b]
                    # Pad shorter list
                    if len(decs_a) != len(decs_b):
                        k = min(len(decs_a), len(decs_b))
                        decs_a, decs_b = decs_a[:k], decs_b[:k]
                    if decs_a:
                        delta_heatmap(
                            decs_a, decs_b,
                            title=f"|c_i({sense_a}) - c_i({sense_b})|  {a} | {m} | {wname}",
                            save_path=os.path.join(
                                wdir_a, f"delta_heatmap_{sense_a}_vs_{sense_b}.png"),
                        )

    print("\nPhase 2 done. Results under:", out_root, "contextual/")


def cmd_context_all(args: argparse.Namespace):
    cmd_context_extract(args)
    cmd_context_analyze(args)


# --------------------------------------------------------------------------- #
# Phase 3: Attention wavelet analysis
# --------------------------------------------------------------------------- #

DEFAULT_ATTN_SENTENCES = [
    "The bank approved the loan for the new house today.",
    "The fisher sat on the bank of the river and waited.",
    "She opened her book and started to read quietly.",
]

def _attention_src_dir(args) -> str:
    """Where the per-model attention sub-dir lives."""
    return os.path.join(args.data_dir or _default_results_dir(), "attention")


def cmd_attention_extract(args: argparse.Namespace):
    """Run an attention-instrumented forward pass on each sentence and dump
    every (layer, head) matrix to NPZ files."""
    from attention.extractor import extract_and_save

    models = args.models or DEFAULT_MODELS
    sentences = args.sentences or DEFAULT_ATTN_SENTENCES
    out_dir = _attention_src_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    for m in models:
        for sid, sentence in enumerate(sentences):
            # Tag an id so we can save multiple sentences per model.
            sub_dir = os.path.join(
                out_dir, "_sentences",
                f"{m}_{sid:02d}_" + sentence[:30].replace(" ", "_").replace("?", "")
                  .replace(",", "").replace(".", "")
            )
            print(f"[attention-extract] model={m} sentence_id={sid} "
                  f"sentence={sentence!r}")
            try:
                paths = extract_and_save(m, sentence, sub_dir,
                                         max_length=args.max_length or 64)
                print(f"    saved {len(paths) - 2} head .npz under {sub_dir}")
            except Exception as e:
                print(f"    WARN: extraction failed: {e}")
    print("[attention-extract] Done.")
    print("\nNOTE: The analyze command operates ONLY on the per-model dir "
          "results/attention/<model>/ (a single snapshot). Currently we save "
          "snapshots under results/attention/_sentences/<key>/. Use "
          "--src-dir to point to the directory you wish to analyse.")


def cmd_attention_analyze(args: argparse.Namespace):
    """Run all Phase-3 experiments on every snapshot directory present."""
    from attention import AttentionLoader, AttentionWaveletDecomposer, compute_head_metrics
    from experiments.attention_compare import aggregate_model, run_model_comparison
    from experiments.head_similarity import compute_pair_matrix
    from experiments.layer_progression import compute_and_save_layer_progression
    from experiments.redundancy import compute_redundancy
    from experiments.attention_reconstruction import compress_all_heads
    from experiments.head_clusters import run_clustering
    from visualization.attention_heatmaps import (
        original_attention_heatmap, wavelet_coefficient_heatmap,
        energy_spectrum_bars, coefficient_histogram, reconstruction_comparison,
    )
    from visualization.layer_evolution import plot_layer_evolution_across_models
    from wavelets import make_decomposer

    wavelets = args.wavelets or DEFAULT_WAVELETS
    wavelets = [_fix_wavelet_name(w) for w in wavelets]
    out_root = args.out or _default_results_dir()
    # Default src dir: instanced under "attention/_sentences/" -- the user must
    # pass --src-dir if they want a specific snapshot; otherwise we iterate
    # ALL snapshot dirs found there.
    src_root = args.src_dir
    if src_root is None:
        src_root = os.path.join(_attention_src_dir(args), "_sentences")
    if not os.path.isdir(src_root):
        print(f"[attention-analyze] No snapshots found under {src_root}. "
              f"Run `python main.py attention-extract` first.")
        return

    snapshot_dirs = sorted([
        os.path.join(src_root, d)
        for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    ])
    # Optional subset filter: process only snapshots whose directory name
    # contains any of the given substrings (lets callers shard the work
    # across parallel processes).
    only = getattr(args, "only", None)
    if only:
        snapshot_dirs = [d for d in snapshot_dirs
                          if any(s in os.path.basename(d) for s in only)]
    if not snapshot_dirs:
        print(f"[attention-analyze] No snapshot directories under {src_root}.")
        return

    all_layer_per_model: Dict[str, list] = {}   # for cross-model evolution plot
    model_aggregates: Dict[str, dict] = {}      # keyed by (snapshot, wavelet)

    for snap_dir in snapshot_dirs:
        # Find the inner model subdir (one entry under snapshot == model name)
        sub_models = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir)
                      if os.path.isdir(os.path.join(snap_dir, d))]
        if not sub_models:
            continue
        model_dir = sub_models[0]
        snap_label = os.path.basename(snap_dir).split("_", 1)[0]  # model_key prefix
        try:
            loader = AttentionLoader(model_dir)
        except Exception as e:
            print(f"[attention-analyze] skip {model_dir}: {e}")
            continue
        print(f"[attention-analyze] snapshot={os.path.basename(snap_dir)} "
              f"model={snap_label} L={loader.n_layers} H={loader.n_heads} "
              f"T={loader.seq_len}")

        for wname in wavelets:
            wdir_root = os.path.join(out_root, "attention_analysis",
                                     os.path.basename(snap_dir),
                                     "wavelet_" + wname)
            os.makedirs(wdir_root, exist_ok=True)
            print(f"\n  --- Wavelet: {wname} ---")

            # 1. Per-head metrics CSV
            print("  [exp] compute_head_metrics ...")
            rows: List[Dict[str, float]] = []
            attn_decomposer = AttentionWaveletDecomposer(wname)
            for L in range(loader.n_layers):
                for H in range(loader.n_heads):
                    head = loader.load_head(L, H)
                    m = compute_head_metrics(head.normalized, attn_decomposer)
                    row = {"layer": L, "head": H, **m}
                    rows.append(row)
            import csv
            with open(os.path.join(wdir_root, "head_metrics.csv"), "w",
                      newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["layer", "head"] + list(rows[0].keys())[2:])
                for r in rows:
                    writer.writerow([r["layer"], r["head"]]
                                     + [f"{r[k]:.5f}" if isinstance(r[k], float)
                                        else r[k] for k in list(r.keys())[2:]])

            # 2. Head similarity
            print("  [exp] head_similarity ...")
            sim_dir = os.path.join(wdir_root, "similarity")
            sim_mats = compute_pair_matrix(
                loader,
                decomposer_factory=lambda w=wname: AttentionWaveletDecomposer(w),
                save_dir=sim_dir,
            )

            # 3. Layer progression
            print("  [exp] layer_progression ...")
            summaries = compute_and_save_layer_progression(
                loader,
                decomposer_factory=lambda w=wname: AttentionWaveletDecomposer(w),
                save_dir=os.path.join(wdir_root, "layer_progression"),
            )
            all_layer_per_model.setdefault(snap_label, summaries)

            # 4. Redundancy
            print("  [exp] redundancy ...")
            compute_redundancy(
                loader,
                similarity_npy_path=os.path.join(sim_dir,
                                                  "similarity_wavelet.npy"),
                rows=rows,
                save_dir=os.path.join(wdir_root, "redundancy"),
            )

            # 5. Compression experiment (per head, across ratios)
            print("  [exp] attention_compression ...")
            compress_all_heads(
                loader,
                decomposer_factory=lambda w=wname: AttentionWaveletDecomposer(w),
                save_dir=os.path.join(wdir_root, "compression"),
            )

            # 6. Head clustering (wavelet features vs raw cos features)
            print("  [exp] head_clusters ...")
            run_clustering(
                loader,
                decomposer_factory=lambda w=wname: AttentionWaveletDecomposer(w),
                save_dir=os.path.join(wdir_root, "clusters"),
            )

            # 7. Head-level visualisations: original + wavelet coeffs for a
            # representative subset (first head of each layer).
            print("  [viz] attention_heatmaps/wavelet ...")
            for L in range(loader.n_layers):
                head = loader.load_head(L, 0)
                decomposer_for_viz = AttentionWaveletDecomposer(wname)
                d = decomposer_for_viz.decompose(head.normalized)
                viz_dir = os.path.join(wdir_root, "visualisations",
                                         f"layer{L}_head0")
                original_attention_heatmap(
                    head, loader.tokens,
                    save_dir=viz_dir, basename="attention_heatmap",
                )
                wavelet_coefficient_heatmap(
                    d, save_dir=viz_dir,
                    basename="wavelet_coefficients",
                )
                energy_spectrum_bars(
                    d, save_dir=viz_dir, basename="energy_spectrum",
                )
                coefficient_histogram(
                    d, save_dir=viz_dir, basename="coefficient_histogram",
                )
                rec = decomposer_for_viz.reconstruct(d, threshold_ratio=0.30,
                                                       crop=True)
                reconstruction_comparison(
                    head.normalized, rec,
                    save_dir=viz_dir,
                    basename="reconstruction_comparison",
                )

            # 8. Per-model aggregate
            _, _agg = aggregate_model(loader,
                                       AttentionWaveletDecomposer(wname),
                                        redundancy_volume=0)
            # Aggregate for cross-model evolution is later rebuilt per wavelet to
            # avoid swinging decomposers across wavelet passes.
            model_aggregates[(snap_label, wname)] = (snap_label, wname, rows)

    # Cross-model layer evolution comparison (per wavelet)
    if all_layer_per_model:
        for wname in wavelets:
            cross_models: Dict[str, list] = {}
            for snap_dir in snapshot_dirs:
                sub_models = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir)
                              if os.path.isdir(os.path.join(snap_dir, d))]
                if not sub_models:
                    continue
                try:
                    loader = AttentionLoader(sub_models[0])
                except Exception:
                    continue
                summaries = compute_and_save_layer_progression(
                    loader,
                    decomposer_factory=lambda w=wname: AttentionWaveletDecomposer(w),
                    save_dir=os.path.join(out_root, "attention_analysis",
                                          os.path.basename(snap_dir),
                                          "wavelet_" + wname, "layer_progression"),
                )
                cross_models[os.path.basename(snap_dir).split("_", 1)[0]] = summaries
            out_viz = os.path.join(out_root, "attention_analysis", "cross_model",
                                    "wavelet_" + wname)
            os.makedirs(out_viz, exist_ok=True)
            plot_layer_evolution_across_models(cross_models, save_dir=out_viz)

    print("\nPhase 3 done. Results under:", out_root, "attention_analysis/")


def cmd_attention_all(args: argparse.Namespace):
    cmd_attention_extract(args)
    cmd_attention_analyze(args)


# --------------------------------------------------------------------------- #
# Phase 4: predictive pruning
# --------------------------------------------------------------------------- #

DEFAULT_PRUNING_SENTENCES = [
    "The bank approved the loan for the new house today.",
    "The fisher sat on the bank of the river and waited.",
    "She opened her book and started to read quietly.",
    "The young boy kicked the red ball across the field.",
    "The judge read the brief and dismissed the case entirely.",
]


def cmd_pruning_analyze(args: argparse.Namespace):
    """Per-head ablation experiment + predictor validation vs baselines."""
    from embeddings.extract import EmbeddingExtractor, MODEL_REGISTRY
    from attention import (
        AttentionLoader, AttentionWaveletDecomposer, compute_head_metrics,
    )
    from pruning.registry import PREDICTOR_NAMES, compute_predictor
    from experiments.predictive_validation import (
        validate_predictor, ranked_aggregate_pruning,
    )
    from experiments.baselines import validate_all_predictors
    from visualization.predictive_plots import (
        predictor_correlation_bars, predictor_scatter, ranked_pruning_curves,
    )

    models = args.models or DEFAULT_MODELS
    wavelets = args.wavelets or DEFAULT_WAVELETS
    wavelets = [_fix_wavelet_name(w) for w in wavelets]
    sentences = args.sentences or DEFAULT_PRUNING_SENTENCES
    out_root = args.out or _default_results_dir()
    data_dir = args.data_dir or _default_results_dir()

    src_root = args.src_dir
    if src_root is None:
        src_root = os.path.join(_attention_src_dir(args), "_sentences")
    if not os.path.isdir(src_root):
        print(f"[pruning-analyze] No attention snapshots under {src_root}. "
              f"Run `python main.py attention-extract` first.")
        return

    snapshot_dirs = sorted([
        os.path.join(src_root, d)
        for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
    ])
    for snap_dir in snapshot_dirs:
        sub_models = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir)
                      if os.path.isdir(os.path.join(snap_dir, d))]
        if not sub_models:
            continue
        model_dir = sub_models[0]
        snap_label = os.path.basename(snap_dir).split("_", 1)[0]
        if models and snap_label not in models:
            continue
        try:
            loader = AttentionLoader(model_dir)
        except Exception as e:
            print(f"[pruning-analyze] skip {model_dir}: {e}")
            continue
        # Re-instantiate the underlying HF model for ablation
        print(f"[pruning-analyze] snapshot={os.path.basename(snap_dir)} "
              f"model={snap_label} L={loader.n_layers} H={loader.n_heads} "
              f"T={loader.seq_len}")
        ext = EmbeddingExtractor(snap_label)
        ext.load()

        # Build per-head attention tensor needed for some predictors
        head_attention_list = []
        for L_i in range(loader.n_layers):
            for H_i in range(loader.n_heads):
                head_attention_list.append(
                    loader.load_head(L_i, H_i).normalized
                )

        # The per-head ablation sweep is independent of both the predictor
        # and the wavelet: compute it ONCE per snapshot and share it across
        # every (predictor, wavelet) combination below.
        from evaluation.task_loss import run_model as _run_model
        from experiments.predictive_validation import _per_head_validation
        all_heads = [(L_i, H_i)
                      for L_i in range(loader.n_layers)
                      for H_i in range(loader.n_heads)]
        print("  [exp] shared per-head ablation sweep ...")
        orig_runs = _run_model(ext.model, snap_label, sentences,
                                ext.tokenizer, device=ext.device)
        cached_effects = _per_head_validation(
            ext.model, snap_label, ext.tokenizer, sentences,
            all_heads, orig_runs, device=ext.device,
        )

        for wname in wavelets:
            print(f"\n  --- Wavelet: {wname} ---")
            # 1. Compute per-head metrics
            from attention.analyzer import (
                AttentionWaveletDecomposer as AWD,
            )
            dec = AttentionWaveletDecomposer(wname)
            rows = []
            for L_i in range(loader.n_layers):
                for H_i in range(loader.n_heads):
                    m = compute_head_metrics(loader.load_head(L_i, H_i).normalized,
                                                dec)
                    rows.append({"layer": L_i, "head": H_i, **m})
            extra = {"head_attention": head_attention_list}

            prune_dir = os.path.join(out_root, "pruning",
                                       os.path.basename(snap_dir),
                                       "wavelet_" + wname)
            os.makedirs(prune_dir, exist_ok=True)

            # 2. Per-predictor correlation
            print("  [exp] validate predictors ...")
            reports = {}
            for name in PREDICTOR_NAMES:
                print("    - " + name)
                reports[name] = validate_predictor(
                    ext.model, snap_label, ext.tokenizer, sentences,
                    rows, extra, predictor_name=name,
                    device=ext.device,
                    cached_effects=cached_effects,
                )
            # Save CSV
            import csv
            with open(os.path.join(prune_dir, "predictor_correlations.csv"),
                      "w", newline="", encoding="utf-8") as f:
                w_ = csv.writer(f)
                w_.writerow([
                    "predictor", "pearson_r", "spearman_rho", "kendall_tau",
                    "aggregate_kl", "aggregate_cosine",
                ])
                for n, r in reports.items():
                    w_.writerow([
                        n, f"{r.pearson_r:+.4f}", f"{r.spearman_rho:+.4f}",
                        f"{r.kendall_tau:+.4f}",
                        f"{r.aggregate_kl:.5f}", f"{r.aggregate_cosine:.5f}",
                    ])
            # Per-head supervised
            with open(os.path.join(prune_dir, "per_head_validation.csv"),
                      "w", newline="", encoding="utf-8") as f:
                w_ = csv.writer(f)
                w_.writerow([
                    "layer", "head", "predictor",
                    "predicted_importance", "predicted_unimportance",
                    "cosine_drop", "kl_div", "attention_drift",
                ])
                for n, r in reports.items():
                    for p in r.per_head:
                        w_.writerow([
                            p.layer, p.head, n,
                            f"{p.predicted_importance:.4f}",
                            f"{p.predicted_unimportance:.4f}",
                            f"{p.cosine_drop:.5f}",
                            f"{p.kl_div_next_token:.5f}",
                            f"{p.attention_drift:.5f}",
                        ])
            # Aggregate ranked pruning
            print("  [exp] aggregate pruning sweep ...")
            agg_by_pred = {}
            for name in PREDICTOR_NAMES:
                agg_by_pred[name] = ranked_aggregate_pruning(
                    ext.model, snap_label, ext.tokenizer, sentences,
                    rows, extra, predictor_name=name,
                    device=ext.device,
                )
            with open(os.path.join(prune_dir, "aggregate_pruning.csv"),
                      "w", newline="", encoding="utf-8") as f:
                w_ = csv.writer(f)
                w_.writerow([
                    "predictor", "ratio", "cosine_drop",
                    "kl_div", "attention_drift",
                ])
                for name, lst in agg_by_pred.items():
                    for r in lst:
                        w_.writerow([
                            name, f"{r.ratio:.2f}",
                            f"{r.cosine_drop:.4f}", f"{r.kl_div:.4f}",
                            f"{r.attention_drift:.4f}",
                        ])
            # Visuals
            predictor_correlation_bars(reports, save_dir=prune_dir)
            predictor_scatter(reports, save_dir=prune_dir)
            ranked_pruning_curves(agg_by_pred, save_dir=prune_dir)

        ext.close()
    print("\nPhase 4 done. Results under:", out_root, "pruning/")


def cmd_pruning_all(args: argparse.Namespace):
    cmd_attention_extract(args)
    cmd_pruning_analyze(args)


def cmd_benchmark_run(args: argparse.Namespace):
    """Run the reproducible head-pruning benchmark.

    Iterates models x datasets x seeds, scores every predictor (the existing
    wavelet, the simple baselines and the new published baselines) and reports
    mean +/- SD, bootstrap CIs, paired statistics and effect sizes.
    """
    from benchmark.runner import run_benchmark, write_results

    out_root = args.out or os.path.join(_default_results_dir(), "benchmark")
    os.makedirs(out_root, exist_ok=True)
    result = run_benchmark(
        models=args.models, datasets=args.datasets, seeds=args.seeds,
        max_sentences=args.max_sentences, max_len=args.max_length,
        cache_dir=args.cache_dir, metric=args.metric,
        device=args.device, verbose=True,
    )
    doc = write_results(result, out_root, test="wilcoxon")
    print("\nPhase 5 benchmark done.")
    print(" cells:", doc["n_cells"])
    print(" predictors:", ", ".join(doc["predictors"]))
    print(" per-input CSV :", doc["per_input_csv"])
    print(" aggregate CSV:", doc["aggregate_csv"])
    print(" seed var CSV :", doc.get("seed_variance_csv"))
    print(" comparisons:")
    for c in doc["comparisons"][:12]:
        print(f"   wavelet vs {c['baseline']:<14} "
              f"diff={c['diff_mean']:+.4f} "
              f"p={c['p_value']:.4f} "
              f"d={c['cohens_d']:+.3f} delta={c['cliffs_delta']:+.3f} "
              f"[{c['metric']} {c['model']} {c['dataset']}]")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 1: Wavelet Analysis of Transformer Embeddings"
    )
    sub = p.add_subparsers(dest="stage", required=True)

    pe = sub.add_parser("extract", help="Pull embeddings from HF model hub")
    pe.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help=f"model keys, default: {DEFAULT_MODELS}")
    pe.add_argument("--data-dir", default=None,
                    help="location to store the extracted .npz files")
    pe.set_defaults(func=cmd_extract)

    common = dict()
    pa = sub.add_parser("analyze", help="Run experiments on extracted embeddings")
    pa.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pa.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS,
                    help=f"wavelet names; defaults: {DEFAULT_WAVELETS}. "
                         f"Aliases: symlet/coiflet/daubechies accepted.")
    pa.add_argument("--sample", type=int, default=2000,
                    help="sample size per model for batch stats (default 2000)")
    pa.add_argument("--out", default=None,
                    help="results root directory (default ./results/)")
    pa.add_argument("--data-dir", default=None,
                    help="location of extracted .npz files (default ./results/embeddings)")
    pa.set_defaults(func=cmd_analyze)

    pall = sub.add_parser("all", help="Extract + analyze in sequence")
    pall.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pall.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS)
    pall.add_argument("--sample", type=int, default=2000)
    pall.add_argument("--out", default=None)
    pall.add_argument("--data-dir", default=None)
    pall.set_defaults(func=cmd_all)

    # ---- Phase 2: contextual ----
    pce = sub.add_parser("context-extract",
                         help="Pull per-token contextual vectors for polysemy dataset")
    pce.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pce.add_argument("--anchors", nargs="+", default=None,
                     help="anchor tokens to run (default: full POLYSEMY_DATASET)")
    pce.add_argument("--data-dir", default=None,
                    help="location to store the extracted .npz files")
    pce.set_defaults(func=cmd_context_extract)

    pca = sub.add_parser("context-analyze",
                         help="Run Phase-2 experiments on contextual vectors")
    pca.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pca.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS)
    pca.add_argument("--anchors", nargs="+", default=None,
                     help="anchor tokens to analyze")
    pca.add_argument("--out", default=None)
    pca.add_argument("--data-dir", default=None)
    pca.set_defaults(func=cmd_context_analyze)

    pcall = sub.add_parser("context-all",
                            help="Context extract + analyze in one go")
    pcall.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pcall.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS)
    pcall.add_argument("--anchors", nargs="+", default=None)
    pcall.add_argument("--out", default=None)
    pcall.add_argument("--data-dir", default=None)
    pcall.set_defaults(func=cmd_context_all)

    # ---- Phase 3: attention wavelet analysis ----
    pae = sub.add_parser("attention-extract",
                          help="Forward pass with output_attentions and dump every head matrix")
    pae.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pae.add_argument("--sentences", nargs="+", default=None,
                     help="sentences to run; defaults to three pre-built ones")
    pae.add_argument("--max-length", type=int, default=64)
    pae.add_argument("--data-dir", default=None)
    pae.set_defaults(func=cmd_attention_extract)

    paa = sub.add_parser("attention-analyze",
                          help="Run Phase-3 attention analysis experiments")
    paa.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    paa.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS)
    paa.add_argument("--only", nargs="+", default=None,
                     help="substring filter on snapshot dir names; only "
                          "matching snapshots are analysed")
    paa.add_argument("--src-dir", default=None,
                     help="directory containing snapshot subdirs "
                          "(default: results/attention/_sentences/<key>/)")
    paa.add_argument("--out", default=None)
    paa.add_argument("--data-dir", default=None)
    paa.set_defaults(func=cmd_attention_analyze)

    pa_all = sub.add_parser("attention-all",
                             help="Attention extract + analyze in one go")
    pa_all.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pa_all.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS)
    pa_all.add_argument("--sentences", nargs="+", default=None)
    pa_all.add_argument("--max-length", type=int, default=64)
    pa_all.add_argument("--src-dir", default=None)
    pa_all.add_argument("--out", default=None)
    pa_all.add_argument("--data-dir", default=None)
    pa_all.set_defaults(func=cmd_attention_all)

    # ---- Phase 4: predictive pruning ----
    ppz = sub.add_parser("pruning-analyze",
                          help="Run Phase-4 head-pruning validation experiment")
    ppz.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ppz.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS)
    ppz.add_argument("--sentences", nargs="+", default=None,
                     help="proxy-task sentences to score with")
    ppz.add_argument("--src-dir", default=None,
                     help="directory of attention snapshots")
    ppz.add_argument("--out", default=None)
    ppz.add_argument("--data-dir", default=None)
    ppz.set_defaults(func=cmd_pruning_analyze)

    pp_all = sub.add_parser("pruning-all",
                             help="Phase-1 attention-extract + Phase-4 pruning-analyze")
    pp_all.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    pp_all.add_argument("--wavelets", nargs="+", default=DEFAULT_WAVELETS)
    pp_all.add_argument("--sentences", nargs="+", default=None)
    pp_all.add_argument("--max-length", type=int, default=64)
    pp_all.add_argument("--src-dir", default=None)
    pp_all.add_argument("--out", default=None)
    pp_all.add_argument("--data-dir", default=None)
    pp_all.set_defaults(func=cmd_pruning_all)

    # ---- Phase 5: reproducible head-pruning benchmark ----
    DEFAULT_BENCH_MODELS = [
        "bert-base", "distilbert", "gpt2", "roberta-base",
        "deberta-base", "tinyllama",
    ]
    DEFAULT_BENCH_DATASETS = ["wikitext2", "penn_treebank", "glue_subset"]
    DEFAULT_SEEDS = [0, 1, 2]

    pbr = sub.add_parser("benchmark-run",
                          help="Run the reproducible head-pruning benchmark "
                               "across models, datasets and seeds.")
    pbr.add_argument("--models", nargs="+",
                       default=DEFAULT_BENCH_MODELS)
    pbr.add_argument("--datasets", nargs="+",
                       default=DEFAULT_BENCH_DATASETS)
    pbr.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    pbr.add_argument("--max-sentences", type=int, default=2000)
    pbr.add_argument("--max-length", type=int, default=64)
    pbr.add_argument("--metric", default="cosine_drop",
                       choices=["cosine_drop", "attention_drift",
                                "kl_div_next_token"])
    pbr.add_argument("--out", default=None)
    pbr.add_argument("--cache-dir", default=None)
    pbr.add_argument("--device", default=None)
    pbr.set_defaults(func=cmd_benchmark_run)
    return p


def main(argv: List[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
