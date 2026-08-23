"""Smoke test for the wavelet-embeddings pipeline (no HF download required).

Creates a fake .npz with a vocab of ~30 tokens + categories, then runs a
mini version of every analysis module to verify the pipeline works end to end.
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embeddings.loader import EmbeddingLoader, DEFAULT_PROBE_TOKENS, NEIGHBOR_TOKENS
from wavelets import make_decomposer, SUPPORTED_WAVELETS, wavelet_similarity
from analysis.energy import energy_summary, total_energy
from analysis.entropy import entropy_summary, coefficient_statistics
from analysis.sparsity import sparsity_summary
from analysis.compression import (
    compress_embedding, compress_batch, neighbor_preservation,
    DEFAULT_COMPRESSION_RATIOS,
)


def _make_fake_npz(path: str, vocab_tokens, dim=128, seed=0):
    rng = np.random.default_rng(seed)
    V = len(vocab_tokens)
    # Diverse embeddings with smooth + rough components across tokens
    emb = rng.normal(size=(V, dim)).astype(np.float32)
    # Add a low-freq component to nouns etc.
    for i, t in enumerate(vocab_tokens):
        emb[i] += 0.5 * np.sin(np.arange(dim) * (0.1 + 0.05 * (i % 5)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, embeddings=emb,
             tokens=np.array(vocab_tokens, dtype=object))
    return path


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    tmp = os.path.join(here, "results", "_smoke", "embeddings")
    os.makedirs(tmp, exist_ok=True)
    tokens = (NEIGHBOR_TOKENS + ["the", "and", "of", "1", "100",
             ".", "?", "[CLS]", "[SEP]", "##s", "Ġthe", "##ing"] +
             ["run", "eat", "good", "bad", "king", "queen"])
    tokens = list(dict.fromkeys(tokens))
    npz = _make_fake_npz(os.path.join(tmp, "smoke_embeddings.npz"), tokens)

    loader = EmbeddingLoader(npz)
    print(f"Loaded {loader.vocab_size} tokens x dim {loader.embed_dim}")
    print("Categories present:", sorted(set(loader.categories)))

    for wname in SUPPORTED_WAVELETS:
        print(f"\n--- Wavelet: {wname} ---")
        dec = make_decomposer(wname)
        d = dec.decompose(loader.embeddings[0])
        print(f"level={d.level} approx_len={len(d.approx)} "
              f"#bands={len(d.details)}")
        print("Energy:", {k: round(v, 4) for k, v in energy_summary(d).items()})
        print("Entropy:", {k: round(v, 4) for k, v in entropy_summary(d).items()})
        print("CoeffStats:", {k: round(v, 4) for k, v in coefficient_statistics(d).items()})
        print("Sparsity:", {k: round(v, 4) for k, v in sparsity_summary(d).items()})

    # Compression on the full matrix
    dec = make_decomposer("db4")
    recs, stats = compress_batch(loader.embeddings, dec,
                                 ratios=DEFAULT_COMPRESSION_RATIOS)
    print("\nBatch compression stats (db4):")
    for r, s in stats.items():
        print(f"  r={r:.2f} cos={s.mean_cosine:.4f} snr={s.mean_snr_db:.3f}db "
              f"cr={s.mean_compression_ratio:.2f}")

    # Reconstruction on king
    res = compress_embedding(loader.embeddings[loader.index_of("king")], dec)
    print("\nReconstruction (king, db4):")
    for r in res:
        print(f"  r={r.ratio:.2f} cos={r.cosine:.4f} dr={r.relative_drift:.4f} "
              f"snr={r.snr_db:.3f}db cr={r.compression_ratio:.2f}")

    # Neighbour analysis
    decomp_king = dec.decompose(loader.embeddings[loader.index_of("king")])
    decomp_queen = dec.decompose(loader.embeddings[loader.index_of("queen")])
    print(f"\nWavelet sim (king, queen) = {wavelet_similarity(decomp_king, decomp_queen):.4f}")
    print(f"Cosine sim (king, queen) = "
          f"{np.dot(loader.embeddings[loader.index_of('king')], loader.embeddings[loader.index_of('queen')]) / (np.linalg.norm(loader.embeddings[loader.index_of('king')]) * np.linalg.norm(loader.embeddings[loader.index_of('queen')])):.4f}")

    # Neighbour preservation
    pres, drift = neighbor_preservation(
        loader.embeddings,
        recs[0.30], k=5,
    )
    print(f"\nNeighbour preservation @30%: pres={pres:.3f} drift={drift:.3f}")

    print("\n[OK] Smoke test passed.")


if __name__ == "__main__":
    main()
