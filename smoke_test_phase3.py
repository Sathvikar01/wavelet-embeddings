"""Phase-3 offline smoke test (no HF download) for the analyzer + loader.

We synthesise a fake attention snapshot dir + verify all metrics compute
reliably.
"""

from __future__ import annotations
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention import (AttentionWaveletDecomposer, compute_head_metrics,
                       head_feature_vector)
from attention.loader import HeadMatrix


def _fake_head(L: int, H: int, T: int = 16, seed: int = 0) -> HeadMatrix:
    rng = np.random.default_rng(seed + L * 1000 + H)
    raw = rng.normal(size=(T, T)).astype(np.float32)
    raw = np.abs(raw) / 1.2
    # Make row-normalised
    row_sums = raw.sum(axis=1, keepdims=True)
    return HeadMatrix(layer=L, head=H, raw=raw,
                      normalized=raw / np.maximum(row_sums, 1e-9))


def _cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    print("[phase3 smoke] starting...", flush=True)
    T = 16
    for w in ('haar', 'db4', 'sym4', 'coif2'):
        dec = AttentionWaveletDecomposer(wavelet_name=w)
        head = _fake_head(2, 3, T=T)
        m = compute_head_metrics(head.normalized, dec)
        print(f"  {w:<5} metrics:", {k: round(v, 3) for k, v in m.items()
                                       if isinstance(v, (int, float))}, flush=True)
        # Reconstruction sanity
        d = dec.decompose(head.normalized)
        rec = dec.reconstruct(d, threshold_ratio=0.30, crop=True)
        print(f"      rec shape={rec.shape}  L2(err)="
              f"{np.linalg.norm(head.normalized - rec):.4f}", flush=True)
    print("[phase3 smoke] OK", flush=True)

if __name__ == '__main__':
    main()
