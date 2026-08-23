"""Embedding compression shootout: dimension reduction vs sparsification vs
quantization vs product quantization, all compared at EQUAL bytes/token.

Methods (original payload: 768 x float32 = 3072 bytes/token):

  pca_<d>            dense truncated PCA projection (fit on train split)
  rp_<d>             orthogonal random projection (structure-free control)
  topk_f32           keep nnz largest RAW coordinates (4B value + 2B index)
  wav_f32            same nnz budget, selected in db4-wavelet domain
  topk_int8          raw top-k, values quantised symmetric int8 (1B + 2B idx)
  wav_int8           wavelet-selected coefficients, values quantised int8
  pq_m               dense product quantisation, m bytes/code (K=256/subspace)

Storage accounting: float32 payload counts 4B/dim; sparse entries count
value-bytes + 2B uint16 index; PQ adds its codebook (256 x D x 4B)
amortised over the full vocabulary. PCA/RP add their projection matrix
amortised too (negligible but included).

Usage:
  python experiments/dim_reduction.py                 # all models, defaults
  python experiments/dim_reduction.py --budgets 128 256 512
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import sys

import numpy as np
from scipy.fft import dct, idct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embeddings.loader import EmbeddingLoader          # noqa: E402
from wavelets.base import WaveletDecomposer            # noqa: E402

DEFAULT_BUDGETS = (64, 128, 256, 512, 1024)
K_NEIGHBOURS = 10


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _nn_indices(X: np.ndarray, k: int) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)
    return np.argsort(-sims, axis=1)[:, :k]


def _jaccard_vs(nn_orig: np.ndarray, X_recon: np.ndarray,
                k: int = K_NEIGHBOURS) -> float:
    nn_rec = _nn_indices(X_recon, k)
    jacs = []
    for a, b in zip(nn_orig, nn_rec):
        sa, sb = set(a.tolist()), set(b.tolist())
        jacs.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(jacs))


def _mean_cosine(X: np.ndarray, R: np.ndarray) -> float:
    num = np.sum(X * R, axis=1)
    den = np.linalg.norm(X, axis=1) * np.linalg.norm(R, axis=1) + 1e-12
    return float(np.mean(num / den))


# --------------------------------------------------------------------------- #
# Compression primitives
# --------------------------------------------------------------------------- #

def _quantise_int8(vals: np.ndarray):
    """Symmetric per-vector int8 quantisation -> (q, scale)."""
    s = float(np.max(np.abs(vals))) / 127.0 if vals.size else 0.0
    if s == 0.0:
        return np.zeros_like(vals), 0.0
    q = np.clip(np.round(vals / s), -127, 127)
    return q, s


def _sparse_raw_recon(x: np.ndarray, nnz: int, quantise: bool) -> np.ndarray:
    """Keep nnz largest-|.| coordinates; optional int8 values."""
    r = np.zeros_like(x)
    if nnz <= 0:
        return r
    idx = np.argpartition(np.abs(x), len(x) - nnz)[len(x) - nnz:]
    vals = x[idx]
    if quantise:
        q, s = _quantise_int8(vals)
        vals = (q * s).astype(x.dtype)
    r[idx] = vals
    return r


def _dct_sparse_recon(x: np.ndarray, nnz: int) -> np.ndarray:
    """Keep nnz largest-|DCT-II| coefficients (global frequency basis,
    no multiresolution); inverse DCT. Real coefficients -> 6 B/kept,
    same accounting as fp32 sparse variants."""
    c = dct(x.astype(np.float64), norm="ortho")
    if nnz <= 0:
        return idct(np.zeros_like(c), norm="ortho")
    idx = np.argpartition(np.abs(c), len(c) - nnz)[len(c) - nnz:]
    mask = np.zeros_like(c, dtype=bool)
    mask[idx] = True
    return idct(c * mask, norm="ortho")


def _fft_sparse_recon(x: np.ndarray, nnz: int) -> np.ndarray:
    """Keep nnz largest-magnitude rfft coefficients (complex payload,
    10 B/kept incl. uint16 index); inverse real FFT."""
    C = np.fft.rfft(x.astype(np.float64))
    k = min(nnz, len(C))
    if k <= 0:
        return np.fft.irfft(np.zeros_like(C), n=len(x))
    idx = np.argpartition(np.abs(C), len(C) - k)[len(C) - k:]
    mask = np.zeros_like(C, dtype=bool)
    mask[idx] = True
    return np.fft.irfft(C * mask, n=len(x))


def _wavelet_sparse_recon(dec: WaveletDecomposer, dcp, nnz: int,
                          quantise: bool) -> np.ndarray:
    """Keep approx + (nnz - |approx|) largest detail coefficients (flat
    magnitude order); optional joint int8 quantisation of kept values;
    inverse DWT."""
    approx, details = dcp.approx, list(dcp.details)
    det_total = sum(c.size for c in details)
    k_det = min(max(nnz - approx.size, 0), det_total)
    flat = np.concatenate([np.abs(c).ravel() for c in details])
    if k_det > 0:
        keep_idx = np.argpartition(flat, len(flat) - k_det)[len(flat) - k_det:]
        keep_mask = np.zeros(len(flat), dtype=bool)
        keep_mask[keep_idx] = True
    else:
        keep_mask = np.zeros(len(flat), dtype=bool)

    parts = [approx.ravel()]
    shapes, cursor = [], 0
    for c in details:
        sub = keep_mask[cursor:cursor + c.size].reshape(c.shape)
        parts.append(np.where(sub, c, 0.0).ravel())
        shapes.append(c.shape)
        cursor += c.size
    sizes = [p.size for p in parts]
    cat = np.concatenate(parts)
    if quantise:
        nz = cat != 0.0
        q, s = _quantise_int8(cat[nz])
        cat[nz] = q * s
    new_approx = cat[:sizes[0]].reshape(approx.shape)
    new_details, cur = [], sizes[0]
    for sz, shape in zip(sizes[1:], shapes):
        new_details.append(cat[cur:cur + sz].reshape(shape))
        cur += sz
    dcp2 = dataclasses.replace(dcp, approx=new_approx, details=new_details)
    return dec.reconstruct(dcp2, threshold_ratio=0.0, crop=True)


class ProductQuantizer:
    """Dense PQ: split D into m subspaces, k-means (K=256) per subspace."""

    def __init__(self, m: int, K: int = 256, seed: int = 0):
        self.m, self.K, self.seed = m, K, seed

    def fit_encode(self, X_train: np.ndarray, X_test: np.ndarray):
        from sklearn.cluster import MiniBatchKMeans
        n, D = X_train.shape
        assert D % self.m == 0, f"D={D} not divisible by m={self.m}"
        sd = D // self.m
        codes_tr = np.empty((n, self.m), dtype=np.uint8)
        codes_te = np.empty((len(X_test), self.m), dtype=np.uint8)
        books = []
        for j in range(self.m):
            sl = slice(j * sd, (j + 1) * sd)
            km = MiniBatchKMeans(n_clusters=self.K, n_init=3, max_iter=50,
                                  batch_size=1024,
                                  random_state=self.seed + j)
            km.fit(X_train[:, sl])
            books.append(km.cluster_centers_)
            codes_tr[:, j] = km.predict(X_train[:, sl])
            codes_te[:, j] = km.predict(X_test[:, sl])
        self.books = np.stack(books)              # (m, K, sd)
        recon_tr = self._decode(codes_tr)
        recon_te = self._decode(codes_te)
        return recon_tr, recon_te

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        n, m = codes.shape
        sd = self.books.shape[2]
        out = np.empty((n, m * sd), dtype=np.float64)
        for j in range(m):
            out[:, j * sd:(j + 1) * sd] = self.books[j][codes[:, j]]
        return out


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #

def run(model_key: str, budgets, n_sample: int, seed: int, out_dir: str,
        only_methods=None):
    npz_path = os.path.join("results", "embeddings",
                             f"{model_key}_embeddings.npz")
    loader = EmbeddingLoader(npz_path)
    X = loader.embeddings
    n_vocab, D = X.shape
    rng = np.random.default_rng(seed)
    idx_all = rng.choice(n_vocab, size=min(2 * n_sample, n_vocab),
                          replace=False)
    tr, te = idx_all[:n_sample], idx_all[n_sample:2 * n_sample]
    X_train, X_test = X[tr].astype(np.float64), X[te].astype(np.float64)
    print(f"[{model_key}] vocab={n_vocab} dim={D} train={len(tr)} "
          f"test={len(te)}")

    dec = WaveletDecomposer("db4")
    decomps = [dec.decompose(x) for x in X_test]
    approx_size = decomps[0].approx.size
    nn_orig = _nn_indices(X_test, K_NEIGHBOURS)

    # Shared PCA basis (fit once).
    mean = X_train.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X_train - mean, full_matrices=False)
    tot_var = float((S ** 2).sum())
    pq_book_overhead = (256 * D * 4) / n_vocab   # amortised codebook
    mean_row_overhead = (D * 4) / n_vocab        # amortised PCA mean vector

    rows = []

    def _record(method, budget, param, recon, bpt, extra=""):
        cos = _mean_cosine(X_test, recon)
        jac = _jaccard_vs(nn_orig, recon)
        rows.append(dict(model=model_key, method=method,
                          budget_bytes=budget, param=param,
                          cosine=round(cos, 5),
                          jaccard10=round(jac, 4),
                          bytes_per_token=round(bpt, 1), extra=extra))
        return cos, jac

    for B in budgets:
        line = f"  B={B:<5}"

        # PCA
        d = max(2, B // 4)
        comps = Vt[:d]
        rec = (X_test - mean) @ comps.T @ comps + mean
        c, j = _record("pca", B, f"d={d}", rec,
                        d * 4 + (D * d * 4) / n_vocab + mean_row_overhead)
        line += f" pca {c:.3f}/{j:.2f}"

        # Random projection
        rngQ = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rngQ.normal(size=(D, d)))
        Q = q.astype(np.float64)
        rec = (X_test @ Q) @ Q.T
        c, j = _record("rand_proj", B, f"d={d}", rec,
                        d * 4 + (D * d * 4) / n_vocab)
        line += f" | rp {c:.3f}/{j:.2f}"

        # Sparse variants (float32 vs int8 payloads + spectral controls)
        nnz6, nnz3 = B // 6, B // 3
        recs = {mm: np.zeros_like(X_test) for mm in
                 ("topk_f32", "wav_f32", "dct_f32", "fft_c64",
                   "topk_i8", "wav_i8")}
        for i, (x, dcp) in enumerate(zip(X_test, decomps)):
            recs["topk_f32"][i] = _sparse_raw_recon(x, nnz6, quantise=False)
            recs["topk_i8"][i] = _sparse_raw_recon(x, nnz3, quantise=True)
            recs["wav_f32"][i] = _wavelet_sparse_recon(dec, dcp, nnz6,
                                                        quantise=False)
            recs["wav_i8"][i] = _wavelet_sparse_recon(dec, dcp, nnz3,
                                                       quantise=True)
            recs["dct_f32"][i] = _dct_sparse_recon(x, nnz6)
            recs["fft_c64"][i] = _fft_sparse_recon(x, nnz6)
        for name, bpt in (("topk_f32", nnz6 * 6), ("wav_f32", nnz6 * 6),
                           ("dct_f32", nnz6 * 6), ("fft_c64", nnz6 * 10),
                           ("topk_i8", nnz3 * 3 + 2),
                            ("wav_i8", nnz3 * 3 + 2)):
            if only_methods and name not in only_methods:
                continue
            c, j = _record(name, B, f"nnz={nnz6 if ('f32' in name or 'c64' in name) else nnz3}",
                            recs[name], bpt)
            line += f" | {name} {c:.3f}/{j:.2f}"

        # Product quantization (only where D % m == 0)
        if D % B == 0 and B <= D:
            pqt = ProductQuantizer(m=B, K=256, seed=seed)
            _, rec_pq = pqt.fit_encode(X_train, X_test)
            c, j = _record("pq", B, f"m={B}", rec_pq,
                            B + pq_book_overhead)
            line += f" | pq {c:.3f}/{j:.2f}"
        else:
            line += " | pq      -     "

        print(line, flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{model_key}_compression_shootout.csv")
    # Merge with any previous run: keep old rows whose method was not
    # recomputed now, so incremental passes (e.g. adding spectral
    # controls without re-fitting PQ) do not lose data.
    old = []
    if os.path.isfile(out_csv):
        with open(out_csv, encoding="utf-8") as f:
            old = [r for r in csv.DictReader(f)]
    new_methods = {r["method"] for r in rows}
    merged = [r for r in old if r["method"] not in new_methods] + rows
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    print(f"  saved {out_csv} ({len(merged)} rows)")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                    default=["bert-base", "distilbert", "gpt2"])
    p.add_argument("--budgets", nargs="+", type=int,
                    default=list(DEFAULT_BUDGETS))
    p.add_argument("--sample", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("results",
                                                  "dim_reduction"))
    p.add_argument("--only-methods", default=None,
                    help="comma-separated method filter (skips PQ refits "
                         "etc.; existing rows for other methods are kept)")
    args = p.parse_args()

    all_rows = []
    only = set(args.only_methods.split(",")) if args.only_methods else None
    for m in args.models:
        all_rows.extend(run(m, args.budgets, args.sample, args.seed,
                             args.out, only_methods=only))

    print("\n=== pooled mean over models: cosine / neighbour-Jaccard@10 ===")
    methods = ["pca", "rand_proj", "topk_f32", "wav_f32",
                "topk_i8", "wav_i8", "pq"]
    header = f"{'bytes':>7} " + " ".join(f"{mm:>13}" for mm in methods)
    print(header)
    for B in args.budgets:
        cells = []
        for mm in methods:
            sel = [r for r in all_rows
                    if r["method"] == mm and r["budget_bytes"] == B]
            if not sel:
                cells.append(f"{'-':>13}")
                continue
            c = float(np.mean([r["cosine"] for r in sel]))
            j = float(np.mean([r["jaccard10"] for r in sel]))
            cells.append(f"{c:.3f}/{j:.2f}".rjust(13))
        print(f"{B:>7} " + " ".join(cells))


if __name__ == "__main__":
    main()
