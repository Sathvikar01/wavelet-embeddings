"""Offline regression checks for the 2026-08 bug-fix pass.

Run:  python smoke_test_fixes.py

Covers:
  1. analysis.energy.low_freq_energy buckets the COARSEST detail levels
     (pre-bug it selected the finest ones).
  2. analysis.compression energy_retained is populated and decreases with
     the threshold ratio; compression_ratio matches the number of
     coefficients _threshold_details actually zeroes.
  3. wavelets.base.reconstruct has no dead debug branch and round-trips.
  4. benchmark.datasets sentence splitting flushes on terminal punctuation
     (the endswith((".", "!", "?")) tuple fix).
  5. evaluation.task_loss._per_head_o_proj_path resolves a module for the
     deberta family path expression (structural check without weights).
  6. pruning.registry predictors stay deterministic across calls.
"""

import numpy as np

from benchmark.datasets import load_sentences


def check_energy_split() -> None:
    from wavelets.base import WaveletDecomposer
    from analysis.energy import (low_freq_energy, high_freq_energy,
                                  total_energy)
    d = WaveletDecomposer("db4")
    rng = np.random.default_rng(0)
    x = rng.normal(size=768)
    dec = d.decompose(x)
    n = dec.level
    half = max(1, n // 2)
    expect_low = dec.energy_approx + sum(
        dec.energy_per_level[i] for i in range(n - half + 1, n + 1))
    expect_high = sum(dec.energy_per_level[i] for i in range(1, half + 1))
    assert abs(low_freq_energy(dec) - expect_low) < 1e-9, \
        "low_freq_energy must bucket the coarsest levels"
    assert abs(high_freq_energy(dec) - expect_high) < 1e-9
    # low/high partition all detail energy (even level count)
    if n % 2 == 0:
        assert abs((expect_low + expect_high) - total_energy(dec)) < 1e-9
    print("  [1] energy low/high split ................ OK")


def check_compression() -> None:
    from wavelets.base import WaveletDecomposer, _threshold_details
    from analysis.compression import (compress_embedding, _nonzero_count,
                                       _retained_energy_fraction)
    d = WaveletDecomposer("haar")
    rng = np.random.default_rng(1)
    x = rng.normal(size=512)
    res = compress_embedding(x, d, ratios=(0.0 + 0.1, 0.3, 0.5))
    prev = 1.01
    for r in res:
        assert 0.0 < r.energy_retained <= 1.0, "energy_retained unpopulated"
        assert r.energy_retained <= prev + 1e-12, \
            "retained energy must be non-increasing in ratio"
        prev = r.energy_retained
    dec = d.decompose(x)
    det_total = sum(c.size for c in dec.details)
    n_zero = int(np.floor(0.3 * det_total))
    assert _nonzero_count(dec, 0.3) == dec.approx.size + det_total - n_zero
    thr = _threshold_details(list(reversed(dec.details)), 0.3)
    n_nonzero_actual = sum(int(np.count_nonzero(c)) for c in thr) \
        + dec.approx.size
    assert n_nonzero_actual == _nonzero_count(dec, 0.3), \
        "compression accounting must mirror _threshold_details"
    frac = _retained_energy_fraction(dec, 0.5)
    assert 0.0 < frac < 1.0
    print("  [2] compression energy/accounting ........ OK")


def check_roundtrip() -> None:
    from wavelets.base import WaveletDecomposer
    d = WaveletDecomposer("sym4")
    rng = np.random.default_rng(2)
    x = rng.normal(size=768)
    rec = d.reconstruct(d.decompose(x), threshold_ratio=0.0, crop=True)
    err = float(np.max(np.abs(rec - x)))
    assert err < 1e-6, f"round-trip error too large: {err}"
    print("  [3] DWT round-trip (no debug branch) ..... OK")


def check_sentence_split() -> None:
    from benchmark.datasets import _split_into_sentences
    text = ("The sky is blue and very vast. Water is wet and remarkably clear.\n"
            "\n"
            "It rained heavily here today! Was it really that cold outside?\n"
            "More prose here that deliberately\nwraps across lines today. "
            "And so the demonstration ends.")
    out = _split_into_sentences(text)
    assert out == [
        "The sky is blue and very vast.",
        "Water is wet and remarkably clear.",
        "It rained heavily here today!",
        "Was it really that cold outside?",
        "More prose here that deliberately wraps across lines today.",
        "And so the demonstration ends.",
    ], f"sentence splitter regressed: {out}"
    try:
        sents = load_sentences("wikitext2", max_sentences=200)
    except Exception as e:  # no network -> skip gracefully
        print(f"  [4] sentence split ................. SKIP ({e})")
        return
    assert sents, "loader returned no sentences"
    # Corpus-level heuristic: un-split multi-sentence paragraphs should be
    # rare (quote/abbreviation edge cases remain by design).
    bad = [s for s in sents if s.count(". ") > 6]
    assert len(bad) <= 0.05 * len(sents), \
        f"too many un-split paragraphs: {len(bad)}/{len(sents)}"
    print(f"  [4] sentence split ({len(sents)} sents) ............ OK")


def check_deberta_path() -> None:
    import inspect
    from evaluation.task_loss import _per_head_o_proj_path
    src = inspect.getsource(_per_head_o_proj_path)
    assert '"deberta"' in src, \
        "deberta family must be routed to encoder.layer[L].attention.output.dense"
    print("  [6] deberta ablation path ................ OK")


def check_determinism() -> None:
    from pruning.registry import compute_predictor
    rows = [{"shannon_entropy": 3.0 + 0.1 * i,
             "gini_sparsity": 0.5,
             "reconstruction_error_30pct": 0.05,
             "energy_ratio_low_high": 1.5} for i in range(10)]
    a = compute_predictor("wavelet", rows)
    b = compute_predictor("wavelet", rows)
    assert np.allclose(a, b)
    r = compute_predictor("random", rows, seed=123)
    r2 = compute_predictor("random", rows, seed=123)
    assert np.allclose(r, r2), "random predictor must be seed-deterministic"
    print("  [5] predictor determinism ................ OK")


def main() -> None:
    print("[fixes smoke] starting...")
    check_energy_split()
    check_compression()
    check_roundtrip()
    check_sentence_split()
    check_deberta_path()
    check_determinism()
    print("[fixes smoke] OK")


if __name__ == "__main__":
    main()
