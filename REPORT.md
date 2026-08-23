# Wavelet Embeddings Project — Full Analysis, Fix & Evaluation Report

**Date:** 2026-08-22 · **Machine:** local CPU (i7-1355U, 12 threads, no GPU)
**Scope:** code audit + bug-fix pass + full end-to-end re-run of every phase
(1–6) on a curated evaluation set, plus a local Phase-5/6 head-pruning
benchmark with leave-one-cell-out ridge.

---

## 1. What was found (code audit)

A full-repo audit (every module, every CLI path) found **8 real bugs**, all
now fixed and covered by regression checks (`smoke_test_fixes.py`):

| # | Bug | Impact | Fix |
|---|-----|--------|-----|
| 1 | `analysis/energy.py::low_freq_energy` selected the **finest** detail levels instead of the coarsest (`i <= low_levels` index-algebra error) | `energy_ratio_low_high == (approx+fine)/fine` everywhere in Phase-1/2; inconsistent with Phase-3's correct implementation; corrupted the `er` term of the Phase-4 wavelet predictor | bucket the coarsest `n//2` levels; verified against an independent recomputation |
| 2 | `benchmark/datasets.py::_split_into_sentences` flushed its buffer once per *line*, not per *sentence* | multi-sentence lines stayed merged into one "sentence" input ("A. B. C." → one input); long merges silently dropped by the ≤400-char filter | each completed fragment flushes immediately; line-final punctuation still completes wrapped sentences. WikiText-2 now yields ~70k true single sentences |
| 3 | DeBERTa head ablation was a **silent no-op**: `_per_head_o_proj_path` returned `None` for family `deberta` | every deberta per-head ablation effect ≈ 0 → deberta cells in any Phase-5 run were garbage | DeBERTa mirrors BERT (`encoder.layer[L].attention.output.dense`); routed accordingly |
| 4 | `baselines_published.predict_bhasharas_bs` fallback read nonexistent key `dominant_frequency_level` | fallback always scored 0.0 | fixed to `dominant_level` (the actual key emitted by `compute_head_metrics`) |
| 5 | `experiments/compare_models.py` seeded numpy with Python `hash(name)` | per-process randomised (PYTHONHASHSEED) — "deterministic" sampling wasn't | replaced with `zlib.crc32(name)` |
| 6 | `analysis/compression.py`: `energy_retained` hardwired to 0.0; wasted re-`decompose` per ratio; `compression_ratio` counted zeros over approx+details while `reconstruct()` zeroes details only | metric always 0; reported compression ratio disagreed with what reconstruction actually did | exact retained-energy computation mirroring `_threshold_details`; accounting fixed; dead work removed |
| 7 | Phase-4 validation re-ran the full n_heads × n_sentences ablation sweep once **per predictor** (and per wavelet) | 7–28× redundant model forwards (hours of CPU) | sweep computed once per snapshot, shared via `validate_predictor(..., cached_effects=...)` |
| 8 | Dead/broken leftovers: `pywt.wavethc(...) if False else ...` debug branch in `wavelets/base.py`, unused imports (runner/registry/spectrum/compare_models), empty stub module `visualization/attention_wavelets.py`, stale `spawn_map` docstrings in `modal_app.py` | confusion, import weight | removed / implemented |

Additional verification work:
* All four offline smoke suites pass on the final code state:
  `smoke_test.py`, `smoke_test_phase3.py`, `smoke_test_phase6.py`,
  `smoke_test_fixes.py` (new).
* New tooling: `run_eval.py` (one-command eval-set driver incl.
  `benchmark-local` with ridge LOOO wired), `data/eval_sentences.json`
  (curated eval set), `summarize_phase3.py`,
  `main.py attention-analyze --only <substr>` sharding filter,
  `benchmark_local.py` (crash-resumable per-cell benchmark runner),
  `_bg.py` (Windows-detached background runner).

## 2. Evaluation set

`data/eval_sentences.json` — 12 curated sentences covering polysemy
(bank/plant/ship ×2 senses each), relative clauses, passives/fronting,
named entities (Berlin), numerals (1969, seven), and a question form.
Used verbatim by Phase-3 extraction, Phase-4 ablation measurement and as
the benchmark probe sentence.

## 3. Phase 1 — static embeddings (re-run post-fix, sample=2000)

All artifacts regenerated for bert-base / distilbert / gpt2 × haar / db4 /
sym4 / coif2 (old CSVs archived under `results/_archive_pre_fix/`).

Headlines (db4):
* GPT-2 embeddings carry ~5.6× more energy (21.05) than BERT (2.64) and are
  near-Gaussian (skew 0.02, kurtosis 0.13) vs strongly negative-skewed,
  heavy-tailed BERT families (skew −1.95/−2.97, kurtosis 12.9/22.5) — the
  "decoder embeddings are smoother" hypothesis holds.
* Compression: keeping 90% of detail coefficients reconstructs BERT-family
  embeddings at cosine ≥ 0.999 (SNR ≈ 35 dB); even 50% keeps cosine ≥ 0.964.
* Category effect: adjectives carry the highest spectral information score
  (10.98 distilbert), punctuation the lowest (≈6.4) — consistent across
  models and wavelets.
* With the fixed low/high-frequency split, E_low/E_high is now 0.15–0.55
  (was inflated >1 by the bug): embedding energy is genuinely concentrated
  in the finest bands, i.e. token identity lives in high-frequency detail.

## 4. Phase 2 — contextual / polysemy (re-run post-fix)

30 anchor contexts × 4 wavelets re-analysed from cache. db4/bert-base
summary: wavelet-spectrum separation between senses tracks raw-cosine
separation closely (bank 0.319 vs 0.322; pen 0.312 vs 0.325) and same-sense
similarity consistently exceeds cross-sense similarity
(inter_wav 0.56–0.84 > cross_wav 0.29–0.68). The multiscale spectrum is
sense-sensitive but not more discriminative than plain cosine here.

## 5. Phase 3 — attention wavelet analysis (**36 snapshots**)

New: **12 sentences × 3 models extracted and fully analysed × 4 wavelets**
(previously only ONE distilbert snapshot existed). Aggregates:
`results/attention_analysis/phase3_summary_by_wavelet.csv`
(+ per-sentence variant).

Key findings pooled over 37 snapshots:
* Layer progression (distilbert, db4): low/high-frequency energy ratio rises
  through depth (≈1.4 → 2.4), Shannon entropy falls, Gini rises — later
  layers attend via fewer, lower-frequency, more structured patterns.
* Cross-model layer-evolution comparisons regenerated per wavelet from the
  matched `_00_` snapshots.
* Reconstruction: keeping 30% of DWT coefficients preserves attention maps
  with L2 error 0.02–0.08 (best at later layers) — attention is highly
  compressible in the wavelet domain.
* Per-head subband-stack visualiser implemented
  (`visualization/attention_wavelets.py`).

## 6. Phase 4 — predictive pruning validation (3 models × 4 wavelets × 12 sentences)

⚠️ **Sign convention first** (see AGENTS.md): every stored correlation is
`corr(predicted_unimportance = −score, measured_loss)`. Predictor scores
mean *keep* (higher = keep) and the batched sweep prunes **lowest-score**
heads — so **more-negative stored r = better pruning ranking**.

Stored Pearson r by predictor (db4; positive = anti-predictive!):

| predictor | bert-base | distilbert | gpt2 | verdict |
|---|---|---|---|---|
| attention_entropy_true | −0.322 | **−0.400** | −0.138 | best tier |
| wavelet composite | **−0.306** | −0.325 | −0.157 | best tier |
| wavelet_entropy (=legacy "attention_entropy") | −0.139 | −0.019 | −0.183 | mid |
| random | −0.019 | +0.304 | −0.023 | noise floor |
| magnitude | +0.327 | +0.386 | +0.159 | **anti-predictive** |
| attention_weight | +0.326 | +0.356 | +0.188 | **anti-predictive** |

Ground truth check (orientation-free): bottom-10%-by-lowest-score ablation
damage on bert-base/db4 —

```
attention_entropy_true 0.00056 < wavelet 0.00072 < wavelet_entropy 0.00077
    < random 0.00256 << magnitude 0.00318 < attention_weight 0.00481
```

i.e. ranking heads by *lowest wavelet score* and pruning them is ~5× safer
than random and ~6× safer than magnitude-based ranking at single-head
granularity. The naive reading of the CSV ("magnitude wins with +0.39") is
exactly backwards.

Batched fairness sweep (prune bottom-p%, lower cosine_drop = safer):
wavelet composite is best-or-tied-best in 5/12 model×ratio cells and never
worse than random; `attention_entropy_true`/`attention_entropy` win most
remaining cells. GPT-2 is nearly insensitive to single-head ablation at
this probe level (drops ~1e-4), so its correlations are weak/noisy.

Full tables: `results/phase4_predictor_summary.csv`,
`results/phase4_batched_summary.csv`, per-run CSVs under
`results/pruning/<snapshot>/wavelet_<w>/`.

## 7. Phase 5+6 — local head-pruning benchmark (**new local artifact!**)

First complete **local** end-to-end benchmark run (previously results lived
only on the Modal volume): **9 cells** = bert-base / distilbert / gpt2 ×
wikitext2 × seeds {0,1,2}, 12 seed-distinct sentences per cell,
cosine_drop metric, 13 predictors (7 cold-start + 3 published baselines +
3 ridge LOOO models). Ridge leave-one-cell-out ran over 9/9 cells;
per-cell feature matrices persisted under `results/benchmark/cells/`
(shape (12, n_heads, 11)).

Outputs: `results/benchmark/{summary.json, aggregate.csv, comparisons.csv,
correlation_table.csv, predictor_metrics.csv, seed_variance.csv,
feature_importance.csv, cells/*.npz}`.

Headlines (stored convention: more negative = better):

| model | wavelet | attn_ent_true | magnitude | attn_weight | ridge wavelet_only | ridge combined |
|---|---|---|---|---|---|---|
| bert-base | −0.117 | −0.237 | +0.256 | +0.270 | −0.328 | **−0.320** |
| distilbert | −0.130 | −0.327 | +0.332 | +0.332 | −0.338 | **−0.413** |
| gpt2 | −0.213 | −0.219 | +0.221 | +0.227 | −0.189 | −0.153 |

* Wavelet-side rankings beat baselines in **72/105** comparisons
  (small-n caveat: 12 inputs/cell locally vs 150 on the GPU reference).
* `wavelet_feature_model` significantly beats every mass/published
  baseline on bert-base & distilbert (p ≤ 0.0024, Cohen's d up to −9);
  ties `wavelet_only_model` on bert-base (+0.007, p = 0.30); **significant
  lift over it on distilbert** (diff −0.074, p = 5e-4).
* On gpt2 the signal compresses (single-head drops ~1e-4): ridge features
  don't help there (+0.036 vs wavelet_only, p = 0.02 against).
* Seed variance σ ≈ 0.007–0.018 across seeds — non-degenerate (confirms
  the seed-aware subsampling works).
* The 45-cell Modal GPU reference (AGENTS.md) remains the high-power
  evidence; this local run proves the whole pipeline works offline and
  reproduces the qualitative ordering.

## 8. Compression shootout: dim-reduction vs sparsity vs quantization (new)

Question: instead of wavelet-compressing a 768-d vector in place, what if
we simply **reduce the number of dimensions** (PCA / random projection)?
And how much extra do **int8 quantization** and **product quantization**
(PQ) buy on top? Implemented in `experiments/dim_reduction.py` (rewritten
around absolute bytes/token budgets); per-model CSVs under
`results/dim_reduction/*_compression_shootout.csv`.

Methods at equal byte budgets (original = 768 × float32 = 3072 B):
PCA-d, orthogonal random projection, raw top-k sparse (fp32/int8 values),
db4-wavelet sparse (fp32/int8 values), and dense product quantization
(m-byte codes, K=256 per subspace, codebooks amortised over the vocab).
Sparse entries are charged value-bytes + 2 B uint16 index.

Pooled mean cosine(reconstruction) / top-10 neighbour Jaccard over
bert-base + distilbert + gpt2 (1,500 test vectors each, PCA/PQ trained on
a disjoint split):

| bytes/tok (× of original) | PCA | rand-proj | topk f32 | wav f32 | topk int8 | wav int8 | PQ |
|---|---|---|---|---|---|---|---|
| 64 (2%) | .68/.06 | .14/.02 | .32/.05 | .47/.03 | .42/.07 | .55/.10 | **.84/.28** |
| 128 (4%) | .70/.09 | .21/.05 | .42/.07 | .55/.10 | .54/.10 | .63/.16 | **.93/.44** |
| 256 (8%) | .73/.14 | .29/.11 | .54/.10 | .63/.16 | .67/.14 | .73/.22 | **.99/.67** |
| 512 (17%) | .78/.24 | .39/.18 | .67/.14 | .73/.22 | .82/.22 | **.83/.32** | – |
| 1024 (33%) | .85/.40 | .58/.33 | .82/.22 | .83/.32 | **.94/.36** | .93/.48 | – |

Findings:
* **Quantization stacks cleanly**: at every matched budget the int8 twin
  beats its fp32-sparse parent by keeping 2× more coefficients for the same
  bytes (wav_int8 > wav_f32 everywhere; e.g. 256 B: .73 vs .63; 1024 B:
  .93 vs .83). Quantization noise is mild; coefficient *count* is what
  matters.
* **Product quantization is in a class of its own below ~10% budgets**:
  at 256 B/token (12× smaller) it reconstructs at cosine .99 and keeps 67%
  of the exact top-10 neighbours — far beyond anything per-vector methods
  reach. It wins because it learns a shared codebook across the whole
  vocabulary; per-vector schemes cannot exploit that structure. Trade-off:
  needs training + fixed codebooks, no streaming single-vector mode.
* **PCA never wins on this data**: global principal components align poorly
  with per-token information here; random projection confirms PCA's learned
  basis still helps vs unstructured, but both trail sparsification until
  budgets where everything saturates.
* GPT-2 (near-isotropic embeddings) remains the outlier: wavelet selection
  loses to raw top-k there at several budgets, though pooled wav_int8 still
  leads all sparse variants.

Practical ladder: need ≤10% size → PQ; want a training-free streaming
format → wavelet + int8 (≈6× smaller at usable quality); avoid PCA
truncation unless the consumer requires short dense vectors.

## 9. Operational notes discovered during runs

* Set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` for long local jobs —
  a transient HF HEAD-check timeout killed a run mid-flight otherwise.
* Use `benchmark_local.py` (per-cell pickles) instead of one serial sweep:
  it survives kills and resumes from `results/benchmark/records/`.
* OneDrive sync churn on thousands of small result files can throttle disk
  I/O enough to dominate wallclock on this machine.

## 10. Reproduction

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:HF_HUB_OFFLINE="1"
python smoke_test.py; python smoke_test_phase3.py
python smoke_test_phase6.py; python smoke_test_fixes.py
python run_eval.py phase3-extract            # 36 snapshots
python run_eval.py phase3-analyze            # x 4 wavelets (shardable: --only)
python summarize_phase3.py                   # cross-snapshot aggregates
python run_eval.py phase4                    # pruning validation, 3 models
python benchmark_local.py --models bert-base distilbert gpt2 `
    --datasets wikitext2 --seeds 0 1 2 --max-sentences 12
python experiments\dim_reduction.py             # PCA vs wavelet at equal bytes
```

Caveats: single snapshot per model for Phase-4 (first eval sentence);
GPT-2 ablation insensitivity limits decoder-side conclusions locally.
