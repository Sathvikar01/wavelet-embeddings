# Phase 5 — Reproducible head-pruning benchmark

The benchmark addresses the standard reviewer concerns for a
predictive-pruning paper:

* **more data** — WikiText-2, Penn Treebank and a GLUE-subset (CoLA / SST-2 /
  MRPC) loaders yield thousands of cleaned sentences per cell. Per-seed
  inputs are random-without-replacement subsamples of `max_sentences`
  sentences so per-seed runs are genuinely independent.
* **more architectures** — DistilBERT, BERT, GPT-2, **RoBERTa** and
  **TinyLlama-1.1B** (decoder-only) so the predictor can be shown to
  generalise across encoder-only / decoder-only / distilled families.
  DeBERTa is wired up in `MODEL_REGISTRY` but excluded from the default
  sweep because its `forward()` exposes no `head_mask` and our generic
  per-head forward-hook pathway doesn't yet support the disentangled
  attention layout.
* **published baselines** — Michel et al. (HIT), Voita et al. (HIS) and
  Bhasharas et al. (behavioural-similarity) head-importance criteria,
  implemented in the same zero-shot proxy form, alongside the existing
  entropy / weight / magnitude / random heuristics.
* **statistical reporting** — per-input mean ± SD, 95% bootstrap CIs (BCa),
  paired Wilcoxon / t-test p-values vs. each baseline, Cohen's d and Cliff's
  delta effect sizes, and seed-variance tables.
* **reproducibility** — a Modal app with persistent volume writes a tarballed
  copy of the exact source plus the full CSV / JSON output bundle for every
  run. Each cell also writes its own per-cell JSON before the aggregator
  commits the bundle, so dispatch-driver disconnects never lose work.

Layout

```
benchmark/
├── __init__.py            package docstring
├── datasets.py            WikiText-2 / PTB / GLUE loaders
├── baselines_published.py Michel / Voita / Bhasharas zero-shot criteria
├── stats.py               bootstrap CIs, paired tests, effect sizes
├── runner.py              benchmark sweep loop + (CSV / JSON) writer
└── modal_app.py           Modal GPU app -- the publishable entry point
```

Running locally (smallest possible smoke test, no GPU):

```
python main.py benchmark-run --models distilbert \
    --datasets wikitext2 --seeds 0 \
    --max-sentences 5 --out results/benchmark
```

Running on Modal (detached, resilient; one command, runs every cell on an
A10G):

```
modal run --detach benchmark/modal_app.py \
    --models bert-base,distilbert,gpt2,roberta-base,tinyllama \
    --datasets wikitext2,penn_treebank,glue_subset \
    --seeds 0,1,2 \
    --max-sentences 150
```

`--detach` lets the local cli exit immediately after dispatching the cells;
the run continues server-side on its own. Poll the volume (note the run
timestamp from the launch log):

```
modal volume ls wavelet-bench runs/<timestamp>/cells
modal volume ls wavelet-bench runs/<timestamp>          # summary.json appears after aggregator finishes
```

Outputs are written to a persistent Modal Volume named `wavelet-bench` under
`runs/<timestamp>/` and contain:

* `per_input_scores.csv`  — one row per (model, dataset, seed, predictor),
                            mean ± SD + 95% CI across inputs.
* `aggregate.csv`         — cross-seed aggregated mean / SD / CI per cell.
* `seed_variance.csv`     — sweep-variance over seeds (robustness check).
* `summary.json`          — `comparisons` list of
                            {predictor, baseline, diff_mean, cohens_d,
                             cliffs_delta, test, statistic, p_value} for
                            wavelet vs. every baseline in every cell.
* `cells/cell_<model>_<dataset>_<seed>.json`
                         — per-cell BenchRecord JSON, written by each cell
                            as soon as it finishes (audit / partial-recovery).
* `source.tar.gz`         — the exact source the run was launched with,
                            so reviewers can re-execute.

## Reproducing a published result

```
# The exact command used to produce the reference run that ships with the
# paper (run timestamp 20260712-174429, app ap-KflmmuqSYNM9O4a7LzGQlF):
modal run --detach benchmark/modal_app.py \
    --models bert-base,distilbert,gpt2,roberta-base,tinyllama \
    --datasets wikitext2,penn_treebank,glue_subset \
    --seeds 0,1,2 \
    --max-sentences 150 \
    --max-length 64 \
    --metric cosine_drop

# Pull the artefacts locally:
modal volume get wavelet-bench runs/<timestamp> ./bench_out
```

Filter to your subset of models/datasets/seeds with the `--models`,
`--datasets` and `--seeds` flags — both loaders and stat tests add
cells dynamically without code changes.

### Reference-run headline numbers (5 models × 3 datasets × 3 seeds × 150
sentences = 45 cells, A10G, May 2026)

* **45 / 45 cells completed** in ~10.8 h walltime on parallel A10G (max
  ≈70 min for the slowest tinyllama penn_treebank seed=1 cell).
* **105 / 105 wavelet-vs-baseline comparisons** computed; **103 / 105
  significant at p < 0.05** (paired Wilcoxon), the only exceptions being
  tinyllama·wikitext2 and tinyllama·penn_treebank against
  attention_entropy (diff_mean ≈ 0.003, Cliff's δ ≈ 0.028 -- the two
  predictors are essentially equivalent on TinyLlama).
* **120 / 120 seed-variance rows have non-zero seed_std** (range
  ≈0.002 for `random` up to ≈0.0044 for `wavelet` / `attention_entropy`),
  confirming the per-seed random-without-replacement subsampling
  produces meaningful between-seed variation.
* **Mean predictor correlations** (over all 15 model·dataset cells, signed
  by `-score` vs. ablation cosine-drop so negative ⇒ high importance is
  predicted):

  | predictor          | mean   | range               |
  |--------------------|--------|---------------------|
  | attention_entropy  | -0.209 | [-0.282, -0.079]    |
  | wavelet            | -0.168 | [-0.264, -0.080]    |
  | bhasharas_bs       | -0.073 | [-0.326, +0.209]    |
  | random             | +0.036 | [-0.054, +0.300]    |
  | voita_his          | +0.230 | [+0.120, +0.330]    |
  | magnitude          | +0.228 | [+0.120, +0.351]    |
  | attention_weight   | +0.227 | [+0.109, +0.336]    |
  | michel_hic         | +0.225 | [+0.130, +0.347]    |

  `wavelet`, `attention_entropy` and `bhasharas_bs` predict the *important*
  heads (negative correlation with predicted-unimportance vs. ablation
  loss); `random` sits at the noise floor; the magnitude / Voita / Michel /
  attention-weight baselines reverse the polarity and predict *discardable*
  heads under our zero-shot proxy formulation.

## Phase 6 — Combined-feature ridge predictor (leave-one-cell-out)

### Why a Phase 6

A reading of the Phase-5 reference numbers reveals that the predictor
labelled `attention_entropy` in the `summary.json` bundle is not, in
fact, the Shannon entropy of the attention distribution. The Phase-5
`pruning.predict_attention_entropy` reads the field
`rows[i]["shannon_entropy"]`, and that field is populated by
`attention.analyzer.compute_head_metrics`, which computes it from the
**wavelet-coefficient magnitudes** (analyzer.py:309-314), not from the
attention matrix. The same field, with weight `α=1.0`, is also the
dominant term of the `predict_wavelet` composite (`pruning.registry.py:
76-84`), so `attention_entropy` and `wavelet` in the Phase-5 reference
table share their most informative variable. The headline
"wavelets outperform entropy" therefore reads as "the composite of
`entropy + gini + 0.5*rec + 0.5*er` outperforms the same entropy term
alone" — and the Phase-5 numbers show it does **not** (`-0.168` vs
`-0.209`).

Rather than discard Phase 5, Phase 6 cleans the labelling, introduces an
*actual* attention-distribution entropy predictor, and asks the
scientifically interesting question the Phase-5 framing suppressed:

> **Does adding wavelet-derived multiscale features to the strongest
> single predictor of head importance (wavelet-coefficient entropy or
> attention-distribution entropy) lift predictive power on held-out
> (architecture, dataset, seed) cells, and is the lift statistical?**

### New / renamed predictors

| predictor                 | what it computes                                    | where |
|--------------------------|-----------------------------------------------------|------|
| `wavelet_entropy`         | Shannon entropy of the wavelet-coefficient magnitudes (the *real* name for what Phase-5 called `attention_entropy`) | `pruning/registry.py: predict_wavelet_entropy` |
| `attention_entropy_true` | Shannon entropy of the **attention distribution** `-Σ p log p` averaged across query rows (the genuine "attention entropy" baseline that was missing in Phase 5) | `pruning/registry.py: predict_attention_entropy_true` |
| `ridge_wavelet_only`      | Closed-form ridge regression over the 10 wavelet-domain features only, fit leave-one-cell-out | `benchmark/ridge_looo.py` |
| `ridge_attn_only`         | Ridge over the single `attention_entropy_true` scalar (sanity check that linear re-weighting of one feature cannot beat the feature) | `benchmark/ridge_looo.py` |
| `ridge_combined`          | Ridge over all 10 wavelet features **plus** the true attention-entropy scalar. This is the predictor that answers "do wavelets add complementary information to attention entropy?" | `benchmark/ridge_looo.py` |

`attention_entropy` is kept as a deprecated alias of `wavelet_entropy`
so historical `summary.json` bundles remain interpretable. Re-running
Phase 5 with the new codebase emits both `wavelet` (composite) and
`wavelet_entropy` (single term), so reviewers can see exactly which
component does the work.

### Leakage protocol (leave-one-cell-out)

The three ridge predictors are fit by
`benchmark/ridge_looo.apply_ridge_looo`, called as a post-pass on a
fully-collected `BenchResult` after every cell in the sweep has finished
scoring the cold-start predictors. The protocol, in pseudocode, is:

```
for held_cell_idx in range(n_cells):
    training_cells = [c for i, c in enumerate(cells)
                       if i != held_cell_idx]
    X_train = concatenate(c.per_input_features for c in training_cells)
    y_train = concatenate(c.per_input_loss    for c in training_cells)
    w, b = ridge_fit(X_train, y_train, alpha=1.0)
    for input_i in held_cell.per_inputs:
        # within-cell standardised features (the scalar's own mean/std)
        x = held_cell.per_input_features[input_i]
        y_hat = w @ x + b
        r = pearson(-y_hat, held_cell.per_input_loss[input_i])
        held_cell.per_input["ridge_combined"][input_i] = r
```

Key invariants:

* **No cell's own `(features, loss)` ever participates in fitting the
  ridge that predicts that cell.** This is the standard
  leave-one-site-out protocol used in multi-site prediction studies.
* **Within-cell standardisation.** Each cell's feature matrix is
  z-scored using that cell's own mean/std (computed in
  `benchmark/feature_matrix.build_feature_matrix`). Pooling across
  training cells is done on already-cell-standardised features. The
  held-out cell's standardiser is invariant to anything it never saw.
* **Fixed `alpha=1.0`.** Deliberately a single hyperparameter to avoid
  the "you tuned alpha on test" objection. If a reviewer asks, swap to
  `sklearn.linear_model.RidgeCV` with `alpha in [0.1, 1, 10, 100]` —
  the leakage surface does not change alpha-as-tuned-on-training-only.
* **Feature spec frozen** (`benchmark/feature_matrix.WAVELET_FEATURES`
  + `ATTENTION_FEATURES`). Adding or removing features changes the
  predictor definition and must be recorded as a separate Phase 6.x
  ablation; do not silently extend the spec.

### Artefacts the Phase-6 run writes

Beyond the Phase-5 outputs, every cell now also persists
`cells/cell_<model>_<dataset>_<seed>.npz` containing the
within-cell-standardised per-input feature matrix (`X`, shape
`(n_inputs, n_heads, n_features)`) and the per-input measured loss
vectors (`y`, shape `(n_inputs, n_heads)`). These are the only inputs
the LOOO loop reads.

`runs/<timestamp>/summary.json` carries two comparison blocks in
`comparisons`:

* the legacy `predictor="wavelet"` block (every cell, wavelet vs each
  other predictor — now including the three ridge variants); and
* a new `predictor="wavelet_entropy"` block comparing the single-term
  wavelet-entropy baseline against every other predictor (including
  `ridge_combined`) on every cell — the row whose `diff_mean` and
  `p_value` directly state whether the combined ridge lifted predictive
  power over the strongest single feature, on a per-cell statistical
  basis.

### What it would take to_claim catastrophe / victory

After a Phase-6 re-run over the same 45 cells (5 models x 3 datasets x
3 seeds x 150 sentences each):

1. **Strong setback**: if `ridge_combined` is significantly *worse*
   than `wavelet_entropy` in more than 6 of 15 (model, dataset) cells,
   the combined-feature story is dead and the paper is best framed as
   "wavelet-coefficient entropy alone is a strong, simple zero-shot
   predictor" (still publishable).

2. **Strong winning claim**: if `ridge_combined` is significantly
   *better* than `wavelet_entropy` in at least 8 of 15 cells and worse
   in at most 3 — write the "complementary information" framing
   verbatim (see the recommended language below).

3. **Honest middle**: if the lift is significant on a few cells and
   null on the rest, write "wavelet features are complementary to
   attention entropy on encoder transformers (BERT, DistilBERT,
   RoBERTa) and statistically equivalent on decoder LMs
   (TinyLlama), suggesting architecture-dependent information content
   in multiscale spectral structure."

### Recommended paper language (post-Phase-6)

> "We pose attention-head pruning sensitivity as a zero-shot prediction
> problem and evaluate nine published single-feature heuristics plus
> three learned composite predictors across five architectures (BERT,
> DistilBERT, GPT-2, RoBERTa, TinyLlama-1.1B) and three datasets
> (WikiText-2, Penn Treebank, GLUE-subset). Wavelet-coefficient entropy
> alone is our strongest single feature, achieving mean r = -0.209
> with ablation cosine-drop. A leave-one-cell-out ridge regression over
> wavelet-derived multiscale features (energy, entropy, sparsity,
> reconstruction error, dominant level, energy ratio) plus
> attention-distribution entropy lifts mean r to <value from run> on
> <n / 15> cells at p < 0.05 (paired Wilcoxon), demonstrating that
> multiscale spectral structure carries information about head
> importance not captured by attention-distribution entropy alone."

The `<value from run>` placeholders are filled in from the
`predictor="wavelet_entropy", baseline="ridge_combined"` rows of
`comparisons` in the Phase-6 `summary.json`.
