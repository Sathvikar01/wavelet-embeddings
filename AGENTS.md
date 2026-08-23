# AGENTS.md - Wavelet Embeddings project

## Commands

### Phase 1 (static embeddings)
```
python main.py extract --models bert-base distilbert gpt2
python main.py analyze --models bert-base distilbert gpt2 \
    --wavelets haar db4 sym4 coif2 --sample 500
python main.py all
```

### Phase 2 (contextual embeddings)
```
python main.py context-extract --models bert-base distilbert gpt2 [--anchors ...]
python main.py context-analyze --models bert-base distilbert gpt2 --wavelets db4
python main.py context-all
```

### Phase 3 (attention wavelet analysis)
```
python main.py attention-extract --models bert-base distilbert gpt2
    [--sentences "..."] [--max-length 32]
python main.py attention-analyze --wavelets haar db4 sym4 coif2
python main.py attention-all
```

### Phase 4 (predictive pruning validation)
```
python main.py pruning-analyze --models distilbert --wavelet db4
    [--sentences "..." "..."]
python main.py pruning-all --models bert-base distilbert gpt2 --wavelet db4
```

### Phase 5 (reproducible head-pruning benchmark)
```
# Local CPU smoke (small):
python main.py benchmark-run --models distilbert --datasets wikitext2 \
    --seeds 0 --max-sentences 5 --out results/benchmark

# Full GPU sweep on Modal (detached, resilient; ~5h on parallel A10G):
modal run --detach benchmark/modal_app.py \
    --models bert-base,distilbert,gpt2,roberta-base,tinyllama \
    --datasets wikitext2,penn_treebank,glue_subset \
    --seeds 0,1,2 --max-sentences 150

# Poll progress (the local entrypoint returns immediately under --detach):
modal volume ls wavelet-bench runs
modal volume ls wavelet-bench runs/<timestamp>/cells

# Download after the aggregator has written summary.json:
modal volume get wavelet-bench runs/<timestamp> ./bench_out
```
The Modal run schedules 45 (model, dataset, seed) cells via
``bench_cell.spawn(...)``; each cell writes a per-cell JSON to a persistent
`wavelet-bench` Volume at `runs/<timestamp>/cells/` as soon as it finishes (so
driver disconnects / client cancellations never lose work). A separate
`aggregate_run` Modal function polls the Volume every 2 min and, once all
expected cells have landed (deadline 8 h), writes the canonical CSV + JSON
bundle to `runs/<timestamp>/`. Download with
`modal volume get wavelet-bench runs/<timestamp> ./bench_out`.

### Smoke / unit tests
- Run `python smoke_test.py` for an offline (no HF download) end-to-end check
  of the Phase-1 wavelet / analysis / compression modules.
- Run `python smoke_test_phase3.py` for an offline check of the
  Phase-3 attention analyzer (no HF download).

### Notes
- No lint/typecheck config exists in this project.
- CSV writers must open with `encoding="utf-8"` because GPT-2 tokens contain
  non-ASCII chars (e.g. `Ġ`).
- Matplotlib >=3.10 removed `use_line_collection` and the `tab:color-format`
  combined strings; pass colors as `(color, alpha)` tuples.
- The HF tokenizer's `offset_mapping` returns plain lists of pairs when not
  using `return_tensors='pt'`; the ContextualExtractor uses `_as_list` to
  normalise this.
- Phase-3 attention analysis uses ``AttentionWaveletDecomposer`` (2-D DWT),
  not the Phase-1 ``make_decomposer`` (1-D). Only call Phase-3 functions
  with 2-D decomposer factories.
- Phase-4 ``HeadAblator`` issues an HF ``head_mask`` tensor; the forward
  call must pass ``head_mask=...`` explicitly. The ``run_model`` helper
  does this internally -- callers should use ``run_model(...,
  head_mask=...)``.
- ``PredictorValidationReport`` uses the keyword ``predictor`` (not
  ``predictor_name``).
- Phase-5 ``benchmark/`` is a self-contained package -- it re-uses
  ``embedding.extract.MODEL_REGISTRY``, ``pruning.runner.HeadAblator``,
  ``pruning.registry`` (existing predictors) and adds the published
  baselines in ``benchmark.baselines_published``. New model families
  (``roberta``, ``deberta``, ``llama``) were added to ``MODEL_REGISTRY`` /
  ``HeadAblator._discover_dims``; extend those two sites when adding more.
- Phase-5 ``MODEL_REGISTRY`` gains two optional ``ModelSpec`` fields:
  ``has_lm_head`` and ``is_decoder_only``. ``evaluation.task_loss`` uses
  ``is_decoder_only`` to gate last-token KL projection; encoder-only models
  fall back to cosine_drop when ``--metric kl_div_next_token`` is requested.
- Phase-5 ``datasets.py`` filters WikiText / PTB / GLUE lines with a
  sentence-end + length filter; lines like ``= = = History = = =`` are
  dropped. Penn Treebank is fetched from a raw mirror because the legacy
  ``ptb_text_only`` HF dataset is script-based and unsupported by
  ``datasets>=2.18``; on failure it falls back to the WikiText-2 *validation*
  split so the pipeline never hard-fails mid-run.
- Phase-5 ``stats.bootstrap_ci`` falls back to a percentile CI if scipy's
  ``bootstrap`` (BCa) raises; the pure-numpy sign-test fallback in
  ``paired_test`` is only used when scipy is missing.
- Phase-5 architecture-agnostic head ablation (``evaluation.task_loss``):
  ``_HeadAblationHooks`` registers a ``forward_pre_hook`` on every
  per-layer post-attention Linear/Conv1D and zeroes the head's slice of the
  concatenated context tensor in place. ``run_model_ablated`` uses it in
  place of HF's ``head_mask`` (which Llama silently drops in transformers
  4.42+). Targets: encoder.layer[L].attention.output.dense (BERT/RoBERTa),
  transformer.layer[L].attention.out_lin (DistilBERT), h[L].attn.c_proj
  (GPT-2 Conv1D -- in-place mutation visibility verified), and
  layers[L].self_attn.o_proj (Llama).
- Phase-5 eager attention pin (``embeddings.extract.EmbeddingExtractor.load``
  and ``attention.extractor.AttentionExtractor.load``): ``bert``, ``roberta``,
  ``gpt2`` and ``llama`` families are loaded with
  ``attn_implementation="eager"``. SDPA silently falls back to a slow manual
  attention path whenever ``output_attentions=True`` is set; pinning eager
  upfront removes the warning *and* roughly halves per-cell walltime on
  TinyLlama / RoBERTa.
- Phase-5 resilent Modal design (``benchmark.modal_app``): ``run_bench``
  uses ``bench_cell.spawn(*job)`` (fire-and-forget) per cell rather than
  ``.starmap`` so that client/local_entrypoint disconnects don't cancel
  spawned cells. Each ``bench_cell`` writes its
  ``cell_<model>_<dataset>_<seed>.json`` to the Volume and commits before
  returning. A separate ``aggregate_run`` Modal function (also spawned)
  polls the Volume every 120 s until all expected cells land (or 8 h
  deadline) and then writes the canonical CSV + ``summary.json`` +
  ``source.tar.gz`` bundle. Always invoke with ``modal run --detach`` so the
  local process can exit without cancelling anything.
- Phase-5 seed-aware subsampling (``benchmark.runner.run_cell``,
  ``benchmark.datasets._subsample``): ``_subsample`` is a no-op (returns
  the full cleaned corpus); ``run_cell`` then takes
  ``rng.choice(len(sentences), size=max_sentences, replace=False)`` per
  seed so per-seed inputs genuinely differ. Earlier deterministic-stride
  subsampling was making per-seed means identical and seed_std=0 across
  every (model, dataset, predictor) row -- the random-without-replacement
  selection produces meaningful seed_std (≈0.002 for random up to ≈0.0044
  for wavelet / attention_entropy at 150-cell scale).
- Phase-5 reference run (5 models × 3 datasets × 3 seeds × 150 sentences =
  45 cells, A10G): dispatched at 2026-07-12 17:44:29 IST under app
  ``ap-KflmmuqSYNM9O4a7LzGQlF``; aggregator committed outputs at
  ~04:31 IST (~10.8 h wallclock including the slowest tinyllama
  penn_treebank seed=1 cell). Headline numbers: 105/105 comparisons vs
  every non-wavelet baseline, 103/105 significant at p<0.05; attention_entropy
  and wavelet are statistically equivalent on TinyLlama's two LM cells
  (the two non-significant cells).
- Phase-5 / Phase-6 predictor naming pitfall: the Phase-5 predictor
  ``attention_entropy`` is **not** the Shannon entropy of the attention
  distribution. ``pruning.predict_attention_entropy`` (legacy
  ``registry.py``) reads ``rows[i]["shannon_entropy"]``, which
  ``attention.analyzer.compute_head_metrics`` populates from the
  wavelet-coefficient magnitudes (analyzer.py:309-314). The same wavelet
  entropy is the dominant weighted term of ``predict_wavelet``
  (registry.py:76-84, weight α=1.0). Phase 5 was therefore comparing
  "wavelet-coefficient entropy alone" vs "wavelet-coefficient entropy +
  gini + 0.5*rec + 0.5*er" — not "attention entropy vs wavelet
  composite".

  Phase 6 renames without breaking Phase 5:
  * ``wavelet_entropy`` is the new name for the wavelet-coefficient
    entropy; the legacy ``attention_entropy`` is a back-compat alias
    of it (kept so historical ``summary.json`` lists remain readable).
  * ``attention_entropy_true`` is the **new** baseline that computes
    ``-Σ p log p`` over the actual attention distribution
    (``pruning.predict_attention_entropy_true``).
  * Three ridge predictors ``ridge_wavelet_only``,
    ``ridge_attn_only`` and ``ridge_combined`` are added; they are fit
    leave-one-cell-out by ``benchmark.ridge_looo.apply_ridge_looo``
    after every cell has finished scoring the cold-start predictors
    (no per-cell leakage, fixed ``alpha=1.0``, within-cell
    standardisation; cells persist ``cells/cell_*.npz`` with
    ``X.shape=(n_inputs, n_heads, n_features)`` so the aggregator
    can run the LOOO loop).

  See ``benchmark/README.md`` Phase-6 section for the recommended
  paper language conditioned on the comparison rows whose ``predictor``
  is ``"wavelet_entropy"`` and whose ``baseline`` is
  ``"ridge_combined"`` in the new ``summary.json``.
- Windows console + Modal CLI Unicode: the Modal CLI emits progress chars
  like ``\u2713`` (✓) and spinner glyphs that the default Windows cp1252 /
  ``charmap`` codec cannot encode -- spawning ``modal`` from PowerShell
  without a UTF-8 environment raises ``'charmap' codec can't encode
  character '\u2713'`` and the process dies mid-dispatch. Set both
  ``PYTHONIOENCODING=utf-8`` and ``PYTHONUTF8=1`` (and ``chcp 65001`` if
  invoking interactively) before any ``modal run`` invocation. When
  spawning Modal in a hidden / detached Windows process via
  ``Start-Process``, set these env vars on the parent PowerShell session
  first so the child inherits them.

## 2026-08 fix pass (verified by smoke_test_fixes.py)

- ``analysis/energy.py::low_freq_energy`` selected the *finest* detail
  levels instead of the coarsest (index algebra bug), making
  ``energy_ratio_low_high == (approx + fine)/fine``. Fixed to bucket the
  coarsest ``n//2`` levels; matches ``attention.analyzer.compute_head_metrics``.
  All Phase-1/2 artifacts were re-generated after the fix (old CSVs kept
  under ``results/_archive_pre_fix/``).
- ``benchmark/datasets.py::_split_into_sentences`` only flushed its buffer
  at paragraph ends, so multi-sentence lines stayed merged into one input.
  Now each completed fragment flushes immediately; line-final punctuation
  still completes wrapped sentences. WikiText corpora now yield true
  single sentences (~70k from wikitext2 train).
- DeBERTa head ablation was a silent no-op: ``_per_head_o_proj_path``
  returned ``None`` for family ``deberta`` so no hook registered and every
  per-head effect was ~0. DeBERTa(v1/v2) mirror BERT
  (``encoder.layer[L].attention.output.dense``) -- routed accordingly.
- ``benchmark/baselines_published.predict_bhasharas_bs`` fallback read a
  nonexistent metric key (``dominant_frequency_level``); fixed to
  ``dominant_level``.
- ``experiments/compare_models.py`` seeded numpy with Python ``hash(name)``
  (per-process randomised) -- replaced with ``zlib.crc32`` so sampled vocab
  is stable across runs.
- ``analysis/compression.py``: ``energy_retained`` was hardwired 0.0 --
  now computed exactly (mirrors ``_threshold_details`` selection); the
  redundant re-``decompose`` per ratio was removed;
  ``compression_ratio`` accounting now counts zeros over details only,
  matching what ``reconstruct`` actually zeroes.
- Phase-4 predictor validation re-ran the full n_heads x n_sentences
  ablation sweep once PER PREDICTOR (7x waste). The sweep is
  predictor- AND wavelet-independent: it is now computed once per
  snapshot in ``cmd_pruning_analyze`` / ``validate_all_predictors`` and
  shared via ``validate_predictor(..., cached_effects=...)``.
- Dead code removed: ``wavelets.base.reconstruct`` debug branch,
  unused imports (runner, registry, spectrum, compare_models).
- ``visualization/attention_wavelets.py`` was an empty stub; now implements
  per-head wavelet subband stacks (``plot_head_wavelet_stack``,
  ``save_stacks_for_heads``).
- New tooling: ``run_eval.py`` (eval-set driver: phase3-extract /
  phase3-analyze / phase4 / benchmark-local), ``data/eval_sentences.json``
  (12-sentence curated eval set), ``summarize_phase3.py`` (per model x
  wavelet aggregation of all attention snapshots),
  ``smoke_test_fixes.py`` (regression checks for this pass),
  ``experiments/dim_reduction.py`` (compression shootout at EQUAL
  byte budgets: PCA / random-projection / raw-top-k / wavelet-sparse,
  each with fp32 and int8 payloads, plus dense product quantization;
  PQ dominates below ~10% budgets, int8 stacking strictly beats fp32
  sparse, PCA never wins -- see ``results/dim_reduction/``),
  ``benchmark_local.py`` (crash-resumable per-cell local benchmark:
  pickles each finished cell under ``results/benchmark/records/``, then
  finalizes with ridge LOOO + ``write_results``; use it instead of the
  serial ``run_eval.py benchmark-local`` on a flaky machine),
  ``_bg.py <logfile> <script> [args]`` (detached Windows background
  runner that redirects at the Python level so Start-Process returns
  immediately), and ``main.py attention-analyze --only <substr>`` for
  sharding snapshots across parallel processes.
- Local end-to-end results refreshed under ``results/``: 36 eval
  snapshots x 4 wavelets analysed (Phase 3); pruning validation on one
  snapshot per model against the full eval set (Phase 4); local Phase-5+6
  benchmark with ridge LOOO under ``results/benchmark/`` (9 cells =
  bert-base/distilbert/gpt2 x wikitext2 x seeds 0/1/2, 12 seed-distinct
  sentences each; summary.json + aggregate/comparisons/correlation/
  seed_variance/feature_importance CSVs + cells/*.npz feature matrices).
- SIGN-CONVENTION PITFALL (misreading this inverts every conclusion):
  ``predictor_correlations.csv``, ``per_head_validation.csv``,
  ``BenchRecord.per_input`` and every ``comparisons.csv`` row store
  ``corr(predicted_unimportance, measured_loss)`` where
  ``predicted_unimportance = -score``. Because predictor scores mean
  "importance / keep" (higher = keep) and the batched sweep prunes the
  LOWEST-score heads, **more-negative stored r = better pruning
  ranking**. Empirical proof (bert-base db4, bottom-10%-by-lowest-score
  ablation damage): attention_entropy_true 0.00056 < wavelet 0.00072 <
  wavelet_entropy 0.00077 < random 0.00256 << magnitude 0.00318 <
  attention_weight 0.00481 -- while their stored pearson_r signs point
  the opposite way (+0.33 for magnitude/attention_weight, which are
  therefore ANTI-predictive at single-head granularity, not "the
  winners"). The ridge predictors fit y_hat ~= loss and store
  corr(-y_hat, loss): same convention, consistent with cold predictors.
  When in doubt, trust only the batched sweep's raw cosine_drop column,
  which is orientation-free.
- Local benchmark headline (9 cells, small-n caveat): wavelet-side
  rankings beat baselines in 72/105 comparisons;
  ``wavelet_feature_model`` significantly beats every mass/published
  baseline on bert-base & distilbert (p<=0.0024, |d| up to 9), ties
  ``wavelet_only_model`` on bert-base (p=0.30), lifts significantly over
  it on distilbert (diff -0.074, p=5e-4); gpt2 signal is compressed
  (single-head ablation drops ~1e-4 there). Seed std 0.007-0.018
  (non-degenerate). The 45-cell Modal GPU reference remains the
  high-power evidence.
- Offline runs: set ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1``
  once models/datasets are cached -- a transient network blip during an
  HF HEAD check otherwise crashes long jobs mid-flight (observed).
  Beware OneDrive sync churn when results contain thousands of small
  files; it can throttle disk I/O enough to dominate wallclock.

## Paper (IEEE)

- ``paper/ieee_paper.tex`` + compiled ``paper/ieee_paper.pdf`` (5 pp.,
  IEEEtran conference style). Author block: Sathvik A R, PES University.
- Figures auto-generated by ``make_paper_figs.py`` into ``paper/figs/``
  from result CSVs; generalization table regenerated by
  ``gen_gen_table.py`` -> ``paper/gen_table.tex``. Recompile:
  ``cd paper; pdflatex ieee_paper.tex`` (twice), MiKTeX AutoInstall=1.
- Generalization shoot-out added for roberta-base / deberta-base /
  tinyllama (D=2048): UNIVERSAL findings = int8 stacking strictly beats
  fp32 sparse at every budget on every model; PQ dominates <10% budgets
  everywhere; PCA truncation never wins overall. CONDITIONAL finding =
  wavelet-vs-raw-topk ranking follows table heavy-tailedness: wavelet
  wins on heavy-tailed BERT/DistilBERT tables, raw top-k int8 wins on
  isotropic RoBERTa/DeBERTa/TinyLlama tables. CSVs under
  ``results/dim_reduction/``; extraction via
  ``python main.py extract --models roberta-base deberta-base tinyllama``.
- Reviewer-fix pass (paper v2, 6 pp.): (a) compression budgets now
  charged at EFFECTIVE bytes (payload + metadata + amortised shared
  params; see ``experiments/effective_storage.py`` ->
  ``results/dim_reduction/all_effective_bytes.csv`` +
  ``pooled_effective_table.csv``; PQ codebook = 26 B/tok @768 / 65 B/tok
  @2048 at these vocab sizes); (b) downstream task validation added
  (``experiments/downstream_pruning.py``, SST-2 acc for fine-tuned
  BERT/DistilBERT + GPT-2 WikiText-2 PPL under ranked head ablation;
  ``results/downstream/downstream_pruning.csv``): rankings REORDER
  across objectives -- attention_entropy_true best on SST-2 but
  catastrophic on GPT-2 PPL, wavelet composite the reverse,
  Frobenius-mass anti-predictive at rep level yet strong on DistilBERT
  SST-2; (c) 45-cell GPU reference bundle pulled from Modal volume
  ``runs/20260712-174429`` into ``results/benchmark_gpu/``: wavelet-side
  vs non-wavelet baselines better in 87/90, ALL 90 significant p<0.05;
  DeBERTa cells absent from that run (predate ablation fix); TinyLlama's
  bhasharas_bs is anomalously strong (-0.32). GPT-2 head ranking requires
  ``attn_implementation="eager"`` (SDPA silently returns no attentions).
  Paper renamed baselines: magnitude -> Frobenius mass ||A||_F,
  attention_weight -> max-column mass; composite documented as raw-scale
  heuristic with ridge models as principled counterpart; related work
  expanded to 27 refs.
- Paper v3 (second review pass): DCT-II / rFFT sparse controls added to
  ``experiments/dim_reduction.py`` (--only-methods dct_f32,fft_c64;
  CSV writes are now merge-aware so incremental runs keep old rows).
  Pooled result at equal EFFECTIVE bytes: DCT >= wavelet on cosine at
  every budget, Jaccard tied -> the compression advantage is GENERIC
  spectral sparsity, not wavelet-specific (paper claims updated
  accordingly). Abstract leads with the downstream-inconsistency finding;
  87/90 wording fixed; BH-FDR stated (90/90 and 103/105 survive);
  GPT-2 columns flagged as noise floor in Table I caption; ridge rows
  foregrounded in benchmark table.
- Paper v4 (third review pass): reframed as a falsifiable DIAGNOSTIC
  question ("can cheap spectral statistics detect redundancy uniformly,
  and where does that fail?"); title changed accordingly. NEW experiments:
  leave-one-model-out ridge (``benchmark/lomo_eval.py``, combined ridge
  holds at -0.28/-0.40/-0.23 held-out) and downstream transfer of
  COMPRESSED tables (``experiments/downstream_compression.py`` ->
  ``results/downstream/downstream_compression.csv``): wav+int8 nearly
  lossless on SST-2 encoders (-0.3 pts @33%), PCA catastrophic there
  (+36 pts), GPT-2 DIVERGES under every recipe (>1000x ppl; tied
  wte/lm_head propagates table noise into the softmax). All section
  cross-refs now use \ref labels (no hardcoded V-x); baselines defined
  with equations; unit-of-analysis paragraph added; abstract slimmed.
  NOTE: an edit-tool failure once truncated ieee_paper.tex to 0 bytes --
  reconstructed from session history; keep ``paper/ieee_paper_backup.tex``
  in sync after major edits.
- Paper v5 (fourth review pass): published-baseline framing replaced by
  "zero-shot proxies inspired by published criteria" (michel_inspired /
  voita_inspired / behavioural_similarity; no uncited-author names);
  encoder "freely compress"/"untie decoder" claims rescoped to the two
  tested settings with tied-embedding divergence framed as hypothesis;
  downstream compression table extended with DCT+int8 / TopK+int8
  (``experiments/downstream_compression.py`` is now resumable per-config):
  ALL int8-sparse bases are near-lossless on both SST-2 encoders, but on
  tied-embedding GPT-2 only raw-coordinate TopK@33% survives (1.2x ppl)
  while wavelet/DCT diverge >1000x -> "where the perturbation lands"
  matters more than reconstruction similarity; the LEARNED combined ridge
  was finally validated downstream too (``experiments/downstream_ridge.py``,
  LOMO-fitted weights): it degrades like the composite on SST-2
  (+14.3/+41.2/+41.5 pts BERT) -- learning weights against representation
  damage does NOT repair objective mismatch (thesis sharpened); conclusion
  softened ("characterise which formats perform best", not "identify
  exactly"); unit-of-analysis + FDR statements retained.
