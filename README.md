# Wavelet Analysis of Transformer Embeddings

[![Public](https://img.shields.io/badge/visibility-public-brightgreen)](https://github.com/Sathvikar01/wavelet-embeddings) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Paper](https://img.shields.io/badge/paper-IEEE%20PDF-red)](paper/ieee_paper.pdf) [![Phases 1-6](https://img.shields.io/badge/phases-1--6-purple)](#layout)

> Multi-scale DWT on static embeddings, contextual spectra, attention-head wavelets, predictive pruning validation, and a reproducible 45-cell benchmark (BERT/DistilBERT/GPT-2/RoBERTa/TinyLlama × WikiText-2/PTB/GLUE × 3 seeds).

## Phase 1: Wavelet Analysis of Transformer Embeddings

A pipeline that treats every token embedding as a 1-D signal and applies
multilevel discrete wavelet transforms to reveal frequency / multiscale
structure that ordinary cosine analysis cannot see.

## What it does

For every token in a model's vocabulary (BERT-base, DistilBERT, GPT-2 Small):

* Computes **Haar / Daubechies-4 / Symlet-4 / Coiflet-2** decompositions.
* Extracts *approximation* and *detail* coefficients at every level.
* Measures **energy**, **entropy**, **sparsity**, **compression ratio**,
  **skewness**, **kurtosis**, **SNR**, **dominant frequency band**.
* Groups tokens by category (Noun / Verb / Adjective / Number / Punctuation /
  Rare / Frequent / Subword / Special) and asks: *"do different semantic
  categories occupy different frequency spectra?"*
* Compares BERT-base vs DistilBERT vs GPT-2 to test whether GPT learns
  smoother embeddings and whether BERT carries more high-frequency content.
* Runs a **neighbour analysis** on `king / queen / man / woman / apple /
  orange`, comparing **wavelet similarity** to standard **cosine similarity**.
* Runs a **compression experiment**: zero out 10/20/30/40/50 % of the smallest
  detail coefficients, reconstruct, and measure cosine, neighbour
  preservation and embedding drift.
* Visualises heatmaps, per-level energy distributions, reconstruction-error
  curves, and t-SNE / PCA projections before vs. after compression.
* Finally ranks every token by an **Information Score**:
  `entropy * log(1 + energy)` -- a proxy for how much frequency-domain
  information the embedding carries.

## Layout

```
wavelet_embeddings/
├── data/                    # extracted .npz files (default cache)
├── embeddings/
│   ├── extract.py           # Pull static token embeddings from HF models
│   └── loader.py           # Load .npz + token category heuristic
├── wavelets/
│   ├── base.py             # WaveletDecomposer & shared types
│   ├── haar.py             # Haar
│   ├── db4.py              # Daubechies-4
│   ├── symlet.py           # Symlet-4
│   └── coiflet.py          # Coiflet-2
├── analysis/
│   ├── energy.py           # Energy + per-level summaries + Information Score
│   ├── entropy.py          # Shannon entropy, skewness, kurtosis
│   ├── sparsity.py         # Gini, count-sparsity, compression ratio
│   └── compression.py      # Multi-ratio compression + neighbour preservation
├── visualization/
│   ├── heatmaps.py         # univariate heatmaps, per-level energy bars
│   ├── spectrum.py         # coefficient stem plot + reconstruction curves
│   └── tsne.py             # t-SNE / PCA before-after
├── experiments/
│   ├── compare_models.py   # BERT vs DistilBERT vs GPT-2
│   ├── compare_tokens.py   # categories + neighbour analysis
│   └── reconstruction.py   # per-token reconstruction under compression
├── main.py                 # CLI: extract | analyze | all
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Pull static embeddings from the three models (saves .npz)
python main.py extract --models bert-base distilbert gpt2

# 2. Run all experiments with the default set of wavelets
python main.py analyze --models bert-base distilbert gpt2 \
    --wavelets haar db4 sym4 coif2 --sample 2000

# 3. Or do it in one go
python main.py all --models bert-base distilbert gpt2 --wavelet db4
```

Outputs are written under `results/wavelet_<name>/...`:

```
results/
├── embeddings/
│   ├── bert-base_embeddings.npz
│   ├── distilbert_embeddings.npz
│   └── gpt2_embeddings.npz
└── wavelet_db4/
    ├── compare_models/
    │   ├── model_comparison.csv
    │   └── model_reconstruction_comparison.png
    └── bert-base/
        ├── token_category_comparison.csv
        ├── category_energy_ratio.png
        ├── neighbour_analysis.csv
        ├── neighbour_analysis.png
        ├── reconstruction/
        │   ├── reconstruction_results.csv
        │   ├── reconstruction_cosine_per_token.png
        │   └── reconstruction_neighborhood_preservation.png
        ├── coefficient_heatmap.png
        ├── energy_distribution.png
        ├── tokens_energy_heatmap.png
        ├── spectrum_<token>.png
        ├── tsne_before_after.png
        ├── pca_before_after.png
        └── information_ranking.csv
```

## Why wavelets on embeddings?

An embedding is conventionally treated as a feature vector.  Treating it as
a 1-D signal lets us ask **how** the information is structured across the
dimensions:

* **Approximation coefficients cA** capture coarse / global structure.
* **Detail coefficients cD_n** capture progressively finer-grained structure.

If certain token classes (numbers, punctuation, subwords...) consistently
have their energy concentrated in the high-frequency detail bands while
others (nouns, verbs) are concentrated in the low-frequency approximation
band, that is an *empirical signature of semantic organisation in
frequency space* -- independent of the transformer layers themselves.

The Information Score (`entropy * log(1 + energy)`) then ranks tokens by
how densely packed their wavelet-bound information is, and lets us test
whether high-density tokens correspond to **semantically important or
robust** tokens.

## Implementation notes

* Embeddings with non-power-of-two dim (e.g. 768) are symmetrically padded
  to the next power of two (1024) before DWT and cropped back after
  reconstruction.
* Compression thresholds operate on *combined detail coefficients* across
  all levels simultaneously (not per-band), matching the spec.
* Neighbour preservation uses **Jaccard overlap** of the top-k cosine
  neighbours between the original and reconstructed matrices, computed
  against a sub-sample of the vocabulary (default 500) to keep the
  experiment tractable on a laptop.
* All metrics are persisted to CSV; the visualisations are PNGs.
