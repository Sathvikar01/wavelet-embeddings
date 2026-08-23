"""Downstream validation of the LEARNED ridge predictor.

The paper's strongest representation-level predictor (combined-feature
ridge, trained leave-one-MODEL-out so no weight ever saw the evaluated
architecture) is finally validated where the thesis demands: rank heads
by predicted damage, prune the bottom-p%, measure SST-2 accuracy /
WikiText-2 perplexity. Appends rows to
results/downstream/downstream_pruning.csv with predictor='ridge_combined'.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.downstream_compression import (
    get_weight, set_weight,
)
from experiments.downstream_pruning import rank_heads, SetAblator
from benchmark.feature_matrix import build_feature_matrix
from benchmark.lomo_eval import load_cells, GROUP_KEY
from benchmark.feature_matrix import feature_names
from benchmark.ridge_looo import _column_subset, _fit_ridge

OUT = "results/downstream/downstream_pruning.csv"
DEVICE = "cpu"


def fit_lomo(held_model: str, group: str = "wavelet_feature_model"):
    """Fit ridge on all cells whose model != held_model."""
    cells = load_cells(os.path.join("results", "benchmark", "cells"))
    full = feature_names(GROUP_KEY[group])
    Xs, ys = [], []
    for c in cells:
        if c["model"] == held_model:
            continue
        sub = _column_subset(c["X"].reshape(-1, c["X"].shape[-1]),
                              full, GROUP_KEY[group])
        Xs.append(sub)
        ys.append(np.concatenate([yy for yy in c["y"]]))
    return _fit_ridge(np.concatenate(Xs), np.concatenate(ys), 1.0)


RIDGE_VARIANTS = [("ridge_combined", "wavelet_feature_model"),
                   ("ridge_wavelet_only", "wavelet_only_model"),
                   ("ridge_attn_only", "attention_only_model")]


def main() -> None:
    from transformers import AutoModelForSequenceClassification, \
        AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    rows = []
    if os.path.isfile(OUT):
        rows = list(csv.DictReader(open(OUT, encoding="utf-8")))
    have = {(r["model"], r["predictor"], str(r["ratio"])) for r in rows}

    def record(model_name, pred, ratio, metric, value):
        if (model_name, pred, str(ratio)) in have:
            return False
        rows.append(dict(task=("sst2" if metric == "accuracy"
                                 else "wikitext2_ppl"),
                          model=model_name, predictor=pred, ratio=ratio,
                          metric=metric, value=round(value, 4)))
        return True

    DS_MODELS = ["textattack/bert-base-uncased-SST-2",
                  "distilbert-base-uncased-finetuned-sst-2-english",
                  "gpt2"]
    ridge_variants = [v for v in RIDGE_VARIANTS
                       if any((m, v[0], str(r)) not in have
                               for m in DS_MODELS
                               for r in (0.1, 0.2, 0.3))]
    if not ridge_variants:
        print("all ridge downstream rows already recorded; nothing to do",
              flush=True)
        return
    # ---------------- classifiers ----------------
    for hf_name, key in (("textattack/bert-base-uncased-SST-2", "bert"),
                          ("distilbert-base-uncased-finetuned-sst-2-english",
                           "distilbert")):
        variants = [v for v in ridge_variants
                     if any((hf_name, v[0], str(r)) not in have
                             for r in (0.1, 0.2, 0.3))]
        if not variants:
            continue
        tok = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_name, attn_implementation="eager").eval()
        ds = load_dataset("glue", "sst2", split="validation")
        texts = [x["sentence"] for x in ds]
        labels = np.array([x["label"] for x in ds])
        rng = np.random.default_rng(0)
        samples = [texts[i] for i in rng.choice(len(texts), size=8,
                                                  replace=False)]

        heads_all, rows_metrics, attn = [], [], []
        base_m = getattr(model, key)
        with torch.no_grad():
            for t in samples:
                enc = tok(t, return_tensors="pt", truncation=True,
                           max_length=64)
                out = base_m(**enc, output_attentions=True)
                for Li, A in enumerate(out.attentions):
                    for Hi in range(A.shape[1]):
                        attn.append(A[0, Hi].cpu().numpy())
                        from attention import (AttentionWaveletDecomposer,
                                                compute_head_metrics)
                        m = compute_head_metrics(
                            A[0, Hi].cpu().numpy(),
                            AttentionWaveletDecomposer("db4"))
                        m["layer"], m["head"] = Li, Hi
                        rows_metrics.append(m)
                heads_all += [(Li, Hi) for Li in range(len(out.attentions))
                               for Hi in range(A.shape[1])]
        Xh, _ = build_feature_matrix(rows_metrics,
                                      {"head_attention": attn},
                                      group="combined")
        fits = {name: fit_lomo(key, group)
                 for name, group in variants}
        Xmats = {GROUP_KEY[group]:
                  build_feature_matrix(rows_metrics,
                                        {"head_attention": attn},
                                        group=GROUP_KEY[group])[0]
                  for _, group in variants}

        def accuracy(ablate=None):
            ctx = _Null() if ablate is None else SetAblator(key, model, ablate)
            correct = 0
            with torch.no_grad(), ctx:
                for i in range(0, len(texts), 32):
                    enc = tok(texts[i:i + 32], padding=True,
                               truncation=True, max_length=128,
                               return_tensors="pt")
                    logits = model(**enc).logits
                    correct += int((logits.argmax(-1).cpu().numpy()
                                     == labels[i:i + 32]).sum())
            return correct / len(texts)

        clean = accuracy()
        print(f"[ridge-ds] {hf_name} clean={clean:.4f}", flush=True)
        for name, group in variants:
            w, b, _ = fits[name]
            scores = Xmats[GROUP_KEY[group]] @ w + b
            for ratio in (0.1, 0.2, 0.3):
                k = max(1, int(round(ratio * len(scores))))
                order = np.argsort(scores)[:k]  # lowest predicted damage
                ab = [heads_all[i] for i in order]
                acc = accuracy(ab)
                print(f"  {name} p={ratio} acc={acc:.4f}", flush=True)
                if record(hf_name, name, ratio, "accuracy", acc):
                    have.add((hf_name, name, str(ratio)))
        del model

    # ---------------- GPT-2 perplexity ----------------
    hf_name = "gpt2"
    need = [r for r in (0.1, 0.2, 0.3)
             if (hf_name, "ridge_combined", str(r)) not in have]
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, attn_implementation="eager").eval()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(x["text"] for x in ds)
    ids = tok(text, return_tensors="pt").input_ids[0]
    block, n_blocks = 512, min(len(ids) // 512, 64)
    blocks = ids[:n_blocks * block].reshape(n_blocks, block)
    samples = [" ".join(tok.decode(b[:40])) for b in blocks[:8]]
    heads_all, rows_metrics, attn = [], [], []
    with torch.no_grad():
        for t in samples:
            enc = tok(t, return_tensors="pt", truncation=True,
                       max_length=64)
            out = model.transformer(**enc, output_attentions=True)
            for Li, A in enumerate(out.attentions):
                for Hi in range(A.shape[1]):
                    attn.append(A[0, Hi].cpu().numpy())
                    from attention import (AttentionWaveletDecomposer,
                                            compute_head_metrics)
                    m = compute_head_metrics(A[0, Hi].cpu().numpy(),
                                              AttentionWaveletDecomposer(
                                                  "db4"))
                    m["layer"], m["head"] = Li, Hi
                    rows_metrics.append(m)
            heads_all += [(Li, Hi) for Li in range(len(out.attentions))
                           for Hi in range(A.shape[1])]
    Xh, _ = build_feature_matrix(rows_metrics, {"head_attention": attn},
                                  group="combined")
    fits = {name: fit_lomo("gpt2", group)
             for name, group in ridge_variants}
    Xmats = {GROUP_KEY[group]:
              build_feature_matrix(rows_metrics, {"head_attention": attn},
                                    group=GROUP_KEY[group])[0]
              for _, group in ridge_variants}

    def perplexity(ablate=None):
        ctx = _Null() if ablate is None else SetAblator("gpt2", model, ablate)
        nll = total = 0
        with torch.no_grad(), ctx:
            for blk in blocks:
                logits = model(blk[:-1].unsqueeze(0)).logits[0]
                nll += float(torch.nn.functional.cross_entropy(
                    logits, blk[1:], reduction="sum"))
                total += blk[1:].numel()
        return float(np.exp(nll / total))

    base_ppl = perplexity()
    print(f"[ridge-ds] gpt2 clean={base_ppl:.2f}", flush=True)
    for name, group in ridge_variants:
        w, b, _ = fits[name]
        scores = Xmats[GROUP_KEY[group]] @ w + b
        for ratio in (0.1, 0.2, 0.3):
            if (hf_name, name, str(ratio)) in have:
                continue
            k = max(1, int(round(ratio * len(scores))))
            order = np.argsort(scores)[:k]
            ab = [heads_all[i] for i in order]
            ppl = perplexity(ab)
            print(f"  {name} p={ratio} ppl={ppl:.2f}", flush=True)
            if record(hf_name, name, ratio, "ppl", ppl):
                have.add((hf_name, name, str(ratio)))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=["task", "model", "predictor",
                                            "ratio", "metric", "value"])
        w_.writeheader()
        w_.writerows(rows)
    print(f"saved {OUT} ({len(rows)} rows)", flush=True)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
