"""Downstream validation of embedding-table compression.

Replaces a model's word-embedding table with its lossy reconstruction
(wavelet sparse + int8, or PCA truncation at matched effective bytes)
and measures what happens to the TASK metric:

  * SST-2 validation accuracy  (fine-tuned BERT-base / DistilBERT)
  * WikiText-2 perplexity      (GPT-2)

This closes the loop opened by the pruning analysis: if representation
similarity is not sufficient for task preservation under head ablation,
the same question must be asked before recommending an embedding
compression recipe.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.dim_reduction import (                       # noqa: E402
    _wavelet_sparse_recon, _sparse_raw_recon, _dct_sparse_recon,
)
from wavelets.base import WaveletDecomposer                   # noqa: E402

DEVICE = "cpu"
MODELS = [
    ("sst2", "textattack/bert-base-uncased-SST-2", "bert"),
    ("sst2", "distilbert-base-uncased-finetuned-sst-2-english",
     "distilbert"),
    ("wikitext2_ppl", "gpt2", "gpt2"),
]


def get_weight(model, kind):
    if kind == "bert":
        return model.bert.embeddings.word_embeddings.weight
    if kind == "distilbert":
        return model.distilbert.embeddings.word_embeddings.weight
    return model.transformer.wte.weight


def set_weight(model, kind, tensor):
    w = get_weight(model, kind)
    with torch.no_grad():
        w.data = tensor.to(w.device, w.dtype)


def compress_table(W64, method, param):
    """W64: (V, D) float64. Returns reconstructed copy."""
    V, D = W64.shape
    out = np.empty_like(W64)
    if method == "pca":
        rng = np.random.default_rng(0)
        fit = W64[rng.choice(V, size=min(5000, V), replace=False)]
        mean = fit.mean(axis=0, keepdims=True)
        _, S, Vt = np.linalg.svd(fit - mean, full_matrices=False)
        comps = Vt[:param]
        # centre with global mean for reconstruction fidelity
        gmean = W64.mean(axis=0, keepdims=True)
        out = (W64 - gmean) @ comps.T @ comps + gmean
        return out.astype(np.float64)
    if method == "topk_int8":
        for i in range(V):
            out[i] = _sparse_raw_recon(W64[i], nnz=param, quantise=True)
            if i % 10000 == 0:
                print(f"    row {i}/{V}", flush=True)
        return out
    if method == "dct_int8":
        from scipy.fft import dct, idct

        def _one_dct(x):
            c = dct(x, norm="ortho")
            k = min(param, len(c))
            idx = np.argpartition(np.abs(c), len(c) - k)[len(c) - k:]
            mask = np.zeros_like(c, dtype=bool)
            mask[idx] = True
            vals = c[idx]
            s = float(np.max(np.abs(vals))) / 127.0
            q = np.clip(np.round(vals / max(s, 1e-30)), -127, 127)
            rec = np.zeros_like(c)
            rec[idx] = q * s
            return idct(rec, norm="ortho")

        for i in range(V):
            out[i] = _one_dct(W64[i])
            if i % 10000 == 0:
                print(f"    row {i}/{V}", flush=True)
        return out
    dec = WaveletDecomposer("db4")
    for i in range(V):
        dcp = dec.decompose(W64[i])
        out[i] = _wavelet_sparse_recon(dec, dcp, nnz=param, quantise=True)
        if i % 10000 == 0:
            print(f"    row {i}/{V}", flush=True)
    return out


def eval_sst2(model, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    texts = [x["sentence"] for x in ds]
    labels = np.array([x["label"] for x in ds])
    correct = 0
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], padding=True, truncation=True,
                             max_length=128, return_tensors="pt")
            logits = model(**enc).logits
            correct += int((logits.argmax(-1).cpu().numpy()
                             == labels[i:i + 32]).sum())
    return correct / len(texts)


def eval_ppl(model, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(x["text"] for x in ds)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    block = 512
    n_blocks = min(len(ids) // block, 64)
    blocks = ids[:n_blocks * block].reshape(n_blocks, block)
    nll = total = 0
    with torch.no_grad():
        for b in blocks:
            logits = model(b[:-1].unsqueeze(0)).logits[0]
            nll += float(torch.nn.functional.cross_entropy(
                logits, b[1:], reduction="sum"))
            total += b[1:].numel()
    return float(np.exp(nll / total))


CONFIGS = [
    ("wav_int8", 170),   # ~512 effective bytes/token (~17% of original)
    ("wav_int8", 341),   # ~1024 effective bytes/token (~33%)
    ("dct_int8", 170),
    ("dct_int8", 341),
    ("topk_int8", 170),
    ("topk_int8", 341),
    ("pca", 128),
    ("pca", 256),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("results", "downstream"))
    args = p.parse_args()

    rows = []
    out_csv = os.path.join(args.out, "downstream_compression.csv")
    if os.path.isfile(out_csv):
        with open(out_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"resuming: {len(rows)} rows already recorded", flush=True)
    have = {(r["model"], r["method"], str(r["param"])) for r in rows}
    for task, hf_name, kind in MODELS:
        if hf_name == "textattack/bert-base-uncased-SST-2":
            from transformers import AutoModelForSequenceClassification, \
                AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_name, attn_implementation="eager").eval()
            score = lambda: eval_sst2(model, tok)          # noqa: E731
            metric = "accuracy"
        elif hf_name.endswith("sst-2-english"):
            from transformers import AutoModelForSequenceClassification, \
                AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                hf_name, attn_implementation="eager").eval()
            score = lambda: eval_sst2(model, tok)          # noqa: E731
            metric = "accuracy"
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(hf_name)
            model = AutoModelForCausalLM.from_pretrained(
                hf_name, attn_implementation="eager").eval()
            score = lambda: eval_ppl(model, tok)           # noqa: E731
            metric = "ppl"

        W = get_weight(model, kind).detach().cpu().numpy().astype(
            np.float64)
        base = score()
        print(f"[{hf_name}] clean {metric}={base:.4f}", flush=True)
        if (hf_name, "none", "") not in have:
            rows.append(dict(model=hf_name, task=task, method="none",
                              param="", metric=metric,
                              value=round(base, 4)))
        keep_base = {task: base}
        for method, param in CONFIGS:
            if (hf_name, method, str(param)) in have:
                continue
            print(f"  compressing: {method} param={param}", flush=True)
            Wrec = compress_table(W, method, param)
            set_weight(model, kind, torch.tensor(Wrec))
            v = score()
            if metric == "accuracy":
                delta = (keep_base[task] - v) * 100.0
                print(f"    {metric}={v:.4f} (drop {delta:+.1f} pts)",
                      flush=True)
            else:
                print(f"    {metric}={v:.2f} "
                      f"(x{v / keep_base[task]:.2f})", flush=True)
            rows.append(dict(model=hf_name, task=task, method=method,
                              param=f"{param}", metric=metric,
                              value=round(v, 4)))
            set_weight(model, kind, torch.tensor(W))  # restore
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w_ = csv.DictWriter(f, fieldnames=["model", "task",
                                                    "method", "param",
                                                    "metric", "value"])
                w_.writeheader()
                w_.writerows(rows)
        del model

    os.makedirs(args.out, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "task", "method",
                                           "param", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print("saved", out_csv, flush=True)


if __name__ == "__main__":
    main()
