"""Downstream-task evaluation of predictor-ranked head pruning.

Answers the reviewer question "does representation-level damage predict
TASK-level damage?": prune the bottom-p% of heads ranked by each
predictor, then measure

  * SST-2 validation accuracy  (fine-tuned BERT-base / DistilBERT)
  * WikiText-2 perplexity      (GPT-2)

Head rankings come from each model's own attention maps on sample text
(db4 wavelet metrics + the standard predictor registry), exactly as in
the representation-level pipeline, so the comparison is apples-to-apples.

Usage:
  python experiments/downstream_pruning.py            # everything
  python experiments/downstream_pruning.py --ratios 0.1 0.2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import AttentionWaveletDecomposer, compute_head_metrics  # noqa: E402
from pruning.registry import PREDICTOR_NAMES, compute_predictor         # noqa: E402

PREDICTORS = ["wavelet", "attention_entropy_true", "magnitude",
               "attention_weight", "random"]
DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# Hook-based set ablation (works inside *ForSequenceClassification wrappers)
# --------------------------------------------------------------------------- #

def _base_and_modules(model_key, model):
    if model_key == "distilbert":
        base = model.distilbert
        n_layers = len(base.transformer.layer)
    elif model_key == "bert":
        base = model.bert
        n_layers = len(base.encoder.layer)
    elif model_key == "gpt2":
        base = model.transformer
        n_layers = len(base.h)
    else:
        raise ValueError(model_key)
    return base, n_layers


def _head_dim(cfg):
    nh = cfg.num_attention_heads
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
    return hidden // nh


class SetAblator:
    """Zero one head's slice of every post-attention projection input."""

    def __init__(self, model_key, model, heads):
        self.model_key = model_key
        self.model = model
        self.heads = list(heads)
        self._handles = []
        self._dim = _head_dim(model.config)

    def __enter__(self):
        dim = self._dim
        by_layer = {}
        for (L, H) in self.heads:
            by_layer.setdefault(int(L), []).append(int(H))
        for L, hs in by_layer.items():
            mod = _module_for(self.model_key, self.model, L)
            start_ends = [(h * dim, (h + 1) * dim) for h in hs]

            def hook(module, args, start_ends=start_ends):
                x = args[0]
                for s, e in start_ends:
                    x[..., s:e] = 0.0
                return (x,) + args[1:]

            self._handles.append(mod.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


def _module_for(model_key, model, layer):
    if model_key == "distilbert":
        return model.distilbert.transformer.layer[layer].attention.out_lin
    if model_key == "bert":
        return model.bert.encoder.layer[layer].attention.output.dense
    if model_key == "gpt2":
        return model.transformer.h[layer].attn.c_proj
    raise ValueError(model_key)


def rank_heads(model_key, model, tokenizer, texts, max_length=64):
    """Score every head with every registered predictor."""
    base, n_layers = _base_and_modules(model_key, model)
    attn_decomp = AttentionWaveletDecomposer("db4")
    head_attn, rows = [], []
    model.eval()
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True,
                             max_length=max_length).to(DEVICE)
            out = base(**enc, output_attentions=True)
            for L, A in enumerate(out.attentions):
                for H in range(A.shape[1]):
                    head_attn.append(A[0, H].cpu().numpy())
                    m = compute_head_metrics(
                        A[0, H].cpu().numpy(), attn_decomp)
                    m["layer"], m["head"] = L, H
                    rows.append(m)
    all_heads = [(r["layer"], r["head"]) for r in rows]
    scores = {p: compute_predictor(p, rows,
                                    extra={"head_attention": head_attn},
                                    seed=0)
               for p in PREDICTORS}
    return all_heads, scores


def bottom_fraction(all_heads, scores, frac):
    order = np.argsort(scores)          # ascending = least important first
    k = max(1, int(round(frac * len(scores))))
    return [all_heads[i] for i in order[:k]]


# --------------------------------------------------------------------------- #
# SST-2
# --------------------------------------------------------------------------- #

def eval_sst2(args, rows_out):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="validation")
    texts = [x["sentence"] for x in ds]
    labels = np.array([x["label"] for x in ds])
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(texts), size=8, replace=False)

    for key, hf_name in (("bert", "textattack/bert-base-uncased-SST-2"),
                          ("distilbert",
                           "distilbert-base-uncased-finetuned-sst-2-english")):
        tok = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_name).to(DEVICE).eval()

        def accuracy(ablate=None):
            ctx = _nullcontext() if ablate is None \
                else SetAblator(key, model, ablate)
            correct = 0
            bs = 32
            with torch.no_grad(), ctx:
                for i in range(0, len(texts), bs):
                    enc = tok(texts[i:i + bs], padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(DEVICE)
                    logits = model(**enc).logits
                    correct += int((logits.argmax(-1).cpu().numpy()
                                     == labels[i:i + bs]).sum())
            return correct / len(texts)

        heads, scores = rank_heads(key, model, tok,
                                    [texts[i] for i in sample_idx])
        base_acc = accuracy(None)
        print(f"[sst2] {hf_name}: clean acc={base_acc:.4f}", flush=True)
        rows_out.append(dict(task="sst2", model=hf_name, predictor="none",
                              ratio=0.0, metric="accuracy",
                              value=round(base_acc, 4)))
        for pred in PREDICTORS:
            for ratio in args.ratios:
                ab = bottom_fraction(heads, scores[pred], ratio)
                acc = accuracy(ab)
                print(f"  {pred:<24} p={ratio:.2f} acc={acc:.4f} "
                      f"(drop {base_acc - acc:+.4f})", flush=True)
                rows_out.append(dict(task="sst2", model=hf_name,
                                      predictor=pred, ratio=ratio,
                                      metric="accuracy", value=round(acc, 4)))
        del model


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- #
# GPT-2 WikiText-2 perplexity
# --------------------------------------------------------------------------- #

def eval_gpt2_ppl(args, rows_out):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE).eval()
    model.config.pad_token = model.config.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(x["text"] for x in ds)
    ids = tok(text, return_tensors="pt").input_ids[0]
    block = 512
    n_blocks = min(len(ids) // block, 64)
    blocks = ids[:n_blocks * block].reshape(n_blocks, block)
    print(f"[ppl] gpt2 on wikitext2-test: {n_blocks} x {block} tokens",
          flush=True)

    sample_texts = [" ".join(tok.decode(b[:40])) for b in blocks[:8]]
    heads, scores = rank_heads("gpt2", model, tok, sample_texts,
                                max_length=64)

    def perplexity(ablate=None):
        ctx = _nullcontext() if ablate is None \
            else SetAblator("gpt2", model, ablate)
        nll, total = 0.0, 0
        with torch.no_grad(), ctx:
            for b in blocks:
                inp = b[:-1].unsqueeze(0).to(DEVICE)
                tgt = b[1:].to(DEVICE)
                logits = model(inp).logits[0]
                nll += float(torch.nn.functional.cross_entropy(
                    logits, tgt, reduction="sum"))
                total += tgt.numel()
        return float(np.exp(nll / total))

    base_ppl = perplexity(None)
    print(f"[ppl] clean ppl={base_ppl:.2f}", flush=True)
    rows_out.append(dict(task="wikitext2_ppl", model="gpt2",
                          predictor="none", ratio=0.0, metric="ppl",
                          value=round(base_ppl, 2)))
    for pred in PREDICTORS:
        for ratio in args.ratios:
            ab = bottom_fraction(heads, scores[pred], ratio)
            ppl = perplexity(ab)
            print(f"  {pred:<24} p={ratio:.2f} ppl={ppl:.2f}", flush=True)
            rows_out.append(dict(task="wikitext2_ppl", model="gpt2",
                                  predictor=pred, ratio=ratio,
                                  metric="ppl", value=round(ppl, 2)))


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ratios", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3])
    p.add_argument("--out", default=os.path.join("results", "downstream"))
    args = p.parse_args()

    rows_out = []
    eval_sst2(args, rows_out)
    eval_gpt2_ppl(args, rows_out)

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, "downstream_pruning.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print("saved", out_csv)


if __name__ == "__main__":
    main()
