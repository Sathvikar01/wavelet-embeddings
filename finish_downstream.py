"""Finish the interrupted downstream run.

Parses completed rows from both prior logs, runs ONLY the missing pieces
(DistilBERT 'random' rankings, GPT-2 perplexity suite) and merges
everything into results/downstream/downstream_pruning.csv.
"""

import csv
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.downstream_pruning import (
    PREDICTORS, bottom_fraction, rank_heads,
)

SST_BERT = "textattack/bert-base-uncased-SST-2"
SST_DISTIL = "distilbert-base-uncased-finetuned-sst-2-english"
OUT = "results/downstream/downstream_pruning.csv"


def parse_log():
    rows = []
    model = None
    pats = [
        ("sst2_clean", re.compile(r"\[sst2\] (\S+): clean acc=([\d.]+)")),
        ("ppl_clean", re.compile(r"\[ppl\] clean ppl=([\d.]+)")),
        ("acc", re.compile(r"^\s+(\S+)\s+p=([\d.]+) acc=([\d.]+)")),
        ("ppl", re.compile(r"^\s+(\S+)\s+p=([\d.]+) ppl=([\d.]+)")),
        ("fin_distil", re.compile(
            r"\[finish\] distil random p=([\d.]+) acc=([\d.]+)")),
        ("fin_clean", re.compile(r"\[finish\] gpt2 clean ppl=([\d.]+)")),
        ("fin_ppl", re.compile(r"\[finish\] gpt2 (\S+) p=([\d.]+) ppl=([\d.]+)")),
    ]
    for path in ("results/_downstream2.log", "results/_finish_ds.log",
                  "results/_finish_ds2.log", "results/_finish_ds3.log"):
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8", errors="replace"):
            if "[finish] distil" in line and "random" not in line:
                continue
            kind = None
            for name, pat in pats:
                m = pat.search(line)
                if m:
                    kind, m2 = name, m
                    break
            if kind is None:
                continue
            if kind == "sst2_clean":
                model = m2.group(1)
                rows.append(dict(task="sst2", model=model, predictor="none",
                                  ratio=0.0, metric="accuracy",
                                  value=round(float(m2.group(2)), 4)))
            elif kind == "ppl_clean":
                rows.append(dict(task="wikitext2_ppl", model="gpt2",
                                  predictor="none", ratio=0.0, metric="ppl",
                                  value=round(float(m2.group(1)), 2)))
            elif kind == "acc":
                if not model:
                    continue
                rows.append(dict(task="sst2", model=model,
                                  predictor=m2.group(1),
                                  ratio=float(m2.group(2)),
                                  metric="accuracy",
                                  value=round(float(m2.group(3)), 4)))
            elif kind == "ppl":
                rows.append(dict(task="wikitext2_ppl", model="gpt2",
                                  predictor=m2.group(1),
                                  ratio=float(m2.group(2)),
                                  metric="ppl",
                                  value=round(float(m2.group(3)), 2)))
            elif kind == "fin_distil":
                rows.append(dict(task="sst2", model=SST_DISTIL,
                                  predictor="random",
                                  ratio=float(m2.group(1)),
                                  metric="accuracy",
                                  value=round(float(m2.group(2)), 4)))
            elif kind == "fin_clean":
                rows.append(dict(task="wikitext2_ppl", model="gpt2",
                                  predictor="none", ratio=0.0, metric="ppl",
                                  value=round(float(m2.group(1)), 2)))
            elif kind == "fin_ppl":
                rows.append(dict(task="wikitext2_ppl", model="gpt2",
                                  predictor=m2.group(1),
                                  ratio=float(m2.group(2)),
                                  metric="ppl",
                                  value=round(float(m2.group(3)), 2)))
    return rows


class SetAb:
    def __init__(self, model_key, model, heads):
        self.model_key = model_key
        self.model = model
        self.heads = [(int(L), int(H)) for L, H in heads]
        cfg = model.config
        nh = cfg.num_attention_heads
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd")
        self._dim = hidden // nh
        self._handles = []

    def __enter__(self):
        by_layer = {}
        for L, H in self.heads:
            by_layer.setdefault(L, []).append(H)
        d = self._dim
        for L, hs in by_layer.items():
            mod = self._module(L)
            spans = [(h * d, (h + 1) * d) for h in hs]

            def hook(module, args, spans=spans):
                x = args[0]
                for s, e in spans:
                    x[..., s:e] = 0.0
                return (x,) + args[1:]

            self._handles.append(mod.register_forward_pre_hook(hook))
        return self

    def _module(self, layer):
        if self.model_key == "distilbert":
            return self.model.distilbert.transformer.layer[layer] \
                .attention.out_lin
        if self.model_key == "bert":
            return self.model.bert.encoder.layer[layer] \
                .attention.output.dense
        if self.model_key == "gpt2":
            return self.model.transformer.h[layer].attn.c_proj
        raise ValueError(self.model_key)

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


class NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main() -> None:
    rows = parse_log()
    print(f"parsed {len(rows)} rows from logs", flush=True)

    have = {(r["model"], r["predictor"], float(r["ratio"])) for r in rows}

    # ---------- DistilBERT 'random' ----------
    if any(m == SST_DISTIL and p == "random"
            for m, p, _ in have) is False or \
            sum(1 for m, p, _ in have
                if m == SST_DISTIL and p == "random") < 3:
        from transformers import AutoModelForSequenceClassification, \
            AutoTokenizer
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        texts = [x["sentence"] for x in ds]
        labels = np.array([x["label"] for x in ds])
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(len(texts), size=8, replace=False)
        tok = AutoTokenizer.from_pretrained(SST_DISTIL)
        model = AutoModelForSequenceClassification.from_pretrained(
            SST_DISTIL, attn_implementation="eager").eval()
        heads, scores = rank_heads("distilbert", model, tok,
                                    [texts[i] for i in sample_idx])

        def accuracy(ablate=None):
            ctx = NullCtx() if ablate is None else SetAb("distilbert",
                                                          model, ablate)
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

        for ratio in (0.1, 0.2, 0.3):
            if (SST_DISTIL, "random", float(ratio)) in have:
                continue
            ab = bottom_fraction(heads, scores["random"], ratio)
            acc = accuracy(ab)
            print(f"[finish] distil random p={ratio} acc={acc:.4f}",
                  flush=True)
            rows.append(dict(task="sst2", model=SST_DISTIL,
                              predictor="random", ratio=ratio,
                              metric="accuracy", value=round(acc, 4)))
        del model

    # ---------- GPT-2 perplexity suite ----------
    need_ppl = [p for p in PREDICTORS
                 if sum(1 for m, pp, _ in have
                         if m == "gpt2" and pp == p) < 3]
    if ("gpt2", "none", 0.0) not in have:
        need_ppl = PREDICTORS
    if need_ppl:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", attn_implementation="eager").eval()
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(x["text"] for x in ds)
        ids = tok(text, return_tensors="pt").input_ids[0]
        block, n_blocks = 512, min(len(ids) // 512, 64)
        blocks = ids[:n_blocks * block].reshape(n_blocks, block)
        sample_texts = [" ".join(tok.decode(b[:40])) for b in blocks[:8]]
        heads, scores = rank_heads("gpt2", model, tok, sample_texts)

        def perplexity(ablate=None):
            ctx = NullCtx() if ablate is None else SetAb("gpt2", model,
                                                          ablate)
            nll = total = 0
            with torch.no_grad(), ctx:
                for b in blocks:
                    inp = b[:-1].unsqueeze(0)
                    tgt = b[1:]
                    logits = model(inp).logits[0]
                    nll += float(torch.nn.functional.cross_entropy(
                        logits, tgt, reduction="sum"))
                    total += tgt.numel()
            return float(np.exp(nll / total))

        if ("gpt2", "none", 0.0) not in have:
            bp = perplexity(None)
            print(f"[finish] gpt2 clean ppl={bp:.2f}", flush=True)
            rows.append(dict(task="wikitext2_ppl", model="gpt2",
                              predictor="none", ratio=0.0, metric="ppl",
                              value=round(bp, 2)))
        for pr in need_ppl:
            for ratio in (0.1, 0.2, 0.3):
                if ("gpt2", pr, float(ratio)) in have:
                    continue
                ab = bottom_fraction(heads, scores[pr], ratio)
                ppl = perplexity(ab)
                print(f"[finish] gpt2 {pr} p={ratio} ppl={ppl:.2f}",
                      flush=True)
                rows.append(dict(task="wikitext2_ppl", model="gpt2",
                                  predictor=pr, ratio=ratio,
                                  metric="ppl", value=round(ppl, 2)))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "model", "predictor",
                                           "ratio", "metric", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"merged {len(rows)} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
