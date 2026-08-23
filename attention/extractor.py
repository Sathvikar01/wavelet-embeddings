"""Extract per-(layer, head) attention matrices from HF transformers.

For a sentence ``X``:

  * tokenize with ``return_tensors='pt'`` and ``output_attentions=True``
  * run a forward pass and read ``outputs.attentions`` which is a tuple of
    length ``L`` (one entry per layer), each entry of shape
    ``[batch, H, T, T]`` where H = n_heads.

For every layer/head we save:

  * raw attention matrix      : float32 [T, T]
  * row-normalized matrix     : float32 [T, T]  (sums to 1 along last dim)
  * input tokens                : list[str]

Layout::

    results/attention/<model>/<layer>_<head>.npz
    results/attention/<model>/tokens.npz
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from embeddings.extract import MODEL_REGISTRY


# --------------------------------------------------------------------------- #
# Per-model attention-instrumented forward pass
# --------------------------------------------------------------------------- #

@dataclass
class AttentionOutput:
    model_key: str
    sentence: str
    tokens: List[str]
    n_layers: int
    n_heads: int
    seq_len: int
    attention: np.ndarray   # (L, H, T, T) float32, row-normalised


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #

class AttentionExtractor:
    """Run a forward pass and collect per-head attention matrices."""

    def __init__(self, model_key: str, device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Choose from {list(MODEL_REGISTRY)}"
            )
        self.spec = MODEL_REGISTRY[model_key]
        self.model_key = model_key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    def load(self):
        """Lazy-load the model + tokenizer."""
        if self._loaded:
            return self
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.hf_name, cache_dir=self.cache_dir
        )
        # Build with attention outputs active.
        # Llama, GPT-2, BERT and RoBERTa all default to SDPA in
        # transformers 4.42, but SDPA does not expose per-head
        # ``output_attentions=True`` (or ``head_mask``) without
        # falling back to a slow manual attention impl. Pin eager
        # attention for those families so attention extraction is
        # both clean and fast (no per-forward warning, ~2x faster
        # attention path).
        kwargs = {}
        if self.spec.family in ("llama", "gpt2", "bert", "roberta"):
            kwargs["attn_implementation"] = "eager"
        if self.spec.family in ("bert", "distilbert", "roberta", "deberta"):
            self.model = AutoModel.from_pretrained(
                self.spec.hf_name, cache_dir=self.cache_dir,
                output_attentions=True, **kwargs,
            )
        elif self.spec.family in ("gpt2", "llama"):
            self.model = AutoModel.from_pretrained(
                self.spec.hf_name, cache_dir=self.cache_dir,
                output_attentions=True, **kwargs,
            )
        else:
            raise ValueError(f"Unknown family '{self.spec.family}'")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return self

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract(self, sentence: str, max_length: int = 64) -> AttentionOutput:
        """Forward-pass and return raw + normalised attention matrices.

        Note: for GPT-2 the AutoModel wraps GPT2Model which only returns
        attentions from the decoder transformer if we call
        ``output_attentions=True``. The attention tensor is full [B,H,T,T].
        """
        if not self._loaded:
            self.load()
        enc = self.tokenizer(
            sentence, return_tensors="pt", truncation=True,
            max_length=max_length,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, output_attentions=True)
        attns = out.attentions        # tuple of length L, each [1, H, T, T]
        if attns is None or len(attns) == 0:
            raise RuntimeError("Model did not return attentions. Ensure "
                               "output_attentions=True is supported.")
        A = np.stack([a.squeeze(0).cpu().numpy().astype(np.float32)
                      for a in attns])  # (L, H, T, T)
        # Identify tokens (subtokens) and their offset mapping; we store every
        # subtoken since attention is subtoken-level.
        ids = enc["input_ids"].squeeze(0).tolist()
        tokens = [self.tokenizer.decode([i]) for i in ids]
        return AttentionOutput(
            model_key=self.model_key, sentence=sentence,
            tokens=tokens,
            n_layers=A.shape[0],
            n_heads=A.shape[1],
            seq_len=A.shape[2],
            attention=A,
        )

    # ------------------------------------------------------------------ #
    def close(self):
        if self._loaded:
            del self.model
            self.model = None
            self._loaded = False
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_attention_output(
    out: AttentionOutput,
    out_dir: str,
    layer_subdir: bool = True,
) -> Dict[str, str]:
    """Persist every (layer, head) attention matrix to its own .npz.

    Returns the dict of saved paths keyed by ``(layer, head)`` +
    ``"tokens"``/``"meta"``.
    """
    model_dir = os.path.join(out_dir, out.model_key)
    os.makedirs(model_dir, exist_ok=True)

    np.savez(
        os.path.join(model_dir, "meta.npz"),
        sentence=out.sentence,
        n_layers=out.n_layers,
        n_heads=out.n_heads,
        seq_len=out.seq_len,
    )
    np.savez(
        os.path.join(model_dir, "tokens.npz"),
        tokens=np.array(out.tokens, dtype=object),
    )

    paths: Dict[str, str] = {"meta": os.path.join(model_dir, "meta.npz"),
                              "tokens": os.path.join(model_dir, "tokens.npz")}
    for L in range(out.n_layers):
        for H in range(out.n_heads):
            A = out.attention[L, H]  # (T, T)
            # Row-normalised copy (defensive; HF attention is already normalised
            # for BERT/GPT-2 over the key dim, but enforce it).
            row_sums = A.sum(axis=1, keepdims=True)
            norm = np.divide(A, row_sums, out=np.zeros_like(A),
                              where=row_sums > 0)
            fname = os.path.join(model_dir,
                                 f"layer{L}_head{H}.npz")
            np.savez(
                fname,
                raw=A,
                normalized=norm,
                layer=np.int32(L),
                head=np.int32(H),
            )
            paths[(L, H)] = fname
    return paths


# --------------------------------------------------------------------------- #
# Convenience entry point used by the CLI
# --------------------------------------------------------------------------- #

def extract_and_save(
    model_key: str,
    sentence: str,
    out_dir: str,
    max_length: int = 64,
) -> Optional[Dict[str, str]]:
    """Run a forward pass on ``sentence`` and save every head's attention."""
    ext = AttentionExtractor(model_key)
    ext.load()
    out = ext.extract(sentence, max_length=max_length)
    paths = save_attention_output(out, out_dir)
    ext.close()
    return paths
