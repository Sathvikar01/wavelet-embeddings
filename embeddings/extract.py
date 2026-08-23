"""Extract token embeddings from transformer models.

Supported models:
  - bert-base-uncased  (BERT-base)
  - distilbert-base-uncased (DistilBERT)
  - gpt2               (GPT-2 Small)

The extractor pulls the static input (word/token) embedding matrix directly
from the model's embedding layer. For BERT-family models this is
``embeddings.word_embeddings.weight``; for GPT-2 it is
``wte.weight`` (the token embedding table).

Optionally, context embeddings can be extracted by running a forward pass
with a token batch and reading the last hidden state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #

@dataclass
class ModelSpec:
    hf_name: str
    family: str          # "bert" | "distilbert" | "gpt2" | "roberta" |
                         # "deberta" | "llama"
    label: str           # short human-readable label
    embed_attr: str      # attribute path to the static embedding weight tensor
    has_lm_head: bool = False   # whether a causal LM head is exposed
    is_decoder_only: bool = False


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "bert-base": ModelSpec(
        hf_name="bert-base-uncased",
        family="bert",
        label="BERT-base",
        embed_attr="embeddings.word_embeddings.weight",
    ),
    "distilbert": ModelSpec(
        hf_name="distilbert-base-uncased",
        family="distilbert",
        label="DistilBERT",
        embed_attr="embeddings.word_embeddings.weight",
    ),
    "gpt2": ModelSpec(
        hf_name="gpt2",
        family="gpt2",
        label="GPT-2 Small",
        embed_attr="wte.weight",
        has_lm_head=True,
        is_decoder_only=True,
    ),
    "roberta-base": ModelSpec(
        hf_name="roberta-base",
        family="roberta",
        label="RoBERTa-base",
        embed_attr="embeddings.word_embeddings.weight",
    ),
    "deberta-base": ModelSpec(
        hf_name="microsoft/deberta-base",
        family="deberta",
        label="DeBERTa-base",
        embed_attr="embeddings.word_embeddings.weight",
    ),
    "tinyllama": ModelSpec(
        hf_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        family="llama",
        label="TinyLlama-1.1B",
        embed_attr="embed_tokens.weight",
        has_lm_head=True,
        is_decoder_only=True,
    ),
}


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #

class EmbeddingExtractor:
    """Load a HF model and expose its static token-embedding matrix."""

    def __init__(self, model_key: str, device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_key}'. "
                f"Choose from {list(MODEL_REGISTRY)}"
            )
        self.spec = MODEL_REGISTRY[model_key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ----- loading ---------------------------------------------------------- #

    def load(self):
        """Lazy-load the model + tokenizer."""
        if self._loaded:
            return self
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.spec.hf_name, cache_dir=self.cache_dir
        )
        kwargs = {}
        # Llama, GPT-2, BERT and RoBERTa all default to SDPA in
        # transformers 4.42; SDPA does not expose per-head
        # ``output_attentions=True`` without falling back to a slow
        # manual attention impl (which also prints a warning per
        # forward). Load eager attention upfront for those families
        # so attention extraction is both clean and fast.
        if self.spec.family in ("llama", "gpt2", "bert", "roberta"):
            kwargs["attn_implementation"] = "eager"
        self.model = AutoModel.from_pretrained(
            self.spec.hf_name, cache_dir=self.cache_dir, **kwargs
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return self

    # ----- embeddings ------------------------------------------------------- #

    def _get_weight(self) -> torch.Tensor:
        """Fetch the static token-embedding weight tensor."""
        if not self._loaded:
            self.load()
        obj = self.model
        for attr in self.spec.embed_attr.split("."):
            obj = getattr(obj, attr)
        return obj.detach()

    def static_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Return (embeddings [V, D], tokens [V]).

        The shape of the returned matrix is (vocab_size, embed_dim).
        """
        w = self._get_weight().cpu().numpy().astype(np.float32)
        tokens = self._vocab_tokens()
        # Some vocab sizes differ slightly (e.g. extra pad tokens); trim if needed.
        if len(tokens) != w.shape[0]:
            tokens = tokens[: w.shape[0]]
        return w, tokens

    def _vocab_tokens(self) -> List[str]:
        toks: List[str] = []
        vocab = self.tokenizer.get_vocab()
        inv = {v: k for k, v in vocab.items()}
        for i in range(len(inv)):
            toks.append(inv[i])
        return toks

    # ----- contextual embeddings ------------------------------------------- #

    @torch.no_grad()
    def contextual_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Run a forward pass and return the last hidden states (averaged per sentence).

        Returns an array of shape (len(sentences), hidden_dim).
        """
        if not self._loaded:
            self.load()
        outs = []
        for s in sentences:
            enc = self.tokenizer(
                s, return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            with torch.no_grad():
                last = self.model(**enc).last_hidden_state  # (1, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            outs.append((summed / counts).squeeze(0).cpu().numpy())
        return np.stack(outs).astype(np.float32)

    # ----- util -------------------------------------------------------------- #

    def embed_dim(self) -> int:
        return int(self._get_weight().shape[1])

    def vocab_size(self) -> int:
        return int(self._get_weight().shape[0])

    def close(self):
        if self._loaded:
            del self.model
            self.model = None
            self._loaded = False
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


# --------------------------------------------------------------------------- #
# Convenience entry points used by main.py
# --------------------------------------------------------------------------- #

def extract_and_save(model_key: str, out_dir: str) -> str:
    """Extract static embeddings for a model and save to disk (.npz)."""
    ext = EmbeddingExtractor(model_key)
    ext.load()
    emb, toks = ext.static_embeddings()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_key}_embeddings.npz")
    np.savez(out_path, embeddings=emb, tokens=np.array(toks, dtype=object))
    ext.close()
    return out_path
