"""Head-ablation engine for Phase-4 using HF's native ``head_mask`` argument.

All HuggingFace encoder/decoder transformer models (BERT, DistilBERT, GPT-2)
support a ``head_mask`` argument to ``forward`` that **zeros out** a
given head's *value-output* contribution. ``head_mask`` is a tensor of
shape ``(num_layers, num_heads)`` where ``1`` keeps a head and ``0`` ablates
it.

This is the cleanest behavioural ablation supported by HF directly; no
monkey-patching of attention modules, no forward-hook footguns.

Usage:

    ablator = HeadAblator(model, 'distilbert', ablate={(1,0), (3,4)})
    head_mask = ablator.head_mask()
    out = model(**enc, head_mask=head_mask, output_attentions=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set, Tuple

import torch
import torch.nn as nn

from embeddings.extract import MODEL_REGISTRY


HeadIndex = Tuple[int, int]


class HeadAblator:
    """Context manager / builder for HF ``head_mask`` ablation."""

    def __init__(self, model: nn.Module, model_key: str,
                 ablate: Iterable[HeadIndex] = (),
                 mode: str = "zero"):
        # Note: HF only supports 'zero' ablation (head_mask = 0 means the head
        # contributes 0 to the value-pathway). Reject 'identity' because that
        # isn't a head_mask primitive.
        if mode not in ("zero", "identity"):
            raise ValueError("mode must be 'zero' or 'identity'")
        if mode == "identity":
            raise NotImplementedError(
                "Use mode='zero'; HF head_mask only supports zero-ablation.")
        self.model = model
        self.model_key = model_key
        self.spec = MODEL_REGISTRY[model_key]
        self.ablate: Set[HeadIndex] = set(ablate)
        self.mode = mode
        self._n_layers, self._n_heads = self._discover_dims()

    # ------------------------------------------------------------------ #
    def _discover_dims(self):
        cfg = self.model.config
        n_heads = int(getattr(cfg, "num_attention_heads",
                                 getattr(cfg, "n_head", 1)))
        # Layers: encoder.layer for BERT, transformer.layer for DistilBERT,
        # h for GPT-2.
        spec = self.spec
        if spec.family in ("bert", "roberta", "deberta"):
            n_layers = len(self.model.encoder.layer)
        elif spec.family == "distilbert":
            n_layers = len(self.model.transformer.layer)
        elif spec.family == "gpt2":
            n_layers = len(self.model.h)
        elif spec.family == "llama":
            n_layers = len(self.model.layers)
        else:
            raise ValueError(f"Unsupported family '{spec.family}'")
        return n_layers, n_heads

    # ------------------------------------------------------------------ #
    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def n_heads(self) -> int:
        return self._n_heads

    # ------------------------------------------------------------------ #
    def head_mask(self, *, as_tensor: bool = True,
                  device: Optional[str] = None):
        """Return a tensor of shape ``(n_layers, n_heads)`` filled with 1.0
        except for the ablated (layer, head) entries which become 0.0."""
        mask = torch.ones((self.n_layers, self.n_heads),
                            dtype=torch.float32,
                            device=device if device is not None
                            else next(self.model.parameters()).device)
        for L, H in self.ablate:
            if 0 <= L < self.n_layers and 0 <= H < self.n_heads:
                mask[L, H] = 0.0
        return mask

    # ------------------------------------------------------------------ #
    def __enter__(self):
        # Nothing to do; callers pass ``head_mask=self.head_mask()`` into
        # the forward call explicitly.
        return self

    def __exit__(self, exc_type, exc, tb):
        return False
