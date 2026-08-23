"""Loader saving the lay-out described in ``extractor.py``.

Reads the per-(model) ``meta.npz`` and ``tokens.npz`` plus the individual
``layerL_headH.npz`` files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HeadMatrix:
    layer: int
    head: int
    raw: np.ndarray         # (T, T)
    normalized: np.ndarray  # (T, T) row-normalised

    @property
    def key(self) -> Tuple[int, int]:
        return (self.layer, self.head)


class AttentionLoader:
    """Read all heads for a specific (model_key, sentence-context) snapshot."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(model_dir)
        meta = np.load(os.path.join(model_dir, "meta.npz"), allow_pickle=True)
        self.sentence: str = str(meta["sentence"])
        self.n_layers: int = int(meta["n_layers"])
        self.n_heads: int = int(meta["n_heads"])
        self.seq_len: int = int(meta["seq_len"])
        tok_path = os.path.join(model_dir, "tokens.npz")
        self.tokens: List[str] = list(np.load(tok_path,
                                                allow_pickle=True)["tokens"])

    # ------------------------------------------------------------------ #
    def load_head(self, layer: int, head: int) -> HeadMatrix:
        fname = os.path.join(self.model_dir, f"layer{layer}_head{head}.npz")
        if not os.path.exists(fname):
            raise FileNotFoundError(fname)
        d = np.load(fname, allow_pickle=True)
        return HeadMatrix(
            layer=int(d["layer"]),
            head=int(d["head"]),
            raw=d["raw"].astype(np.float32),
            normalized=d["normalized"].astype(np.float32),
        )

    def load_all_heads(self) -> Dict[Tuple[int, int], HeadMatrix]:
        out: Dict[Tuple[int, int], HeadMatrix] = {}
        for L in range(self.n_layers):
            for H in range(self.n_heads):
                out[(L, H)] = self.load_head(L, H)
        return out

    def load_layer(self, layer: int) -> List[HeadMatrix]:
        return [self.load_head(layer, H) for H in range(self.n_heads)]


def find_available_models(root_dir: str) -> List[str]:
    """Return the list of model subdirectories present under ``root_dir``."""
    if not os.path.isdir(root_dir):
        return []
    return sorted([d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))])
