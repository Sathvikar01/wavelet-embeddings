"""Loader for contextual vector .npz files saved by ``extractor.py``."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LoadedAnchorContext:
    anchor: str
    vectors: np.ndarray          # (N, D)
    senses: List[str]
    sentences: List[str]
    n_subtokens: List[int]

    @property
    def n(self) -> int:
        return len(self.senses)

    def vectors_for_sense(self, sense: str) -> np.ndarray:
        idxs = [i for i, s in enumerate(self.senses) if s == sense]
        if not idxs:
            return np.empty((0, self.vectors.shape[1]), dtype=np.float32)
        return self.vectors[idxs]

    def mean_per_sense(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for s in sorted(set(self.senses)):
            v = self.vectors_for_sense(s)
            out[s] = v.mean(axis=0) if v.size else np.zeros(self.vectors.shape[1],
                                                             dtype=np.float32)
        return out

    def senses_list(self) -> List[str]:
        return sorted(set(self.senses))


def load_anchor_context(npz_path: str, anchor: str) -> Optional[LoadedAnchorContext]:
    if not os.path.exists(npz_path):
        return None
    d = np.load(npz_path, allow_pickle=True)
    return LoadedAnchorContext(
        anchor=anchor,
        vectors=d["embeddings"].astype(np.float32),
        senses=list(d["senses"]),
        sentences=list(d["sentences"]),
        n_subtokens=list(d["n_subtokens"]),
    )


def load_all_anchors(model_key: str, anchors: List[str],
                     contexts_dir: str) -> Dict[str, LoadedAnchorContext]:
    """Load every (model_key, anchor) contextual .npz from ``contexts_dir``."""
    out: Dict[str, LoadedAnchorContext] = {}
    for a in anchors:
        path = os.path.join(contexts_dir, f"{model_key}_contextual_{a}.npz")
        ctx = load_anchor_context(path, a)
        if ctx is not None and ctx.n > 0:
            out[a] = ctx
    return out
