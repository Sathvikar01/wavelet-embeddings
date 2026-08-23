"""Load pre-extracted embeddings from disk and downcast tokens to category ids.

The loader is responsible for:
  * reading the .npz files produced by ``extract.py``
  * producing ``IndexedEmbedding`` objects for individual tokens
  * persisting and returning a curated probe token set + frequency stats

Token categories (Noun / Verb / Adjective / Number / Punctuation / Rare /
Frequent / Subword / Special) are identified from surface form of the token
without requiring external POS dictionaries - they're rough heuristics fine
for frequency-spectrum analysis.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Probe tokens used throughout the experiments.
# --------------------------------------------------------------------------- #

DEFAULT_PROBE_TOKENS: List[str] = [
    # Nouns
    "king", "queen", "man", "woman", "apple", "orange",
    "country", "city", "computer", "language",
    # Verbs
    "run", "eat", "think", "create", "is", "are", "walk", "read", "write", "see",
    # Adjectives
    "good", "bad", "tall", "short", "beautiful", "ugly", "happy", "sad",
    "fast", "slow",
    # Numbers
    "1", "2", "10", "100", "thousand", "million",
    # Punctuation
    ".", ",", ";", "!", "?", ":", "-",
    # Special / subword nicks used by BERT/GPT-2 tokenizers
    "[CLS]", "[SEP]", "[PAD]", "[MASK]", "##s", "##ing", "[UNK]",
    "</w>", "Ġthe", "Ġand", "Ġa",
    # Frequent common words
    "the", "and", "of", "to", "a", "in", "that", "is", "for", "it",
]

NEIGHBOR_TOKENS: List[str] = ["king", "queen", "man", "woman", "apple", "orange"]


# --------------------------------------------------------------------------- #
# Category heuristic
# --------------------------------------------------------------------------- #

_BRACKET_RE = re.compile(r"^\[.*\]$")
_NUM_RE = re.compile(r"^\d+([.,]\d+)?$")
_PUNCT_CHARS = set(".,;:!?-\"'()[]{}")


def categorize_token(tok: str) -> str:
    """Heuristically bucket a token into one canonical category."""
    t = tok
    if _BRACKET_RE.match(t):
        return "special"
    if t in {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"}:
        return "special"
    if t.startswith("##"):       # BERT wordpiece continuation
        return "subword"
    if t.startswith("Ġ") or t.startswith("</w>"):
        # GPT-2 BPE space-marker prefixes
        core = t.lstrip("Ġ").rstrip("</w>")
        return categorize_token(core) if core else "subword"
    if _NUM_RE.match(t):
        return "number"
    if t and all(ch in _PUNCT_CHARS for ch in t):
        return "punctuation"
    if len(t) <= 3 and t.lower() in {"is", "are", "am", "be", "do", "did", "has", "had", "was", "were", "run", "see", "eat", "sit", "go", "goes"}:
        return "verb"
    if t.lower() in {"king","queen","man","woman","apple","orange","country","city","computer","language","dog","cat","car","house"}:
        return "noun"
    if t.lower() in {"good","bad","tall","short","beautiful","ugly","happy","sad","fast","slow","red","green","big","small"}:
        return "adjective"
    # Fallback - derive from known prefixes
    if t.lower() in {"run","eat","think","create","walk","read","write","saw","made"}:
        return "verb"
    return "other"


def split_by_frequency(tokens: List[str], embedding_matrix: np.ndarray,
                       rare_quantile: float = 0.1,
                       freq_quantile: float = 0.9,
                       token_freq: Optional[Dict[str, int]] = None) -> Dict[str, List[int]]:
    """Group indices into 'rare', 'frequent', 'middle' by token frequency.

    If ``token_freq`` is not provided we synthesize a fake frequency from the
    embedding's L2 norm (just to have a deterministic ordering). In practice
    the caller should pass real corpus counts.
    """
    if token_freq is None:
        # Use norm as deterministic proxy sorted ascending
        norms = np.linalg.norm(embedding_matrix, axis=1)
        order = np.argsort(norms)
        n = len(order)
        rare_idxs = order[: int(n * rare_quantile)].tolist()
        freq_idxs = order[int(n * freq_quantile):].tolist()
        mid_idxs = order[int(n * rare_quantile): int(n * freq_quantile)].tolist()
        return {"rare": rare_idxs, "middle": mid_idxs, "frequent": freq_idxs}

    # Real frequencies
    ordered = sorted(range(len(tokens)), key=lambda i: -token_freq.get(tokens[i], 0))
    n = len(ordered)
    return {
        "frequent": ordered[: int(n * (1 - freq_quantile))],
        "middle": ordered[int(n * (1 - freq_quantile)): int(n * (1 - rare_quantile))],
        "rare": ordered[int(n * (1 - rare_quantile)):],
    }


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

@dataclass
class IndexedEmbedding:
    token: str
    index: int
    vector: np.ndarray   # (D,)
    category: str


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

class EmbeddingLoader:
    """Load & index saved embedding matrices."""

    def __init__(self, npz_path: str):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path, allow_pickle=True)
        self.embeddings: np.ndarray = data["embeddings"].astype(np.float32)
        self.tokens: List[str] = list(data["tokens"])
        self._index_map: Dict[str, int] = {
            t: i for i, t in enumerate(self.tokens)
        }
        # Pre-compute categories
        self.categories: List[str] = [categorize_token(t) for t in self.tokens]

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def embed_dim(self) -> int:
        return int(self.embeddings.shape[1])

    def index_of(self, token: str) -> int:
        if token in self._index_map:
            return self._index_map[token]
        # Try case-insensitive / stripped variants
        if token.lower() in self._index_map:
            return self._index_map[token.lower()]
        raise KeyError(token)

    def get(self, token: str) -> IndexedEmbedding:
        idx = self.index_of(token)
        return IndexedEmbedding(
            token=token,
            index=idx,
            vector=self.embeddings[idx],
            category=self.categories[idx],
        )

    def get_many(self, tokens: List[str]) -> List[IndexedEmbedding]:
        out = []
        for t in tokens:
            try:
                out.append(self.get(t))
            except KeyError:
                continue
        return out

    def indices_in_category(self, category: str) -> List[int]:
        return [i for i, c in enumerate(self.categories) if c == category]

    def vectors_in_category(self, category: str) -> Tuple[List[int], np.ndarray]:
        idxs = self.indices_in_category(category)
        if not idxs:
            return [], np.empty((0, self.embed_dim), dtype=np.float32)
        return idxs, self.embeddings[idxs]

    def all_vectors(self) -> np.ndarray:
        return self.embeddings
