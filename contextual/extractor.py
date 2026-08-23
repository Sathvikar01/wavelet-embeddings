"""Per-token contextual embedding extractor.

We run a forward pass and pull **the hidden state for the anchor's BPE/WordPiece
sub-token span** rather than the pooled sentence vector used in Phase 1.  We
locate the anchor by character offsets in the raw sentence, then map to the
tokenizer's offset mapping which HF tokenizers expose via
``return_offsets_mapping=True``.  We average the hidden states of any sub-token
whose span overlaps with the anchor occurrence (useful for BPE-based GPT-2
where the word may be split).

For BERT-family models we tokenize without special token boundaries since
``[CLS]`` / ``[SEP]`` are auto-added; the offset mapping lines up with
the original sentence characters.  For GPT-2 we disable the leading Ġ-encoding
quirk by tokenizing with ``add_special_tokens=False`` first to compute offset
mapping, then re-tokenize with special tokens for the forward pass and align
indices accordingly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from embeddings.extract import EmbeddingExtractor, MODEL_REGISTRY
from contextual.data import ContextExample, surface_anchor_indices


# --------------------------------------------------------------------------- #
# Output container
# --------------------------------------------------------------------------- #

@dataclass
class AnchorContextVector:
    anchor: str
    sense: str
    sentence: str
    vector: np.ndarray   # (D,)
    n_subtokens: int


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #

class ContextualExtractor(EmbeddingExtractor):
    """Extends :class:`EmbeddingExtractor` with per-token hidden-state lookup."""

    # ----- helpers ---------------------------------------------------------- #

    @staticmethod
    def _as_list(x):
        """Coerce a (1, ...) tensor or a [batch, seq, ...] nested list into a
        flat list at the *sequence* level (i.e. length = max_seq_len).

        For ``offset_mapping`` HF returns:
          - return_tensors='pt' -> Tensor (1, T)       where each T is a 2-list
          - no return_tensors    -> list[(s,e)]         already flat
        So we only strip leading dim when x is a Tensor or when x has shape
        ``[N]`` with each ``x[i]`` itself being a *list* of length 2 (the
        outer dim = batch size = 1).
        """
        if hasattr(x, "squeeze"):
            x = x.squeeze(0)
            return x.tolist() if hasattr(x, "tolist") else list(x)
        if isinstance(x, (list, tuple)):
            x = list(x)
            # Single-batch nested: x = [[(s,e), ...]]  -> strip
            if (len(x) == 1 and isinstance(x[0], (list, tuple))
                    and x[0] and isinstance(x[0][0], (list, tuple))):
                return list(x[0])
            return x
        return list(x) if not hasattr(x, "__iter__") else list(x)

    def _offset_mapping(self, sentence: str, add_special: bool = True
                        ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """Tokenize and return (token_ids, char-offsets spanning ``sentence``)."""
        enc = self.tokenizer(
            sentence, return_tensors="pt", truncation=True,
            max_length=128, return_offsets_mapping=True,
        )
        offsets = self._as_list(enc.pop("offset_mapping"))
        token_ids = self._as_list(enc["input_ids"])
        if not add_special:
            # Identify and strip the special tokens + their offsets
            special_ids = set(self.tokenizer.all_special_ids)
            keep = [(tid, off) for tid, off in zip(token_ids, offsets)
                    if tid not in special_ids]
            if keep:
                token_ids, offsets = map(list, zip(*keep))
            else:
                token_ids, offsets = [], []
        return list(token_ids), list(offsets)

    # ----- locate the anchor span in the token stream ---------------------- #

    def _anchor_subtoken_indices(
        self,
        offsets: List[Tuple[int, int]],
        anchor_char: int,
    ) -> List[int]:
        anchor_end = anchor_char
        out: List[int] = []
        for i, (s, e) in enumerate(offsets):
            if e is None or s is None or e <= s:
                continue
            # Sub-token overlaps with anchor occurrence if its char span intersects
            # the start of the anchor (we treat the anchor as the [anchor_char,
            # anchor_char + len(anchor)) region given by the caller; relax to a
            # 0-length point so any sub-token containing pos is matched).
            if s <= anchor_char < e:
                out.append(i)
                # Continue scanning forward to grab multi-sub-token spans.
                continue
            # If we already started and the next sub-token continues the word,
            # include it as long as it starts <= anchor char OR begins with the
            # continuation markers (##  / Ġ).
            if out and s <= anchor_char + 32 and (e > s):
                # Heuristic: keep only sub-tokens immediately following the start
                # if they are continuations (start at the end of previous subtoken)
                if s == offsets[out[-1]][1]:
                    out.append(i)
        return out

    # ----- public API ------------------------------------------------------- #

    @torch.no_grad()
    def extract_anchor_vector(
        self,
        sentence: str,
        anchor: str,
        occurrence: int = 0,
        layer: Optional[int] = None,
    ) -> Optional[AnchorContextVector]:
        """Run a forward pass on ``sentence`` and return the hidden state for
        the ``occurrence``-th occurrence of ``anchor``.

        ``layer`` selects a hidden layer (None = last).  Returns None if the
        anchor cannot be located in the token stream.
        """
        if not self._loaded:
            self.load()
        positions = surface_anchor_indices(sentence, anchor)
        if occurrence >= len(positions):
            return None
        anchor_char = positions[occurrence]

        # Use the offset mapping to find subtoken indices (strip specials first
        # for stable character offsets).
        _, offsets = self._offset_mapping(sentence, add_special=False)
        # Now get hidden states from the full-input forward pass.
        enc = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=128,
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        if layer is None:
            hidden = out.last_hidden_state.squeeze(0)         # (T, D)
        else:
            hidden = out.hidden_states[layer].squeeze(0)      # (T, D)
        # offsets_without_special aligned with full stream minus specials:
        # to align subtoken indices with the full hidden tensor, recompute the
        # mapping between full input_ids and stripped ones.
        full_ids = enc["input_ids"].squeeze(0).tolist() if hasattr(enc["input_ids"], "squeeze") \
                   else list(enc["input_ids"][0])
        special_ids = set(self.tokenizer.all_special_ids)
        full_offsets = self._as_list(self.tokenizer(
            sentence, return_offsets_mapping=True, truncation=True, max_length=128,
        )["offset_mapping"])
        # Indices of non-special sub-tokens in the full stream:
        nonspecial_full_idx = [
            i for i, tid in enumerate(full_ids) if tid not in special_ids
        ]
        # Now extract the subtoken span covering the anchor, scanning by
        # character offsets in the full stream.
        sub_idx: List[int] = []
        # Find the end of the inflected word containing the anchor
        end_of_word = anchor_char + len(anchor)
        while (end_of_word < len(sentence)
               and sentence[end_of_word].isalpha()):
            end_of_word += 1
        for k, off in enumerate(full_offsets):
            s, e = off
            if s is None or e is None or e <= s:
                continue
            if s <= anchor_char < e:
                sub_idx.append(k)
                # Expand forward while subtokens continue over the surface word.
                end_char = end_of_word
                j = k + 1
                while j < len(full_offsets):
                    s2, e2 = full_offsets[j]
                    if s2 is None or e2 is None or e2 <= s2:
                        j += 1
                        continue
                    if s2 < end_char or s2 == full_offsets[j - 1][1]:
                        sub_idx.append(j)
                        j += 1
                    else:
                        break
                break
        if not sub_idx:
            # Fallback: just pick the subtoken whose char span starts at anchor_char
            for k, (s, e) in enumerate(full_offsets):
                if s == anchor_char:
                    sub_idx = [k]
                    break
        if not sub_idx:
            return None
        vec = hidden[sub_idx].mean(dim=0).cpu().numpy().astype(np.float32)
        return AnchorContextVector(
            anchor=anchor, sense="", sentence=sentence,
            vector=vec, n_subtokens=len(sub_idx),
        )

    # ----- batch ------------------------------------------------------------ #

    def extract_dataset(
        self,
        examples: List[ContextExample],
        layer: Optional[int] = None,
    ) -> List[AnchorContextVector]:
        """Extract anchor vectors for every example in a list."""
        out: List[AnchorContextVector] = []
        for ex in examples:
            v = self.extract_anchor_vector(ex.sentence, ex.anchor,
                                           occurrence=0, layer=layer)
            if v is None:
                continue
            v.sense = ex.sense
            out.append(v)
        return out


# --------------------------------------------------------------------------- #
# Save / load helpers (mirrors extract_and_save)
# --------------------------------------------------------------------------- #

def save_contextual_vectors(
    model_key: str,
    anchor: str,
    vectors: List[AnchorContextVector],
    out_dir: str,
) -> str:
    """Save the contextual vectors for one (model, anchor) pair."""
    os.makedirs(out_dir, exist_ok=True)
    arr = np.stack([v.vector for v in vectors]) if vectors else np.zeros((0, 1))
    meta = [(v.anchor, v.sense, v.sentence, v.n_subtokens) for v in vectors]
    out_path = os.path.join(out_dir, f"{model_key}_contextual_{anchor}.npz")
    np.savez(
        out_path,
        embeddings=arr,
        meta=np.array(meta, dtype=object),
        anchors=np.array([v.anchor for v in vectors], dtype=object),
        senses=np.array([v.sense for v in vectors], dtype=object),
        sentences=np.array([v.sentence for v in vectors], dtype=object),
        n_subtokens=np.array([v.n_subtokens for v in vectors], dtype=np.int32),
    )
    return out_path


def extract_and_save_anchor(
    model_key: str,
    anchor: str,
    examples: List[ContextExample],
    out_dir: str,
    layer: Optional[int] = None,
) -> Optional[str]:
    """End-to-end: pull vectors and save to .npz for one anchor + model."""
    ext = ContextualExtractor(model_key)
    ext.load()
    vecs = ext.extract_dataset(examples, layer=layer)
    if not vecs:
        ext.close()
        return None
    out = save_contextual_vectors(model_key, anchor, vecs, out_dir)
    ext.close()
    return out
