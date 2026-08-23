"""Lightweight task-loss measures for Phase-4.

Metrics transformed straight from the model:
  1. cosine_drop_orig_vs_ablated - mean sentence-pool embedding distance
  2. kl_div_ablated_to_orig - KL(orig||ablated) averaged per sentence's
     last-token predicted distribution when a lm_head is exposed; falls back
     to per-head prediction's softmax KL otherwise
  3. attention_drift - mean |A_orig - A_ablated|_F
  4. label_drift_tiny - optional pseudo-accuracy drop on a hand-curated
     labelled set

The design goal: provide a fast deterministic "information loss" for
*any* pruning decision. The absolute numbers have units; accuracy-Δ should
be interpreted only RELATIVE TO the original model's value. The experiment
runner accounts for the original (unablated) reference, then compares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #

@dataclass
class SentenceRun:
    sentence: str
    last_hidden: np.ndarray         # (T, D)
    sent_embedding: np.ndarray      # (D,)
    attentions: Tuple[np.ndarray, ...]
    last_token_logits: Optional[np.ndarray]  # (V,) or None


@dataclass
class AblationEffect:
    sentences: List[str]
    cosine_drop: float               # 1 - cos(orig_sent, abl_sent)
    kl_div_next_token: float        # KL(orig || abl) on last-token logits
    attention_drift: float          # ||A_orig - A_ablated||_F averaged
    n_ablated: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _stack_sentence_emb(enc, last_hidden):
    mask = enc.get("attention_mask")
    if mask is None:
        return last_hidden.mean(dim=1).squeeze(0)
    mask = mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return (summed / counts).squeeze(0)


def _last_token_logits(model, model_key: str, last_hidden, enc):
    """If the model family exposes an LM head, project last hidden state."""
    from embeddings.extract import MODEL_REGISTRY
    spec = MODEL_REGISTRY[model_key]
    if not spec.is_decoder_only:
        return None
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        return lm_head(last_hidden).squeeze(0).detach()
    # AutoModel won't have lm_head; tie to the input embeddings table.
    w = model.get_input_embeddings()
    return (last_hidden.detach() @ w.weight.T).squeeze(0)


@torch.no_grad()
def run_model(model, model_key: str, sentences: List[str], tokenizer,
              device: Optional[str] = None,
              head_mask: Optional[torch.Tensor] = None) -> List[SentenceRun]:
    device = device or next(model.parameters()).device
    model.eval()
    out: List[SentenceRun] = []
    for s in sentences:
        enc = tokenizer(s, return_tensors="pt", truncation=True,
                         max_length=64).to(device)
        kw = dict(output_attentions=True)
        if head_mask is not None:
            kw["head_mask"] = head_mask
        with torch.no_grad():
            r = model(**enc, **kw)
        L = r.last_hidden_state             # (1, T, D)
        emb = _stack_sentence_emb(enc, L).cpu().numpy().astype(np.float32)
        attns = tuple(a.squeeze(0).cpu().numpy().astype(np.float32)
                      for a in r.attentions)
        nt_logits = _last_token_logits(model, model_key, L.detach(), enc)
        if nt_logits is not None:
            nt_logits = nt_logits.cpu().numpy().astype(np.float32)
        out.append(SentenceRun(
            sentence=s,
            last_hidden=L.squeeze(0).cpu().numpy().astype(np.float32),
            sent_embedding=emb,
            attentions=attns,
            last_token_logits=nt_logits,
        ))
    return out


# --------------------------------------------------------------------------- #
# Universal head-ablation via forward hooks
# --------------------------------------------------------------------------- #
#
# Modern transformers versions diverge on (or outright drop) the top-level
# ``head_mask`` argument, so the benchmark uses a single architecture-agnostic
# mechanism: a ``forward_pre_hook`` on the *post-attention linear layer*
# (the one that receives the concatenated per-head context as its input),
# zeroing out the input slice belonging to a given head. This produces the
# same behavioural effect as ``attention_probs[:, head, :, :] *= 0`` for
# every HF family implemented here.

def _per_head_o_proj_path(model, model_key: str, layer: int):
    """Return the per-layer post-attention Linear/Conv1D module used to
    zero the head's contribution, or ``None`` when unsupported.
    """
    from embeddings.extract import MODEL_REGISTRY
    family = MODEL_REGISTRY[model_key].family
    mod = model
    try:
        if family in ("bert", "roberta", "deberta"):
            # DeBERTa(v2) mirrors BERT here: the post-attention
            # DebertaV2SelfOutput.dense receives the concatenated
            # per-head context, so zeroing the head's slice works
            # identically (disentangled positional attention only
            # affects Q/K, never V or the output projection).
            return mod.encoder.layer[layer].attention.output.dense
        if family == "distilbert":
            return mod.transformer.layer[layer].attention.out_lin
        if family == "gpt2":
            return mod.h[layer].attn.c_proj
        if family == "llama":
            return mod.layers[layer].self_attn.o_proj
    except Exception:
        return None
    return None


def _head_dim_per_head(model, model_key: str) -> int:
    cfg = model.config
    n_heads = int(getattr(cfg, "num_attention_heads",
                            getattr(cfg, "n_head", 1)))
    hidden = int(getattr(cfg, "hidden_size",
                          getattr(cfg, "n_embd", 4096)))
    return max(1, hidden // n_heads)


class _HeadAblationHooks:
    """Context manager that registers forward_pre_hooks on every relevant
    layer to zero one head's contribution to the post-attention Linear's
    input. Enters/exits leave the model state unchanged.

    Supports single-head or set-of-heads ablation (pass the layer/head
    pairs as ``ablate``).
    """

    def __init__(self, model, model_key: str, ablate: Iterable[Tuple[int, int]]):
        self.model = model
        self.model_key = model_key
        self.ablate = list(ablate)
        self._dim = _head_dim_per_head(model, model_key)
        self._handles: List = []

    def _masked_hook_factory(self, head, dim):
        start = head * dim
        end = start + dim

        def hook(module, inputs):
            # Zero the head's slice of the post-attention context tensor in
            # place. The Linear's first positional arg is (batch, T, all_head).
            for tensor in inputs:
                if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                    tensor[:, :, start:end] = 0.0
            return inputs
        return hook

    def __enter__(self):
        by_layer: Dict[int, List[int]] = {}
        for L, H in self.ablate:
            by_layer.setdefault(L, []).append(int(H))
        for L, heads in by_layer.items():
            mod = _per_head_o_proj_path(self.model, self.model_key, int(L))
            if mod is None:
                continue
            for h in heads:
                self._handles.append(
                    mod.register_forward_pre_hook(
                        self._masked_hook_factory(h, self._dim)
                    )
                )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        return False


@torch.no_grad()
def run_model_ablated(model, model_key: str, sentences: List[str],
                       tokenizer, ablate: Iterable[Tuple[int, int]],
                       device: Optional[str] = None) -> List[SentenceRun]:
    """Forward pass with the heads in ``ablate`` behaviourally zeroed via
    forward hooks (architecture-agnostic; replaces ``head_mask`` for any
    model whose forward path drops ``head_mask`` such as Llama)."""
    device = device or next(model.parameters()).device
    model.eval()
    out: List[SentenceRun] = []
    with _HeadAblationHooks(model, model_key, ablate) as _:
        for s in sentences:
            enc = tokenizer(s, return_tensors="pt", truncation=True,
                             max_length=64).to(device)
            with torch.no_grad():
                r = model(**enc, output_attentions=True)
            L = r.last_hidden_state
            emb = _stack_sentence_emb(enc, L).cpu().numpy().astype(np.float32)
            attns = tuple(a.squeeze(0).cpu().numpy().astype(np.float32)
                           for a in r.attentions)
            nt_logits = _last_token_logits(model, model_key, L.detach(), enc)
            if nt_logits is not None:
                nt_logits = nt_logits.cpu().numpy().astype(np.float32)
            out.append(SentenceRun(
                sentence=s,
                last_hidden=L.squeeze(0).cpu().numpy().astype(np.float32),
                sent_embedding=emb,
                attentions=attns,
                last_token_logits=nt_logits,
            ))
    return out


# --------------------------------------------------------------------------- #
# Effect measurement
# --------------------------------------------------------------------------- #

def _cos_drop(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return 1.0 - float(np.dot(a, b) / (na * nb)) if na and nb else 1.0


def _kl_logits_to_orig(orig_logits: np.ndarray, abl_logits: np.ndarray) -> float:
    p = orig_logits.astype(np.float64)
    q = abl_logits.astype(np.float64)
    p = np.exp(p - p.max()); p /= p.sum()
    q = np.exp(q - q.max()); q /= q.sum()
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def _drift_attn(attn_a: Tuple[np.ndarray, ...],
                 attn_b: Tuple[np.ndarray, ...]) -> float:
    out = []
    for a, b in zip(attn_a, attn_b):
        if a.shape != b.shape:
            return 0.0
        out.append(float(np.linalg.norm(a - b)))
    return float(np.mean(out)) if out else 0.0


def measure_effect(orig_runs: List[SentenceRun],
                    abl_runs: List[SentenceRun],
                    n_ablated: int) -> AblationEffect:
    n = min(len(orig_runs), len(abl_runs))
    if n == 0:
        return AblationEffect([], float("nan"), float("nan"),
                                float("nan"), int(n_ablated))
    cos_drops, kls, drifts = [], [], []
    for o, a in zip(orig_runs[:n], abl_runs[:n]):
        cos_drops.append(_cos_drop(o.sent_embedding, a.sent_embedding))
        drifts.append(_drift_attn(o.attentions, a.attentions))
        if o.last_token_logits is not None \
              and a.last_token_logits is not None \
              and o.last_token_logits.shape == a.last_token_logits.shape:
            kls.append(_kl_logits_to_orig(o.last_token_logits,
                                            a.last_token_logits))
    return AblationEffect(
        sentences=[r.sentence for r in orig_runs[:n]],
        cosine_drop=float(np.mean(cos_drops) if cos_drops else float("nan")),
        kl_div_next_token=float(np.mean(kls) if kls else float("nan")),
        attention_drift=float(np.mean(drifts) if drifts else 0.0),
        n_ablated=int(n_ablated),
    )
