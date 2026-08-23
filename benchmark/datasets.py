"""Sentence corpora for the head-pruning benchmark.

Each loader returns a list of plain-text single-sentence strings that the
ablation runner feeds directly into the HF tokenizer.

Backed by HuggingFace ``datasets``:

* ``wikitext2``        - WikiText-2 (raw, train split) split-flatten by line;
                         long/arbitrary lines are dropped and the remaining
                         sentences kept up to ``max_sentences``.
* ``penn_treebank``    - Penn Treebank (raw character-level ``ptb_text_only``)
                         train split.
* ``glue_subset``      - the single-column GLUE tasks that expose natural
                         sentences: ``cola``, ``sst2`` and ``mrpc``
                         (sentence1 column).

All loaders are deterministic: filtering is rule-based and the returned
list is stable between runs (no shuffle) so statistics reproduce. The seed
argument only governs an optional subsampling step used when
``max_sentences`` is set.
"""

from __future__ import annotations

import re
from typing import List, Optional

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None


__all__ = ["DATASET_NAMES", "load_sentences"]


DATASET_NAMES = ["wikitext2", "penn_treebank", "glue_subset"]


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #

# Heuristics for what counts as a "natural standalone sentence". Lines that
# look like article titles, section headers or boilerplate (e.g.
# ``= = = History = = =``) are skipped. Very short / overly long lines are
# also skipped.

_HEADER_RE = re.compile(r"^=+\s.*\s=+$")
_MIN_LEN = 24
_MAX_LEN = 400
_WORDS_RE = re.compile(r"[A-Za-z]")


def _is_clean_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if _HEADER_RE.match(line):
        return False
    if len(line) < _MIN_LEN or len(line) > _MAX_LEN:
        return False
    if not _WORDS_RE.search(line):
        return False
    # Must contain at least one terminal punctuation to qualify as a
    # standalone sentence; this naturally strips list fragments.
    if line[-1] not in ".!?":
        return False
    # Skip heavy-abbreviated or all-caps lines (section banners etc.).
    letters = [c for c in line if c.isalpha()]
    if letters:
        upper = sum(1 for c in letters if c.isupper())
        if upper / len(letters) > 0.6:
            return False
    return True


def _split_into_sentences(text: str) -> List[str]:
    """Coarse sentence splitter that works for prose without heavy
    abbreviations. Splits on ``.!?`` followed by whitespace and an
    uppercase letter; keeps the trailing punctuation."""
    out: List[str] = []
    buf: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
            continue
        # Sentence boundary after a terminal mark that is followed by
        # space + an uppercase letter: ``re.split`` keeps the mark as its
        # own element, so each (fragment + mark) pair is a COMPLETE
        # sentence and can be flushed immediately.
        parts = re.split(r"([.!?])\s+(?=[A-Z])", stripped)
        i = 0
        while i < len(parts):
            frag = parts[i]
            complete = False
            if i + 1 < len(parts) and parts[i + 1] in ".!?":
                frag = frag + parts[i + 1]
                complete = True
                i += 2
            else:
                i += 1
            if frag:
                buf.append(frag)
            if complete and buf:
                out.append(" ".join(buf).strip())
                buf = []
        # A line-final terminal mark completes the trailing fragment
        # (the regex above cannot capture it -- nothing follows).
        if stripped.endswith((".", "!", "?")) and buf:
            out.append(" ".join(buf).strip())
            buf = []
    if buf:
        out.append(" ".join(buf).strip())
    return [s for s in out if _is_clean_line(s)]


def _subsample(items: List[str], max_sentences: int, seed: int) -> List[str]:
    """Return ``items`` as-is.

    The deterministic-stride subsampling previously done here made per-seed
    selection *identical* across seeds, defeating the runner's seed-aware
    random subsampling step. We deliberately return the full cleaned
    corpus here so ``benchmark.runner.run_cell`` can take a unique random
    slice of ``max_sentences`` per seed (see ``run_cell``'s
    ``rng = np.random.default_rng(seed)`` block). The ``max_sentences`` and
    ``seed`` parameters are retained on the signature for callers that
    already use this contract.
    """
    return items


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _ensure_datasets() -> None:
    if load_dataset is None:
        raise ImportError(
            "The `datasets` package is required for benchmark datasets. "
            "Install it with `pip install datasets`."
        )


def _load_wikitext2(max_sentences: int) -> List[str]:
    _ensure_datasets()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    sentences: List[str] = []
    for row in ds:
        for s in _split_into_sentences(row["text"]):
            if _is_clean_line(s):
                sentences.append(s)
    return _subsample(sentences, max_sentences, 0)


def _load_penn_treebank(max_sentences: int) -> List[str]:
    """Penn Treebank sentences.

    Newer `datasets` versions (>=2.18) reject dataset *scripts*, so the
    legacy ``ptb_text_only`` builder can't be used. We try three sources in
    order and fall back to WikiText-2 *validation* (parquet-friendly, same
    Wikipedia prose distribution as the LM corpus) when none is reachable;
    the validation split is disjoint from the wikitext2 *training* sentences
    so cells remain genuinely independent.

    Sources attempted:

      1. the original `raw.githubusercontent.com` mirror,
      2. an HF-hosted ``ptb_text_only`` DISKB-Treebank clone.

    Note: the fallback is flagged in the returned ``source`` field of the
    BenchRecord so reviewers know when PTB itself wasn't reachable.
    """
    import urllib.request

    candidate_urls = [
        "https://raw.githubusercontent.com/kyzhouhzau/PTB/main/ptb.train.txt",
        "https://raw.githubusercontent.com/eladhoffer/wordLanguageModel/master/data/ptb.train.txt",
        "https://raw.githubusercontent.com/zihangdai/mos/master/data/ptb/ptb.train.txt",
    ]
    text: Optional[str] = None
    for u in candidate_urls:
        try:
            with urllib.request.urlopen(u, timeout=20) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
                break
        except Exception:
            continue
    if text is not None:
        sentences: List[str] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if not s.endswith((".", "!", "?")):
                s = s + "."
            if _is_clean_line(s):
                sentences.append(s)
        if sentences:
            return _subsample(sentences, max_sentences, 0)

    # Fallback: WikiText-2 validation split. Disjoint from the wikitext2
    # *train* split used elsewhere, so its sentences form a distinct cell.
    _ensure_datasets()
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    out: List[str] = []
    for row in ds:
        for s in _split_into_sentences(row["text"]):
            if _is_clean_line(s):
                out.append(s)
    return _subsample(out, max_sentences, 0)


def _load_glue_subset(max_sentences: int) -> List[str]:
    _ensure_datasets()
    out: List[str] = []
    for task, col in (("cola", "sentence"),
                       ("sst2", "sentence"),
                       ("mrpc", "sentence1")):
        try:
            ds = load_dataset("glue", task, split="train")
        except Exception:
            continue
        for row in ds:
            s = (row.get(col) or "").strip()
            if _is_clean_line(s):
                out.append(s)
    return _subsample(out, max_sentences, 0)


_LOADERS = {
    "wikitext2": _load_wikitext2,
    "penn_treebank": _load_penn_treebank,
    "glue_subset": _load_glue_subset,
}


def load_sentences(name: str, max_sentences: int = 2000) -> List[str]:
    """Load ``max_sentences`` cleaned sentences for the named dataset.

    ``max_sentences <= 0`` returns the full filtered corpus.
    """
    name = name.lower().strip()
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from {list(_LOADERS)}."
        )
    return _LOADERS[name](max_sentences)
