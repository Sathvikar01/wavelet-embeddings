"""Contextual-embedding data module.

A small curated polysemy / homonym dataset used across the Phase-2 experiments.

Each anchor key maps to a list of (sentence, sense_label) pairs.  The
sentence must contain the literal anchor token (case-insensitive), so the
extractor can locate it.  The sense_label is a short string that we group by
in the experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ContextExample:
    sentence: str
    sense: str           # short semantic sense label ("river", "money", "...)
    anchor: str          # the surface-form anchor token (e.g. "bank")


# --------------------------------------------------------------------------- #
# Curated anchor -> contexts (homonym / polysemy pairs designed to disambiguate
# different meanings of the same surface token).
# --------------------------------------------------------------------------- #

POLYSEMY_DATASET: Dict[str, List[ContextExample]] = {
    # ---------- bank ----------
    "bank": [
        ContextExample("The bank approved the loan for the new house.", "money", "bank"),
        ContextExample("The fisher sat on the bank of the quiet river.", "river", "bank"),
        ContextExample("She deposited cash in the bank on the corner.", "money", "bank"),
        ContextExample("The boat reached the bank just before sunrise.", "river", "bank"),
        ContextExample("The central bank raised interest rates today.", "money", "bank"),
        ContextExample("Red flowers grew along the muddy bank of the river.", "river", "bank"),
    ],
    # ---------- plant ----------
    "plant": [
        ContextExample("The new car plant will open next year.", "factory", "plant"),
        ContextExample("She watered the plant on the windowsill every morning.", "organism", "plant"),
        ContextExample("Thieves planted a small plant outside the factory entrance.", "factory", "plant"),
        ContextExample("The chemical plant exploded last night on the news.", "factory", "plant"),
        ContextExample("A rare plant in the garden attracts many butterflies.", "organism", "plant"),
    ],
    # ---------- match ----------
    "match": [
        ContextExample("They won the tennis match in three sets.", "game", "match"),
        ContextExample("Her shirt matches the colour of her shoes perfectly.", "correspond", "match"),
        ContextExample("He struck a match to light the campfire.", "firestick", "match"),
        ContextExample("The new evidence matches our theory very well.", "correspond", "match"),
        ContextExample("The final match of the tournament starts at noon.", "game", "match"),
    ],
    # ---------- bark ----------
    "bark": [
        ContextExample("The dog began to bark at the stranger.", "sound", "bark"),
        ContextExample("The tree's bark was thick and rough to the touch.", "tree", "bark"),
        ContextExample("A sudden bark startled the children in the garden.", "sound", "bark"),
        ContextExample("Some insects live just beneath the outer bark of the oak.", "tree", "bark"),
    ],
    # ---------- light ----------
    "light": [
        ContextExample("A soft light came through the window.", "illumination", "light"),
        ContextExample("The package was light enough to carry in one hand.", "weight", "light"),
        ContextExample("She wore a light jacket because the evening was cool.", "weight", "light"),
        ContextExample("The room filled with morning light.", "illumination", "light"),
    ],
    # ---------- rock ----------
    "rock": [
        ContextExample("The climber gripped a small rock on the cliff face.", "stone", "rock"),
        ContextExample("The band played loud rock music all night long.", "genre", "rock"),
        ContextExample("A heavy rock fell onto the road from the hillside.", "stone", "rock"),
        ContextExample("She preferred jazz over rock when studying.", "genre", "rock"),
    ],
    # ---------- crane ----------
    "crane": [
        ContextExample("The construction crane towered over the new building.", "machine", "crane"),
        ContextExample("A tall crane stood motionless at the edge of the lake.", "bird", "crane"),
        ContextExample("They used the crane to lift the steel beams into place.", "machine", "crane"),
        ContextExample("The white crane spread its wings over the water.", "bird", "crane"),
    ],
    # ---------- pen ----------
    "pen": [
        ContextExample("He signed the letter with a blue pen.", "writing", "pen"),
        ContextExample("The farmer built a new pen for the sheep.", "enclosure", "pen"),
        ContextExample("She kept a fountain pen on her desk.", "writing", "pen"),
        ContextExample("The pen held three pigs during the night.", "enclosure", "pen"),
    ],
    # ---------- ship ----------
    "ship": [
        ContextExample("The container ship arrived at the harbour at dawn.", "boat", "ship"),
        ContextExample("They decided to ship the goods by air freight instead.", "verb", "ship"),
        ContextExample("A large ship sailed past the lighthouse.", "boat", "ship"),
        ContextExample("The store will ship the order within two days.", "verb", "ship"),
    ],
    # ---------- mouse ----------
    "mouse": [
        ContextExample("I bought a wireless mouse for my laptop.", "device", "mouse"),
        ContextExample("A small mouse ran across the kitchen floor.", "animal", "mouse"),
        ContextExample("She moved the mouse and clicked on the icon.", "device", "mouse"),
        ContextExample("The cat chased the mouse behind the cupboard.", "animal", "mouse"),
    ],
}


# --------------------------------------------------------------------------- #
# Anchors we explicitly run the "meaning evolution" / thinner experiments on.
# --------------------------------------------------------------------------- #

DEFAULT_ANCHORS: List[str] = list(POLYSEMY_DATASET.keys())
DEFAULT_PAIRS: List[List[str]] = [["bank", "river"], ["plant", "factory"], ["crane", "machine"]]


def surface_anchor_indices(sentence: str, anchor: str) -> List[int]:
    """Return character offsets (i, j) for every case-insensitive occurrence
    of ``anchor`` in ``sentence`` as a whole word.
    """
    anchor = anchor.lower()
    sentence_l = sentence.lower()
    out: List[int] = []
    i = 0
    while True:
        pos = sentence_l.find(anchor, i)
        if pos < 0:
            return out
        # Whole-word boundary check.
        # Allow the anchor to be the stem of an inflected surface form (e.g.
        # "match" inside "matches"), since the extractor still needs to find the
        # anchor subtoken span in the inflected tokenization.
        prev = sentence_l[pos - 1] if pos > 0 else " "
        nxt_char = sentence_l[pos + len(anchor)] if pos + len(anchor) < len(sentence_l) else " "
        if not prev.isalnum() and (
            not nxt_char.isalnum()
            or nxt_char in "es"  # leave room for stems like match+es / match+ing
        ):
            out.append(pos)
        i = pos + 1
    return out
