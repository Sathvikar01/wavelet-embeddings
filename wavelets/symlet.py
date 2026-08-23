"""Symlet-4 (sym4) decomposer.

Symlets are near-symmetric modifications of Daubechies wavelets with
better phase symmetry - useful for embeddings where asymmetry would
bias downstream statistics.
"""

from __future__ import annotations

from .base import WaveletDecomposer


class SymletDecomposer(WaveletDecomposer):
    def __init__(self, level: int | None = None, mode: str = "periodization"):
        super().__init__("sym4", level=level, mode=mode)
