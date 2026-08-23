"""Coiflet-2 (coif2) decomposer.

Coiflets have 6 vanishing moments and are approximately symmetric - suitably
fine for subtle low-frequency detections within dense embeddings.
"""

from __future__ import annotations

from .base import WaveletDecomposer


class CoifletDecomposer(WaveletDecomposer):
    def __init__(self, level: int | None = None, mode: str = "periodization"):
        super().__init__("coif2", level=level, mode=mode)
