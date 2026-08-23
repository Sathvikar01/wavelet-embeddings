"""Haar wavelet decomposer - specialisation of :class:`WaveletDecomposer`.

The Haar wavelet is the simplest wavelet: piece-wise constant, length-2
support.  Useful as a baseline because no real signal can be smoother than
the Haar basis predicts.
"""

from __future__ import annotations

from .base import WaveletDecomposer


class HaarDecomposer(WaveletDecomposer):
    def __init__(self, level: int | None = None, mode: str = "periodization"):
        super().__init__("haar", level=level, mode=mode)
