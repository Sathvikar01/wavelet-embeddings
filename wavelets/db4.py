"""Daubechies-4 (db4) decomposer.

Db4 has 8 taps and is smoother than Haar; commonly used as a default for
signal analysis. Provides a good trade-off between compact support and
regularity.
"""

from __future__ import annotations

from .base import WaveletDecomposer


class DB4Decomposer(WaveletDecomposer):
    def __init__(self, level: int | None = None, mode: str = "periodization"):
        super().__init__("db4", level=level, mode=mode)
