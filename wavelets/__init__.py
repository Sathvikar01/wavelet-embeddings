"""Wavelet decomposition dispatch table.

Convenience constructors: ``make_decomposer("haar")`` returns the right
:class:`WaveletDecomposer` subclass instance.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import WaveletDecomposer, WaveletDecomposition, SUPPORTED_WAVELETS
from .haar import HaarDecomposer
from .db4 import DB4Decomposer
from .symlet import SymletDecomposer
from .coiflet import CoifletDecomposer
from .base import wavelet_similarity


_REGISTER: Dict[str, Type[WaveletDecomposer]] = {
    "haar": HaarDecomposer,
    "db4": DB4Decomposer,
    "sym4": SymletDecomposer,
    "coif2": CoifletDecomposer,
    # aliases
    "symlet": SymletDecomposer,
    "coiflet": CoifletDecomposer,
    "daubechies": DB4Decomposer,
}


def make_decomposer(name: str, level: int | None = None,
                    mode: str = "periodization") -> WaveletDecomposer:
    """Instantiate the wavelet decomposer registered under ``name``."""
    key = name.lower().strip()
    if key not in _REGISTER:
        raise ValueError(
            f"Unknown wavelet '{name}'. Available: {list(_REGISTER)}"
        )
    return _REGISTER[key](level=level, mode=mode)


__all__ = [
    "WaveletDecomposer",
    "WaveletDecomposition",
    "SUPPORTED_WAVELETS",
    "HaarDecomposer",
    "DB4Decomposer",
    "SymletDecomposer",
    "CoifletDecomposer",
    "make_decomposer",
    "wavelet_similarity",
]
