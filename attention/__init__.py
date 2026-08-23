"""Attention wavelet analysis modules (Phase 3).

Entry points:

* :class:`AttentionExtractor`        -- forward pass and per-head dumps
* :class:`AttentionLoader`           -- read per-(layer, head) .npz
* :class:`AttentionWaveletDecomposer` -- 2-D DWT and reconstruction
* :func:`compute_head_metrics`        -- numerical metrics per head
"""

from .extractor import AttentionExtractor, AttentionOutput, extract_and_save
from .loader import AttentionLoader, HeadMatrix, find_available_models
from .analyzer import (
    AttentionWaveletDecomposer, HeadDecomposition,
    compute_head_metrics, head_feature_vector, head_wavelet_flat_coeffs,
    SUPPORTED_WAVELETS, METRIC_KEYS,
)


__all__ = [
    "AttentionExtractor", "AttentionOutput", "extract_and_save",
    "AttentionLoader", "HeadMatrix", "find_available_models",
    "AttentionWaveletDecomposer", "HeadDecomposition",
    "compute_head_metrics", "head_feature_vector", "head_wavelet_flat_coeffs",
    "SUPPORTED_WAVELETS", "METRIC_KEYS",
]
