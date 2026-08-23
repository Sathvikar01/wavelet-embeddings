"""Frequency-spectrum plots & reconstruction-error curves."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from analysis.energy import energy_ratio_low_high
from wavelets.base import WaveletDecomposer, WaveletDecomposition


def wavelet_spectrum(
    decomposition: WaveletDecomposition,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Stem plot of approximation + detail coefficient amplitudes."""
    parts = [decomposition.approx.ravel()] + decomposition.details
    fig, ax = plt.subplots(figsize=(14, 3.2))
    x_offset = 0
    for i, p in enumerate(parts):
        x = np.arange(x_offset, x_offset + len(p))
        if i == 0:
            ls, mk = ("tab:red", "o")
        else:
            ls, mk = ("tab:blue", "o")
        ax.stem(x, p, linefmt=ls, markerfmt=mk, basefmt=" ")
        if i == 0:
            ax.axvspan(x_offset - 0.5, x_offset + len(p) - 0.5,
                       alpha=0.08, color="red", label="approx")
        x_offset += len(p) + 4
    ax.set_title(title or f"Wavelet coefficients ({decomposition.wavelet_name}, "
                          f"level L={decomposition.level})")
    ax.set_xlabel("Wavelet coefficient index (approx | details L_high → L_low)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def reconstruction_error_curve(
    ratios: Sequence[float],
    cosines: Sequence[Sequence[float]],
    curves_labels: Sequence[str],
    title: str = "Reconstruction cosine vs. compression ratio",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Line plot of mean cosine vs. compression ratio for several settings."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, vals in zip(curves_labels, cosines):
        ax.plot(list(ratios), list(vals), marker="o", label=label)
    ax.set_xlabel("Ratio of small detail coefficients zeroed")
    ax.set_ylabel("Mean cosine(original, reconstructed)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def snr_curve(
    ratios: Sequence[float],
    snrs_db: Sequence[Sequence[float]],
    curves_labels: Sequence[str],
    title: str = "SNR vs. compression ratio",
    save_path: Optional[str] = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, vals in zip(curves_labels, snrs_db):
        ax.plot(list(ratios), list(vals), marker="s", label=label)
    ax.set_xlabel("Ratio of small detail coefficients zeroed")
    ax.set_ylabel("SNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path
