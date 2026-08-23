"""t-SNE / PCA before & after compression."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _safe_tsne(X: np.ndarray, n_components: int = 2, perplexity: int = 30,
               seed: int = 0) -> np.ndarray:
    from sklearn.manifold import TSNE
    n = X.shape[0]
    if n <= 10:
        return np.zeros((n, n_components))
    perplexity = min(perplexity, max(5, n - 1))
    try:
        return TSNE(
            n_components=n_components, perplexity=perplexity,
            random_state=seed, init="pca",
        ).fit_transform(X)
    except Exception:
        # Fallback to PCa if t-SNE dies
        return _pca(X, n_components=n_components)


def _pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    from sklearn.decomposition import PCA
    n = X.shape[0]
    n_components = min(n_components, max(1, n - 1))
    return PCA(n_components=n_components, random_state=0).fit_transform(X)


def embed_2d(X: np.ndarray, method: str = "tsne") -> np.ndarray:
    if X.shape[0] < 3:
        return np.zeros((X.shape[0], 2))
    if method == "tsne":
        return _safe_tsne(X)
    if method == "pca":
        return _pca(X)
    raise ValueError("method must be 'tsne' or 'pca'")


def scatter_2d(
    coords: np.ndarray,
    labels: Sequence[str],
    colors: Optional[Sequence[str]] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 7))
    uniq = list(dict.fromkeys(labels))
    pal = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
    color_lookup = {lab: pal[i % len(pal)] for i, lab in enumerate(uniq)}
    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, color=color_lookup[labels[i]], s=30)
        if i < 100:
            ax.text(x + 0.3, y + 0.3, str(labels[i]), fontsize=7)
    handles = [ax.scatter([], [], color=color_lookup[lab], label=lab) for lab in uniq]
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.set_title(title or "Embedding projection")
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def before_after(
    original: np.ndarray,
    reconstructed: np.ndarray,
    labels: Sequence[str],
    method: str = "tsne",
    title: str = "Embeddings before / after compression",
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Side-by-side t-SNE / PCA before vs. after compression."""
    emb_o = embed_2d(original, method=method)
    emb_r = embed_2d(reconstructed, method=method)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, emb, lbl_sub in zip(axes, [emb_o, emb_r], ["Original", "Compressed"]):
        uniq = list(dict.fromkeys(labels))
        pal = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
        color_lookup = {lab: pal[i % len(pal)] for i, lab in enumerate(uniq)}
        for i, (x, y) in enumerate(emb):
            ax.scatter(x, y, color=color_lookup[labels[i]], s=20, alpha=0.8)
        handles = [ax.scatter([], [], color=color_lookup[lab], label=lab) for lab in uniq]
        ax.legend(handles=handles, loc="best", fontsize=7)
        ax.set_title(f"{lbl_sub} ({method.upper()})")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path
