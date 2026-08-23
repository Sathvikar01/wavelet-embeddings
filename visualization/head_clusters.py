"""Head clustering visualization.

* Generic 2-D reduction scatter (PCA / t-SNE / UMAP)
* Dendrogram for agglomerative clustering
* KMeans-styled scatter
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _save(fig, save_dir: str, basename: str):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, basename + ".png"), dpi=140,
                bbox_inches="tight")
    fig.savefig(os.path.join(save_dir, basename + ".pdf"),
                bbox_inches="tight")
    plt.close(fig)


def reduce_2d(X: np.ndarray, method: str = "tsne",
              seed: int = 0) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        n = X.shape[0]
        perp = min(30, max(5, n - 1))
        return TSNE(n_components=2, perplexity=perp, random_state=seed,
                    init="pca").fit_transform(X)
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
        except Exception:
            # Fallback to PCA if UMAP unavailable
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=seed).fit_transform(X)
    raise ValueError("method must be pca/tsne/umap")


def cluster_scatter(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    title: str = "",
    save_dir: Optional[str] = None,
    basename: Optional[str] = None,
    show: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    coords = reduce_2d(X, method=method)
    fig, ax = plt.subplots(figsize=(8, 7))
    uniq = sorted(set(labels.tolist() if isinstance(labels, np.ndarray)
                       else list(labels)))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1)))
    for i, c in enumerate(uniq):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1],
                   color=cmap[i % len(cmap)], label=f"cluster {c}", s=40)
    ax.set_title(f"{title} ({method.upper()})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_dir and basename:
        _save(fig, save_dir, basename)
    elif show:
        plt.show()
        plt.close(fig)
    else:
        plt.close(fig)
    return coords, labels


def dendrogram_plot(
    X: np.ndarray,
    labels: List[str],
    save_dir: str,
    basename: str = "dendrogram",
    title: str = "Wavelet-feature dendrogram",
):
    from scipy.cluster.hierarchy import linkage, dendrogram
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(11, 0.3 * max(len(labels), 6) + 4))
    dendrogram(
        Z, labels=labels, leaf_font_size=8, ax=ax,
        color_threshold=0.7 * Z[:, 2].max(),
        orientation="right",
    )
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_dir, basename)


def cluster_comparison_scatter(
    X_wav_features: np.ndarray,
    X_cos_features: np.ndarray,
    save_dir: str,
    basename: str = "wavelet_vs_cos_cluster",
):
    """PCA scatter: clusters by wavelet features vs cluster by raw-attention
    features, side-by-side.
    """
    from sklearn.cluster import KMeans
    n = X_wav_features.shape[0]
    if n == 0:
        return
    # Determine feasible k = sqrt(n/2) capped at 6
    k = max(2, min(6, int(np.sqrt(n / 2))))
    km_w = KMeans(n_clusters=k, random_state=0,
                   n_init="auto").fit_predict(X_wav_features)
    km_c = KMeans(n_clusters=k, random_state=0,
                   n_init="auto").fit_predict(X_cos_features)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, Xc, lbl, km in zip(
        axes,
        [X_wav_features, X_cos_features],
        ["Wavelet-feature cluster", "Cosine-feature cluster"],
        [km_w, km_c]
    ):
        Z2 = reduce_2d(Xc, method="pca")
        for c in sorted(set(km.tolist())):
            m = km == c
            ax.scatter(Z2[m, 0], Z2[m, 1], label=f"cluster {c}", s=40)
        ax.set_title(lbl)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Comparison of head clusters built from wavelet vs raw attention "
                  "features")
    fig.tight_layout()
    _save(fig, save_dir, basename)
