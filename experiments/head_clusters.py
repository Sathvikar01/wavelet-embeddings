"""Head clustering experiment - cluster heads using wavelet metrics / raw cos.

We compute per-head feature vectors:

  * wavelet-features: numerical metrics from compute_head_metrics
                      (energy / entropy / sparsity / ...)
  * raw-attention features: PCA-reduced raw attention matrices

Cluster with:
  * KMeans
  * AgglomerativeClustering (Ward)
  * SpectralClustering

Save clusters CSV + reduction plots (PCA/t-SNE/UMAP) and a dendrogram.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from attention import AttentionLoader, AttentionWaveletDecomposer, compute_head_metrics
from attention.analyzer import head_feature_vector
from experiments.head_similarity import compute_pair_matrix
from visualization.head_clusters import (
    cluster_scatter, dendrogram_plot, cluster_comparison_scatter,
)


# --------------------------------------------------------------------------- #
# Feature builder
# --------------------------------------------------------------------------- #

def build_features(
    loader: AttentionLoader,
    decomposer: AttentionWaveletDecomposer,
    raw_pca_dim: int = 16,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """Return (wavelet_features, raw_features, head_id_list)."""
    wav_rows, raw_rows, ids = [], [], []
    from sklearn.decomposition import PCA
    raw_pca = PCA(n_components=raw_pca_dim, random_state=0)
    raws = []
    for L in range(loader.n_layers):
        for H in range(loader.n_heads):
            head = loader.load_head(L, H)
            ids.append((L, H))
            m = compute_head_metrics(head.normalized, decomposer)
            wav_rows.append(head_feature_vector(m))
            raws.append(head.normalized.ravel())
    raw_features = raw_pca.fit_transform(np.stack(raws))
    return (np.stack(wav_rows).astype(np.float64),
            raw_features.astype(np.float64), ids)


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #

def run_clustering(
    loader: AttentionLoader,
    decomposer_factory,
    save_dir: str,
    n_clusters: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, dict]:
    os.makedirs(save_dir, exist_ok=True)
    dec = decomposer_factory()
    X_wav, X_raw, ids = build_features(loader, dec)
    n = len(ids)
    if n < 4:
        return {}
    k = n_clusters or max(2, int(np.sqrt(n / 2)))
    k = min(k, n - 1)

    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
    from sklearn.preprocessing import StandardScaler
    wav_scaled = StandardScaler().fit_transform(X_wav)
    raw_scaled = StandardScaler().fit_transform(X_raw)

    labels_wavelet: Dict[str, np.ndarray] = {
        "kmeans":   KMeans(n_clusters=k, random_state=seed,
                            n_init="auto").fit_predict(wav_scaled),
        "agglomerative": AgglomerativeClustering(n_clusters=k,
                                                   linkage="ward").fit_predict(wav_scaled),
    }
    try:
        labels_wavelet["spectral"] = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors",
            random_state=seed).fit_predict(wav_scaled)
    except Exception:
        labels_wavelet["spectral"] = labels_wavelet["kmeans"]

    labels_raw: Dict[str, np.ndarray] = {
        "kmeans": KMeans(n_clusters=k, random_state=seed,
                          n_init="auto").fit_predict(raw_scaled),
    }
    # Save CSV
    import csv
    with open(os.path.join(save_dir, "cluster_assignments.csv"), "w",
              newline="", encoding="utf-8") as f:
        w_ = csv.writer(f)
        w_.writerow(["layer", "head", "wavelet_kmeans",
                      "wavelet_agglo", "wavelet_spectral",
                      "raw_kmeans"])
        for i, (L, H) in enumerate(ids):
            w_.writerow([
                L, H,
                int(labels_wavelet["kmeans"][i]),
                int(labels_wavelet["agglomerative"][i]),
                int(labels_wavelet["spectral"][i]),
                int(labels_raw["kmeans"][i]),
            ])
    # Figures
    cluster_scatter(X_wav, labels_wavelet["kmeans"], method="pca",
                    title="Wavelet-feature KMeans (PCA)",
                    save_dir=save_dir, basename="wav_kmeans_pca")
    cluster_scatter(X_wav, labels_wavelet["agglomerative"], method="tsne",
                    title="Wavelet-feature Agglomerative (t-SNE)",
                    save_dir=save_dir,
                    basename="wav_agglomerative_tsne")
    try:
        cluster_scatter(X_wav, labels_wavelet["agglomerative"], method="umap",
                        title="Wavelet-feature Agglomerative (UMAP)",
                        save_dir=save_dir,
                        basename="wav_agglomerative_umap")
    except Exception:
        pass
    try:
        dendrogram_plot(X_wav,
                         labels=[f"L{L}_H{H}" for L, H in ids],
                         save_dir=save_dir, basename="wavelet_dendrogram")
    except Exception:
        pass
    cluster_comparison_scatter(X_wav, X_raw, save_dir=save_dir,
                                 basename="wavelet_vs_cos_cluster")
    return {"wavelet": labels_wavelet, "raw": labels_raw, "k": k}
