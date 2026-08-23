"""Leave-one-cell-out ridge predictors for the Phase-6 combined-feature
experiment.

This module populates the ``ridge_wavelet_only``, ``ridge_attn_only`` and
``ridge_combined`` per-input correlations on every cell of a
``BenchResult`` AFTER the per-cell ``run_cell`` pass has finished. The
training protocol is leave-one-(model, dataset, seed)-cell-out:

    for held_cell in cells:
        fit ridge on (X, y) pooled over every cell != held_cell
        for each input in held_cell:
            y_hat = ridge.predict(held_input's X)
            r = pearson(-y_hat, held_input's measured per-head loss)
        store r sequence as held_cell.per_input["ridge_combined"]

No cell's own ``(X, y)`` ever participates in fitting the weights that
predict that cell. The protocol is therefore immune to the standard
"you fit on test" reviewer objection.

Ridge is fit with features standardised per-cell (see
``feature_matrix.build_feature_matrix``, which standardises within the
sampled cell's own X using that cell's own mean/std). The training set
is the concatenation of other cells' standardised X. Ridge alpha is
fixed at 1.0 across all folds -- this is deliberately a single-
hyperparameter model so cross-validation of alpha is not needed; the
goal is a *comparable* predictor, not a tuned one. If the reviewer asks,
we can swap to per-fold alpha via :class:`sklearn.linear_model.RidgeCV`
without changing the leakage surface.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from benchmark.feature_matrix import feature_names


RIDGE_PREDICTORS: Tuple[str, ...] = (
    "wavelet_only_model",
    "attention_only_model",
    "wavelet_feature_model",
)


def _group_for(pred: str) -> str:
    if pred == "wavelet_only_model":
        return "wavelet_only"
    if pred == "attention_only_model":
        return "attn_only"
    if pred == "wavelet_feature_model":
        return "combined"
    raise ValueError(f"Unknown ridge predictor {pred!r}.")


def _column_subset(X_full: np.ndarray,
                    full_names: Sequence[str],
                    group: str) -> np.ndarray:
    """Slice the combined-group X to the columns of the requested group."""
    if group == "combined":
        return X_full
    keep = [i for i, n in enumerate(full_names) if n in feature_names(group)]
    return X_full[:, keep]


def _fit_ridge(X: np.ndarray, y: np.ndarray,
                alpha: float = 1.0) -> Tuple[np.ndarray, float, float]:
    """Closed-form ridge (no sklearn dep). Returns (w, b, ss_res)."""
    # Augment with bias column. Ridge on (X|1) is the same as least squares
    # with L2 on weights but not on intercept -- standard formulation.
    n, d = X.shape
    if n == 0 or d == 0:
        return np.zeros(d), 0.0, 0.0
    A = np.concatenate([X, np.ones((n, 1))], axis=1)
    # Regularise weights but not intercept.
    reg = alpha * np.eye(d + 1)
    reg[-1, -1] = 0.0
    try:
        w = np.linalg.solve(A.T @ A + reg, A.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(A) @ y
    ss_res = float(((A @ w - y) ** 2).sum())
    return w[:-1], float(w[-1]), ss_res


def apply_ridge_looo(result, alpha: float = 1.0,
                      verbose: bool = False,
                      return_coeffs: bool = True):
    """Populate ``wavelet_only_model``, ``attention_only_model`` and
    ``wavelet_feature_model`` per-input correlations on every record in
    ``result`` IN PLACE. Optionally return a ``RidgeFoldsArtefact`` dict
    with per-fold learned coefficients (for ``feature_importance.csv``).

    The result is a list of ``BenchRecord`` (from ``benchmark.runner``).
    Skips records with no stashed per-input features (i.e. cells that ran
    with ``keep_per_input_features=False``); for those, ``per_input`` for
    the three ridge predictors is left at length 0 so they show up in the
    CSV but as missing/n=0 rows.

    Standardisation reminder: ``per_input_features[i]`` was already
    standardised within its cell by ``build_feature_matrix`` (z-score per
    feature using that cell's own mean/std). Pooling across cells in
    training is therefore done on already-cell-standardised features,
    which is the literature-standard "leave-one-site-out" protocol -- no
    global rescaler is required because each cell's standardiser is
    invariant to the cells it never saw.
    """
    recs = list(result.records)
    n_cells = len(recs)

    # Pre-compute the full-name lookup once per record.
    full_names = feature_names("combined")

    # Index every record carrying per-input feature matrices.
    cell_idx: List[int] = [i for i, r in enumerate(recs)
                            if r.per_input_features]
    if verbose:
        print(f"[ridge_looo] {len(cell_idx)}/{n_cells} cells carry per-input "
              "features; the remaining were run with "
              "`keep_per_input_features=False` and will have empty rows.")

    fold_coeffs: List[Dict] = []
    for held in cell_idx:
        held_rec = recs[held]
        train_X_by_group: dict = {}
        train_y: List[float] = []

        for j, other in enumerate(recs):
            if j == held:
                continue
            if not other.per_input_features:
                continue
            other_X = other.per_input_features
            other_y = other.per_input_loss
            X_stack = np.vstack(other_X) if isinstance(other_X, list) \
                        else other_X
            y_stack = np.concatenate(other_y) if isinstance(other_y, list) \
                        else other_y
            train_X_by_group.setdefault("combined", []).append(X_stack)
            train_y.extend(y_stack.tolist())

        if not train_y:
            for pred in RIDGE_PREDICTORS:
                held_rec.per_input.setdefault(pred, [
                    float("nan")] * len(held_rec.per_input_features))
            continue

        y_train = np.asarray(train_y, dtype=np.float64)
        X_train_combined = np.concatenate(
            train_X_by_group["combined"], axis=0)

        for pred in RIDGE_PREDICTORS:
            group = _group_for(pred)
            X_train = _column_subset(X_train_combined, full_names, group)
            w, b, _ = _fit_ridge(X_train, y_train, alpha=alpha)

            per_input_r: List[float] = []
            per_input_rho: List[float] = []
            per_input_tau: List[float] = []
            held_X_list = held_rec.per_input_features
            held_loss_list = held_rec.per_input_loss
            for X, loss in zip(held_X_list, held_loss_list):
                X_held = _column_subset(X, full_names, group)
                y_hat = X_held @ w + b
                # Predicted *importance* is y_hat (higher = keep).
                # Predicted *unimportance* is -y_hat. The benchmark
                # correlates predicted-unimportance with measured loss.
                pred_unimp = -y_hat
                loss_arr = np.asarray(loss, dtype=np.float64)
                r = _safe_pearson(pred_unimp, loss_arr)
                rho = _safe_spearman(pred_unimp, loss_arr)
                tau = _safe_kendall(pred_unimp, loss_arr)
                per_input_r.append(float(r))
                per_input_rho.append(float(rho))
                per_input_tau.append(float(tau))
            held_rec.per_input[pred] = per_input_r
            held_rec.per_input_spearman[pred] = per_input_rho
            held_rec.per_input_kendall[pred] = per_input_tau
            # Same input-level cosine-preservation / pruning-loss means
            # as the cold-start predictors computed in run_cell -- the
            # ablation losses haven't changed, so reuse them so reviewers
            # can compare predictor_metrics.csv rows side-by-side.
            held_rec.per_input_cosine_pres[pred] = \
                held_rec.per_input_cosine_pres.get("wavelet", [])
            held_rec.per_input_pruning_loss[pred] = \
                held_rec.per_input_pruning_loss.get("wavelet", [])

            if return_coeffs:
                # Per-fold coefficient record. The columns are the
                # group-specific feature names so feature_importance.csv
                # can join on (predictor, feature) cleanly.
                feature_names_group = feature_names(group)
                fold_coeffs.append({
                    "model": held_rec.model,
                    "dataset": held_rec.dataset,
                    "seed": held_rec.seed,
                    "predictor": pred,
                    "alpha": alpha,
                    "intercept": float(b),
                    "coefficients": {
                        n: float(c) for n, c in
                        zip(feature_names_group, w)
                    },
                })

        if verbose:
            print(f"  cell {held_rec.model}/{held_rec.dataset}/"
                    f"{held_rec.seed}: wavelet_feature_model "
                    f"mean={np.nanmean(per_input_r):+.4f} "
                    f"(n_inputs={len(per_input_r)})")

    if not return_coeffs:
        return None
    return {"folds": fold_coeffs,
            "feature_names": {g: feature_names(g)
                                for g in ("wavelet_only", "attn_only",
                                          "combined")}}


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if not x.size or not y.size or x.size != y.size:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    if denom < 1e-12:
        return 0.0
    return float((xm * ym).sum() / denom)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _safe_pearson(_as_rank(x), _as_rank(y))


def _safe_kendall(x: np.ndarray, y: np.ndarray) -> float:
    if not x.size or not y.size or x.size != y.size:
        return float("nan")
    if x.size < 2:
        return 0.0
    # O(n^2) -- fine because heads per cell < 200.
    rx = _as_rank(x)
    ry = _as_rank(y)
    n = x.size
    con = 0
    dis = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = (rx[j] - rx[i]) * (ry[j] - ry[i])
            if d > 0:
                con += 1
            elif d < 0:
                dis += 1
    return float((con - dis) / max(con + dis, 1))


def _as_rank(arr: np.ndarray) -> np.ndarray:
    """Average ranks (handle ties the way scipy default does)."""
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sums = {len(arr): 0.0}
    # Standard "fractional rank" assignment for ties.
    sorted_vals = arr[order]
    i = 0
    while i < arr.size:
        j = i
        while j + 1 < arr.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks

