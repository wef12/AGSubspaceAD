"""Clustering of training tokens for per-cluster PCA anomaly detection.

Pipeline: materialise every training token (together with its optional,
per-token saliency-mask value) into arrays, partition the tokens with the
clustering algorithm selected by the ``--cluster_method`` CLI argument, and then
fit one memory-efficient :class:`~subspacead.core.pca.PCAModel` per cluster.
During inference each token is assigned to the nearest cluster center and scored
by that cluster's PCA model (see ``subspacead.post_process.scoring``).

Each clustering algorithm is implemented as an independent ``cluster_tokens_*``
function. Algorithms may differ substantially in their hyper-parameters and
auxiliary inputs (e.g. some use the per-token ``saliency_masks`` produced by
feature extraction), so no shared class hierarchy is imposed. Dispatch by name
happens through :data:`CLUSTER_FUNCTIONS`.

Implemented algorithms:

- ``kmeans``: K-means on the raw token features.
- ``wkmeans``: Weighted K-means, where the per-token distance weight is taken
  from the aligned DINOv2 ``saliency_masks`` (a pure NumPy Lloyd iteration).

Reserved algorithm names (not implemented yet, placeholders raise
``NotImplementedError``):

- ``recursive_split``: recursive splitting of the token space;
- ``hierarchical_split``: hierarchical splitting of the token space;
- ``adaptive_select``: adaptive selection of clusters/regions.

To implement a reserved algorithm, replace its placeholder with the real
implementation (and give it the parameter set it needs). No dispatch changes are
required: the function will be called as ``func(tokens, **kwargs)`` and must
return ``(labels, cluster_centers)``. Keep the method name in sync with the
``--cluster_method`` choices in ``src/subspacead/config.py``.
"""

import logging
import math

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from .pca import PCAModel


# ---------------------------------------------------------------------------
# Clustering algorithms. Each algorithm is an independent function returning
# ``(labels, cluster_centers)``, where ``labels`` is the integer cluster id per
# token (``[N]``) and ``cluster_centers`` is one representative point per
# cluster (``[C, D]``) used for nearest-center assignment during scoring.
# ---------------------------------------------------------------------------


def cluster_tokens_kmeans(
    tokens: np.ndarray,
    n_clusters: int = 8,
    random_state: int = None,
    n_init: int = 10,
    saliency_masks: np.ndarray = None,
):
    """K-means clustering of raw training tokens (scikit-learn backend).

    Args:
        tokens: Training tokens, shape ``[N, D]``.
        n_clusters: Number of clusters to form.
        random_state: Seed for reproducible initialisation.
        n_init: Number of k-means restarts; the best result is kept.
        saliency_masks: Per-token saliency values, shape ``[N]``. Not used by
            this algorithm; accepted for interface uniformity so that every
            clustering function can be dispatched through the same signature.

    Returns:
        ``(labels, cluster_centers)``: cluster id per token (``[N]``) and
        cluster centers (``[n_clusters, D]``).
    """
    logging.info(
        f"Fitting KMeans (n_clusters={n_clusters}, n_init={n_init}, "
        f"random_state={random_state}) on {tokens.shape[0]} tokens..."
    )
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    ).fit(tokens)
    logging.info(
        f"KMeans clustering done: {n_clusters} clusters; "
        f"{tokens.shape[0]} tokens assigned."
    )
    return model.labels_, model.cluster_centers_


def _validate_saliency_weights(saliency_masks: np.ndarray, num_tokens: int) -> np.ndarray:
    """Validate and normalise the per-token saliency weights for WKMeans.

    Returns a non-negative float64 vector of length ``num_tokens``. Negative
    weights are clamped to zero; if the weight vector would sum to zero the
    caller is warned and uniform weights are returned (plain K-means behaviour).
    """
    if saliency_masks is None:
        raise ValueError(
            "Clustering method 'wkmeans' requires per-token saliency weights "
            "(saliency_masks) but none were provided. Make sure the training "
            "feature generator yields aligned saliency masks (e.g. "
            "--bg_mask_method dino_saliency), or use 'kmeans' instead."
        )
    weights = np.asarray(saliency_masks, dtype=np.float64).reshape(-1)
    if weights.shape[0] != num_tokens:
        raise ValueError(
            f"Length of saliency_masks ({weights.shape[0]}) does not match the "
            f"number of tokens ({num_tokens})."
        )
    if np.any(weights < 0):
        logging.warning(
            "WKMeans: clamping %d negative saliency weight(s) to zero.",
            int(np.sum(weights < 0)),
        )
        weights = np.maximum(weights, 0.0)
    if weights.sum() <= 0:
        logging.warning(
            "WKMeans: all saliency weights are zero; falling back to uniform "
            "weights (equivalent to plain K-means)."
        )
        weights = np.ones(num_tokens, dtype=np.float64)
    return weights


def _weighted_kmeans_plusplus(
    tokens: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weighted K-means++ style initialisation of ``n_clusters`` centres.

    The first centre is sampled with probability proportional to the saliency
    weight; every subsequent centre is sampled with probability proportional to
    ``weight * (distance to the nearest already-chosen centre)^2`` so that both
    saliency and spatial coverage drive the initialisation.
    """
    num_tokens = tokens.shape[0]
    positive_idx = np.flatnonzero(weights > 0)
    if positive_idx.size == 0:
        first = int(rng.integers(num_tokens))
    else:
        p = weights[positive_idx]
        first = int(rng.choice(positive_idx, p=p / p.sum()))

    centers = [tokens[first].copy()]
    min_sq_dist = np.einsum("ij,ij->i", tokens - tokens[first], tokens - tokens[first])
    unselected = np.ones(num_tokens, dtype=bool)
    unselected[first] = False

    while len(centers) < n_clusters:
        cand_idx = np.flatnonzero(unselected)
        d2 = min_sq_dist[cand_idx]
        probs = weights[cand_idx] * d2
        if probs.sum() > 0:
            nxt = int(rng.choice(cand_idx, p=probs / probs.sum()))
        else:
            # No positive-weight candidate left; fall back to the farthest point.
            nxt = int(cand_idx[np.argmax(d2)])
        diff = tokens - tokens[nxt]
        dist_to_new = np.einsum("ij,ij->i", diff, diff)
        min_sq_dist = np.minimum(min_sq_dist, dist_to_new)
        centers.append(tokens[nxt].copy())
        unselected[nxt] = False
    return np.stack(centers)


def _assign_tokens_chunked(
    tokens: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    chunk_size: int = 8192,
):
    """Assign every token to its nearest centre and compute the weighted inertia.

    Squared Euclidean distances are computed in chunks (via the expanded form
    ``||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2``) to keep memory bounded.

    Returns:
        ``(labels, inertia)``: cluster id per token (``[N]``) and the weighted
        within-cluster sum of squares ``sum_i w_i * min_c ||x_i - c||^2``.
    """
    num_tokens, dim = tokens.shape
    num_clusters = centers.shape[0]
    labels = np.empty(num_tokens, dtype=np.int64)
    inertia = 0.0
    center_sq_norm = np.einsum("cd,cd->c", centers, centers)
    for start in range(0, num_tokens, chunk_size):
        chunk = tokens[start : start + chunk_size]
        x_sq_norm = np.einsum("nd,nd->n", chunk, chunk)
        sq_dist = (
            x_sq_norm[:, None]
            - 2.0 * np.dot(chunk, centers.T)
            + center_sq_norm[None, :]
        )
        np.maximum(sq_dist, 0.0, out=sq_dist)
        chunk_labels = np.argmin(sq_dist, axis=1)
        labels[start : start + chunk.shape[0]] = chunk_labels
        inertia += float(
            np.dot(weights[start : start + chunk.shape[0]], sq_dist[np.arange(chunk.shape[0]), chunk_labels])
        )
    return labels, inertia


def _weighted_centroid_update(
    tokens: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
):
    """Replace each centre by the saliency-weighted mean of its members.

    Returns:
        ``(centers, empty_clusters)``: the updated centres (``[C, D]``) and the
        list of cluster ids that received no token or whose members' weights
        sum to zero (they are re-seeded from actual tokens afterwards).
    """
    dim = tokens.shape[1]
    centers = np.zeros((n_clusters, dim), dtype=np.float64)
    weight_sum = np.bincount(labels, weights=weights, minlength=n_clusters)
    empty_clusters = []
    for c in range(n_clusters):
        mask = labels == c
        # A cluster whose members all carry zero weight would produce a 0/0 NaN
        # centre; treat it as empty and let _reinit_empty_centers seed it from
        # an actual token instead.
        if not mask.any() or weight_sum[c] <= 0:
            empty_clusters.append(c)
            continue
        centers[c] = np.sum(weights[mask, None] * tokens[mask], axis=0) / weight_sum[c]
    return centers, empty_clusters


def _reinit_empty_centers(
    tokens: np.ndarray,
    centers: np.ndarray,
    empty_clusters: list,
    chunk_size: int = 8192,
):
    """Reinitialise empty cluster centres with the farthest unattended tokens.

    For each empty cluster, the token farthest from the current (non-empty)
    centres is used as its new centre, guaranteeing ``n_clusters`` centres are
    always returned.
    """
    if not empty_clusters:
        return
    valid = np.ones(centers.shape[0], dtype=bool)
    valid[empty_clusters] = False
    valid_centers = centers[valid]
    num_tokens = tokens.shape[0]
    min_sq_dist = np.empty(num_tokens, dtype=np.float64)
    valid_sq_norm = np.einsum("cd,cd->c", valid_centers, valid_centers)
    for start in range(0, num_tokens, chunk_size):
        chunk = tokens[start : start + chunk_size]
        x_sq_norm = np.einsum("nd,nd->n", chunk, chunk)
        sq_dist = (
            x_sq_norm[:, None]
            - 2.0 * np.dot(chunk, valid_centers.T)
            + valid_sq_norm[None, :]
        )
        min_sq_dist[start : start + chunk.shape[0]] = sq_dist.min(axis=1)
    for c in empty_clusters:
        nxt = int(np.argmax(min_sq_dist))
        centers[c] = tokens[nxt]
        diff = tokens - tokens[nxt]
        dist_to_new = np.einsum("ij,ij->i", diff, diff)
        min_sq_dist = np.minimum(min_sq_dist, dist_to_new)


def _fit_weighted_kmeans_once(
    tokens: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
):
    """Run one Weighted K-means optimisation (Lloyd iterations)."""
    centers = _weighted_kmeans_plusplus(tokens, weights, n_clusters, rng)
    for _ in range(max_iter):
        labels, _ = _assign_tokens_chunked(tokens, centers, weights)
        new_centers, empty_clusters = _weighted_centroid_update(
            tokens, weights, labels, n_clusters
        )
        if empty_clusters:
            _reinit_empty_centers(tokens, new_centers, empty_clusters)
        shift = (
            np.max(np.linalg.norm(new_centers - centers, axis=1))
            if n_clusters > 0
            else 0.0
        )
        centers = new_centers
        if shift < tol:
            break
    labels, inertia = _assign_tokens_chunked(tokens, centers, weights)
    return labels, centers, inertia


def cluster_tokens_wkmeans(
    tokens: np.ndarray,
    n_clusters: int = 8,
    random_state: int = None,
    n_init: int = 10,
    saliency_masks: np.ndarray = None,
    max_iter: int = 300,
    tol: float = 1e-4,
):
    """Weighted K-means clustering of training tokens (pure NumPy backend).

    Minimises the saliency-weighted within-cluster sum of squares

        sum_i  w_i * ||x_i - mu_{labels[i]}||^2

    where the per-token weight ``w_i`` is the aligned DINOv2 saliency-map value
    ``saliency_masks[i]``: tokens DINOv2 deems more salient pull their cluster
    centre towards them more strongly. In the assignment step each token still
    goes to its nearest centre (the constant ``w_i`` cannot change the argmin),
    while the update step replaces every centre with the saliency-weighted mean
    of its members.

    Centres are initialised with a Weighted K-means++ scheme and the whole
    optimisation is repeated ``n_init`` times, keeping the run with the lowest
    weighted inertia.

    Args:
        tokens: Training tokens, shape ``[N, D]``.
        n_clusters: Number of clusters to form.
        random_state: Seed for reproducible initialisation.
        n_init: Number of independent runs; the best (lowest weighted inertia)
            result is kept.
        saliency_masks: Per-token DINOv2 saliency weights, shape ``[N]``.
            Values are clamped to ``>= 0``; when they would sum to zero the
            algorithm falls back to uniform weights (plain K-means behaviour).
        max_iter: Maximum number of Lloyd iterations per run.
        tol: Convergence threshold on the maximum centre displacement.

    Returns:
        ``(labels, cluster_centers)``: cluster id per token (``[N]``) and
        cluster centers (``[n_clusters, D]``).
    """
    tokens = np.asarray(tokens, dtype=np.float64)
    num_tokens, dim = tokens.shape
    if n_clusters <= 0:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}.")
    if num_tokens < n_clusters:
        raise ValueError(
            f"n_clusters ({n_clusters}) cannot exceed the number of tokens "
            f"({num_tokens})."
        )
    weights = _validate_saliency_weights(saliency_masks, num_tokens)
    logging.info(
        f"Fitting Weighted KMeans (n_clusters={n_clusters}, n_init={n_init}, "
        f"random_state={random_state}) on {num_tokens} tokens with "
        f"per-token saliency weights..."
    )

    rng = np.random.default_rng(random_state)
    best_labels, best_centers, best_inertia = None, None, np.inf
    for run in range(n_init):
        labels, centers, inertia = _fit_weighted_kmeans_once(
            tokens, weights, n_clusters, rng, max_iter, tol
        )
        logging.debug(
            f"WKMeans run {run + 1}/{n_init}: weighted inertia = {inertia:.6e}."
        )
        if inertia < best_inertia:
            best_labels, best_centers, best_inertia = labels, centers, inertia

    logging.info(
        f"Weighted KMeans clustering done: {n_clusters} clusters; "
        f"{num_tokens} tokens assigned (weighted inertia={best_inertia:.6e})."
    )
    return best_labels, best_centers


def cluster_tokens_recursive_split(tokens: np.ndarray, **kwargs):
    """Reserved placeholder for the RecursiveSplit algorithm.

    Not implemented yet. Expected to recursively split the token space into
    regions (may rely on ``saliency_masks`` or spatial/token layout
    information). Its dedicated parameter set will be defined together with the
    implementation.
    """
    raise NotImplementedError(
        "Clustering method 'recursive_split' is reserved but not implemented yet."
    )


def cluster_tokens_hierarchical_split(tokens: np.ndarray, **kwargs):
    """Reserved placeholder for the HierarchicalSplit algorithm.

    Not implemented yet. Expected to build a hierarchy of token-space
    partitions (may rely on ``saliency_masks`` or spatial/token layout
    information). Its dedicated parameter set will be defined together with the
    implementation.
    """
    raise NotImplementedError(
        "Clustering method 'hierarchical_split' is reserved but not implemented yet."
    )


def cluster_tokens_adaptive_select(tokens: np.ndarray, **kwargs):
    """Reserved placeholder for the AdaptiveSelect algorithm.

    Not implemented yet. Expected to adaptively select clusters/regions for
    per-cluster PCA training (may rely on ``saliency_masks`` or spatial/token
    layout information). Its dedicated parameter set will be defined together
    with the implementation.
    """
    raise NotImplementedError(
        "Clustering method 'adaptive_select' is reserved but not implemented yet."
    )


# Registry of available clustering algorithms: method name -> algorithm function.
# Keep the names in sync with the "--cluster_method" choices in config.py.
CLUSTER_FUNCTIONS = {
    "kmeans": cluster_tokens_kmeans,
    "wkmeans": cluster_tokens_wkmeans,
    "recursive_split": cluster_tokens_recursive_split,
    "hierarchical_split": cluster_tokens_hierarchical_split,
    "adaptive_select": cluster_tokens_adaptive_select,
}


def run_clustering(cluster_method: str, tokens: np.ndarray, **kwargs):
    """Run the clustering algorithm selected by ``cluster_method``.

    Extra ``kwargs`` (e.g. ``n_clusters``, ``random_state``, ``saliency_masks``)
    are forwarded to the selected ``cluster_tokens_*`` function, which consumes
    only the parameters it needs.

    Returns:
        ``(labels, cluster_centers)``.
    """
    if cluster_method not in CLUSTER_FUNCTIONS:
        raise ValueError(
            f"Unknown clustering method '{cluster_method}'. "
            f"Available methods: {sorted(CLUSTER_FUNCTIONS)}."
        )
    cluster_func = CLUSTER_FUNCTIONS[cluster_method]
    logging.info(f"Running clustering algorithm '{cluster_method}'...")
    return cluster_func(tokens, **kwargs)


def collect_tokens(
    feature_generator, num_batches: int, desc: str = "Token Collection"
):
    """Materialise every batch from ``feature_generator`` into per-token arrays.

    Each batch yielded by ``feature_generator`` is either a bare token array
    ``[B, D]`` or a ``(tokens, saliency_masks)`` tuple whose second element
    holds the aligned per-token saliency values (``[B * tokens_per_image]``).
    The tuple form is used by the training generators in ``main.py`` so that
    saliency information can reach clustering algorithms that need it.

    Args:
        feature_generator: Zero-argument callable returning an iterator over
            batches.
        num_batches: Number of batches the generator yields (progress bar).
        desc: Description shown on the progress bar.

    Returns:
        ``(tokens, saliency_masks)``: ``tokens`` has shape ``[N, D]``;
        ``saliency_masks`` has shape ``[N]`` when the generator provides them,
        otherwise ``None``.
    """
    token_batches = []
    mask_batches = []
    for batch in tqdm(feature_generator(), desc=desc, total=num_batches):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            tokens_batch, masks_batch = batch
            if masks_batch is not None:
                mask_batches.append(masks_batch)
        else:
            tokens_batch = batch
        token_batches.append(tokens_batch)

    tokens = np.concatenate(token_batches, axis=0)
    saliency_masks = (
        np.concatenate(mask_batches, axis=0) if mask_batches else None
    )
    return tokens, saliency_masks


def _fit_per_cluster_pca(
    tokens: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    batch_size: int,
    pca_dim: int,
    pca_ev: float,
    whiten: bool,
) -> list:
    """Fit one :class:`PCAModel` per cluster on ``tokens``.

    Returns a list aligned with the clusterer's ``cluster_centers_``: entry ``c``
    is the fitted PCA parameter dict, or ``None`` when cluster ``c`` is degenerate
    (fewer than two tokens).
    """
    feature_dim = tokens.shape[1]
    cluster_pca_params = []
    for c in range(n_clusters):
        cluster_tokens = tokens[labels == c]
        if cluster_tokens.shape[0] < 2:
            logging.warning(
                f"Cluster {c} has only {cluster_tokens.shape[0]} token(s); "
                "skipping per-cluster PCA."
            )
            cluster_pca_params.append(None)
            continue
        logging.info(
            f"Fitting PCA for cluster {c}: {cluster_tokens.shape[0]} tokens."
        )
        cluster_num_batches = math.ceil(cluster_tokens.shape[0] / batch_size)

        def cluster_feature_generator(
            tokens=cluster_tokens, batch_size=batch_size
        ):
            for i in range(0, tokens.shape[0], batch_size):
                yield tokens[i : i + batch_size]

        pca_model_c = PCAModel(k=pca_dim, ev=pca_ev, whiten=whiten)
        cluster_pca_params.append(
            pca_model_c.fit(
                cluster_feature_generator,
                feature_dim,
                cluster_tokens.shape[0],
                cluster_num_batches,
            )
        )
    return cluster_pca_params


def fit_cluster_pca(
    feature_generator,
    num_batches: int,
    cluster_method: str = "kmeans",
    n_clusters: int = 8,
    batch_size: int = 1,
    pca_dim: int = None,
    pca_ev: float = 0.99,
    whiten: bool = False,
    random_state: int = None,
    clusterer_kwargs: dict = None,
):
    """Collect training tokens, cluster them, and fit one PCA model per cluster.

    The token batches are collected once (together with aligned saliency masks
    when the generator provides them) and passed on to the clustering function
    selected by ``cluster_method``; the resulting per-cluster PCA models are
    then fit with a streaming two-pass fit on each cluster's tokens.

    Args:
        feature_generator: Zero-argument callable returning an iterator over
            batches (see :func:`collect_tokens`).
        num_batches: Number of batches the generator yields.
        cluster_method: Name of the clustering algorithm (see
            ``CLUSTER_FUNCTIONS`` / ``--cluster_method``).
        n_clusters: Number of clusters to form.
        batch_size: Batch size used by the per-cluster streaming PCA fits.
        pca_dim / pca_ev / whiten: PCA hyper-parameters forwarded to ``PCAModel``.
        random_state: Seed for reproducible clustering.
        clusterer_kwargs: Optional extra keyword arguments forwarded to the
            selected clustering function (algorithm-specific hyper-parameters).

    Returns:
        ``(cluster_centroids, cluster_pca_params)``; both are ``None`` when no
        tokens could be collected. ``cluster_centroids`` has shape ``[C, D]``
        and ``cluster_pca_params[c]`` is the fitted PCA parameter dict (or
        ``None`` for degenerate clusters with fewer than two tokens).
    """
    clusterer_kwargs = clusterer_kwargs or {}
    all_tokens, saliency_masks = collect_tokens(
        feature_generator,
        num_batches,
        desc=f"Token Collection for {cluster_method}",
    )
    logging.info(
        f"Collected {all_tokens.shape[0]} tokens with dim={all_tokens.shape[1]}."
    )
    if all_tokens.shape[0] == 0:
        logging.error(
            "No training tokens collected; skipping clustering and per-cluster PCA."
        )
        return None, None

    labels, centroids = run_clustering(
        cluster_method,
        all_tokens,
        n_clusters=n_clusters,
        random_state=random_state,
        saliency_masks=saliency_masks,
        **clusterer_kwargs,
    )

    cluster_pca_params = _fit_per_cluster_pca(
        all_tokens,
        labels,
        centroids.shape[0],
        batch_size,
        pca_dim,
        pca_ev,
        whiten,
    )
    logging.info(
        f"'{cluster_method}' clustering done: {centroids.shape[0]} clusters; "
        f"trained {len(cluster_pca_params)} per-cluster PCA models. "
        "Cluster centers returned for scoring."
    )
    return centroids, cluster_pca_params
