"""
Picklable workers for UMAP and HDBSCAN finetuning. Lives in a separate module so
ProcessPoolExecutor workers import this instead of __main__, avoiding re-running
the main script in child processes.
"""
import numpy as np
from hdbscan import HDBSCAN
from umap import UMAP


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap_2d(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP with n_components=2 for scatter plot; returns (n_samples, 2)."""
    umap = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return umap.fit_transform(embeddings)


def _umap_2d_worker(params: tuple) -> np.ndarray:
    """Worker for ProcessPoolExecutor: (embeddings, n_neighbors, min_dist, random_state) -> coords."""
    embeddings, n_neighbors, min_dist, random_state = params
    return run_umap_2d(embeddings, n_neighbors, min_dist, random_state)


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan_2d(
    embeddings_2d: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
) -> HDBSCAN:
    """Run HDBSCAN on 2D embeddings; returns fitted clusterer (has .labels_, .condensed_tree_)."""
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    clusterer.fit(embeddings_2d)
    return clusterer


def _hdbscan_2d_worker(params: tuple) -> HDBSCAN:
    """Worker for ProcessPoolExecutor: (embeddings_2d, min_cluster_size, min_samples, metric, cluster_selection_method) -> fitted HDBSCAN."""
    (
        embeddings_2d,
        min_cluster_size,
        min_samples,
        metric,
        cluster_selection_method,
    ) = params
    return run_hdbscan_2d(
        embeddings_2d,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric or "euclidean",
        cluster_selection_method=cluster_selection_method or "eom",
    )
