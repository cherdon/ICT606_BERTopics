"""
Picklable worker for UMAP 2D runs. Lives in a separate module so ProcessPoolExecutor
workers import this instead of __main__, avoiding re-running the main script in child processes.
"""
import numpy as np
from umap import UMAP


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
