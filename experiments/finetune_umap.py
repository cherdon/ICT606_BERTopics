"""
Finetune UMAP by visualising 2D projections for different n_neighbors and min_dist.

Loads documents (random 4k per CSV file, then capped at max_docs), embeds them once,
then runs UMAP with varying n_neighbors (or min_dist) and saves 3x3 scatterplot grids.
The 9 UMAP runs per sweep execute concurrently in separate processes (avoids Numba
thread-unsafe workqueue when using threads).
"""

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

# Project root for imports
_root = Path(__file__).resolve().parents[1]
if _root not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.finetuning_workers import _umap_2d_worker
from utils.bertopic_pipeline import get_embedding_model
from utils import prepare_documents_for_finetuning

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = _root / "data" / "covid19_twitter_dataset"
DEFAULT_TEXT_COLUMN = "original_text"
DEFAULT_PROCESSED_COLUMN = "processed_text"
DEFAULT_MIN_LENGTH = 10
DEFAULT_DEDUPE = True
DEFAULT_MAX_DOCS = 20_000
SAMPLE_PER_FILE = 4_000

# n_neighbors: balance local vs global structure. For many docs (10k+), very small
# values (2–5) can overfit and fragment; 15–200 is a typical range. Sweep from
# local (5) to global (200) so you can pick the right granularity for topic clusters.
N_NEIGHBORS_VALUES = [5, 10, 15, 25, 50, 75, 100, 150, 200]

# min_dist: how tightly points pack in 2D (0 = tight clusters, 1 = spread out). For
# topic modeling, 0.0–0.2 is common; higher values (0.5–1.0) give looser, more
# continuous layouts. Sweep full range to see over-clustering vs over-spread.
MIN_DIST_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

# Fixed UMAP params when sweeping the other (sensible defaults for large corpora)
FIXED_MIN_DIST = 0.0
FIXED_N_NEIGHBORS = 25

PLOTS_NEIGHBOURS_DIR = _root / "experiments" / "plots" / "umap"
PLOTS_DIST_DIR = _root / "experiments" / "plots" / "umap"


def load_documents_and_embeddings(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    processed_column: str = DEFAULT_PROCESSED_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    max_docs: int = DEFAULT_MAX_DOCS,
    random_state: int = 42,
):
    """Load documents via shared finetuning pipeline, then compute embeddings once."""
    print("Loading and preprocessing tweets (random 4k per file)...")
    documents, _ = prepare_documents_for_finetuning(
        data_dir=data_dir,
        text_column=text_column,
        processed_column=processed_column,
        min_length=min_length,
        dedupe=dedupe,
        max_docs=max_docs,
        sample_per_file=SAMPLE_PER_FILE,
        random_state=random_state,
    )
    if not documents:
        raise SystemExit("No documents after preprocessing. Exiting.")
    print(f"Documents: {len(documents)}")
    print("Computing embeddings...")
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    return documents, np.array(embeddings)


def plot_3x3_scattergrid(
    coords_list: list[np.ndarray],
    param_values: list,
    param_name: str,
    title_prefix: str,
    out_path: Path,
) -> None:
    """Plot 9 scatter plots in a 3x3 grid; coords_list[i] is (n_samples, 2)."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, (coords, value) in enumerate(zip(coords_list, param_values)):
        ax = axes[i]
        ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.5, c="steelblue")
        ax.set_title(f"{param_name}={value}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title_prefix, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


async def run_umap_2d_async(executor: ProcessPoolExecutor, params: tuple) -> np.ndarray:
    """Run _umap_2d_worker in the process pool (avoids Numba concurrent-access in threads)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _umap_2d_worker, params)


async def run_n_neighbors_experiment_async(
    executor: ProcessPoolExecutor,
    embeddings: np.ndarray,
) -> None:
    """Run 9 UMAP fits (one per n_neighbors) concurrently in separate processes; when all complete, build and save 3x3 scatter grid."""
    print("\n--- n_neighbors experiment (9 runs concurrently) ---")
    tasks = [
        run_umap_2d_async(executor, (embeddings, n, FIXED_MIN_DIST, 42))
        for n in N_NEIGHBORS_VALUES
    ]
    coords_list = await asyncio.gather(*tasks)
    plot_3x3_scattergrid(
        list(coords_list),
        N_NEIGHBORS_VALUES,
        param_name="n_neighbors",
        title_prefix=f"UMAP 2D (min_dist={FIXED_MIN_DIST})",
        out_path=PLOTS_NEIGHBOURS_DIR / "finetune_umap_n_neighbors.png",
    )


async def run_min_dist_experiment_async(
    executor: ProcessPoolExecutor,
    embeddings: np.ndarray,
) -> None:
    """Run 9 UMAP fits (one per min_dist) concurrently in separate processes; when all complete, build and save 3x3 scatter grid."""
    print("\n--- min_dist experiment (9 runs concurrently) ---")
    tasks = [
        run_umap_2d_async(executor, (embeddings, FIXED_N_NEIGHBORS, d, 42))
        for d in MIN_DIST_VALUES
    ]
    coords_list = await asyncio.gather(*tasks)
    plot_3x3_scattergrid(
        list(coords_list),
        MIN_DIST_VALUES,
        param_name="min_dist",
        title_prefix=f"UMAP 2D (n_neighbors={FIXED_N_NEIGHBORS})",
        out_path=PLOTS_DIST_DIR / "finetune_umap_min_dist.png",
    )


async def main_async(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    max_docs: int = DEFAULT_MAX_DOCS,
    run_neighbours: bool = True,
    run_dist: bool = True,
    max_workers: int = 9,
) -> None:
    documents, embeddings = load_documents_and_embeddings(
        data_dir=data_dir,
        text_column=text_column,
        min_length=min_length,
        dedupe=dedupe,
        max_docs=max_docs,
    )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if run_neighbours:
            await run_n_neighbors_experiment_async(executor, embeddings)
        if run_dist:
            await run_min_dist_experiment_async(executor, embeddings)
    print("Done.")


def main(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    max_docs: int = DEFAULT_MAX_DOCS,
    run_neighbours: bool = True,
    run_dist: bool = True,
    max_workers: int = 9,
) -> None:
    asyncio.run(
        main_async(
            data_dir=data_dir,
            text_column=text_column,
            min_length=min_length,
            dedupe=dedupe,
            max_docs=max_docs,
            run_neighbours=run_neighbours,
            run_dist=run_dist,
            max_workers=max_workers,
        )
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune UMAP: n_neighbors and min_dist visualisation")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Covid-19 Twitter CSV directory")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Document text column")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH, help="Min document length")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="Max documents to use (default: 20000)")
    parser.add_argument("--neighbours-only", action="store_true", help="Run only n_neighbors experiment")
    parser.add_argument("--dist-only", action="store_true", help="Run only min_dist experiment")
    parser.add_argument("--max-workers", type=int, default=9, help="Concurrent UMAP runs per sweep (default: 9)")
    args = parser.parse_args()

    run_neighbours = not args.dist_only
    run_dist = not args.neighbours_only
    if args.neighbours_only:
        run_dist = False
    if args.dist_only:
        run_neighbours = False

    main(
        data_dir=args.data_dir,
        text_column=args.text_column,
        min_length=args.min_length,
        dedupe=not args.no_dedupe,
        max_docs=args.max_docs,
        run_neighbours=run_neighbours,
        run_dist=run_dist,
        max_workers=args.max_workers,
    )
