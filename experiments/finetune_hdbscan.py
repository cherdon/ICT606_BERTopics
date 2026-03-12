"""
Finetune HDBSCAN by visualising 2D scatter (UMAP layout, colored by cluster) and
condensed trees for different min_samples and min_cluster_size.

Uses the same preprocessing as finetune_umap (prepare_documents_for_finetuning).
Loads documents, embeds once, runs UMAP 2D once, then runs HDBSCAN with 9 values
of min_samples (fixed min_cluster_size) and 9 values of min_cluster_size (fixed
min_samples). Produces 4 plots: min_samples scatter 3x3, min_samples condensed tree 3x3,
min_cluster_size scatter 3x3, min_cluster_size condensed tree 3x3.
HDBSCAN runs execute concurrently via ProcessPoolExecutor.
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

from experiments.finetuning_workers import _umap_2d_worker, _hdbscan_2d_worker
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

# UMAP 2D (fixed for all HDBSCAN runs)
UMAP_2D_N_NEIGHBORS = 25
UMAP_2D_MIN_DIST = 0.0
UMAP_2D_RANDOM_STATE = 42

# HDBSCAN: 9 values each; when sweeping one param, fix the other
MIN_SAMPLES_VALUES = [1, 2, 3, 5, 10, 15, 20, 25, 30]
MIN_CLUSTER_SIZE_VALUES = [5, 10, 15, 20, 25, 50, 75, 100, 150]
FIXED_MIN_CLUSTER_SIZE_FOR_SAMPLES_SWEEP = 15
FIXED_MIN_SAMPLES_FOR_CLUSTER_SIZE_SWEEP = 5
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

PLOTS_DIR = _root / "experiments" / "plots" / "hdbscan"


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


def _hdbscan_params_tuple(
    embeddings_2d: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple:
    """Build params tuple for _hdbscan_2d_worker."""
    return (
        embeddings_2d,
        min_cluster_size,
        min_samples,
        HDBSCAN_METRIC,
        HDBSCAN_CLUSTER_SELECTION_METHOD,
    )


def plot_3x3_scatter_clusters(
    coords_2d: np.ndarray,
    clusterers: list,
    param_values: list,
    param_name: str,
    title_prefix: str,
    out_path: Path,
) -> None:
    """Plot 9 scatter plots: same 2D coords, each subplot colored by that run's labels."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    for i, (clusterer, value) in enumerate(zip(clusterers, param_values)):
        ax = axes[i]
        labels = clusterer.labels_
        ax.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=labels,
            s=1,
            alpha=0.6,
            cmap="tab20",
        )
        ax.set_title(f"{param_name}={value}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title_prefix, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_3x3_condensed_trees(
    clusterers: list,
    param_values: list,
    param_name: str,
    title_prefix: str,
    out_path: Path,
) -> None:
    """Plot 9 condensed tree plots in a 3x3 grid; each clusterer.condensed_tree_.plot(ax=ax)."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    axes = axes.flatten()
    for i, (clusterer, value) in enumerate(zip(clusterers, param_values)):
        ax = axes[i]
        clusterer.condensed_tree_.plot(ax=ax)
        ax.set_title(f"{param_name}={value}")
    fig.suptitle(title_prefix, fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


async def run_hdbscan_async(executor: ProcessPoolExecutor, params: tuple):
    """Run _hdbscan_2d_worker in the process pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _hdbscan_2d_worker, params)


async def run_min_samples_experiment_async(
    executor: ProcessPoolExecutor,
    embeddings_2d: np.ndarray,
) -> None:
    """Run 9 HDBSCAN fits (varying min_samples), then save scatter 3x3 and condensed tree 3x3."""
    print("\n--- min_samples experiment (9 runs concurrently) ---")
    tasks = [
        run_hdbscan_async(
            executor,
            _hdbscan_params_tuple(
                embeddings_2d,
                FIXED_MIN_CLUSTER_SIZE_FOR_SAMPLES_SWEEP,
                min_samples,
            ),
        )
        for min_samples in MIN_SAMPLES_VALUES
    ]
    clusterers = list(await asyncio.gather(*tasks))

    plot_3x3_scatter_clusters(
        embeddings_2d,
        clusterers,
        MIN_SAMPLES_VALUES,
        param_name="min_samples",
        title_prefix=f"HDBSCAN 2D scatter (min_cluster_size={FIXED_MIN_CLUSTER_SIZE_FOR_SAMPLES_SWEEP})",
        out_path=PLOTS_DIR / "finetune_hdbscan_min_samples_scatter.png",
    )
    plot_3x3_condensed_trees(
        clusterers,
        MIN_SAMPLES_VALUES,
        param_name="min_samples",
        title_prefix=f"HDBSCAN condensed tree (min_cluster_size={FIXED_MIN_CLUSTER_SIZE_FOR_SAMPLES_SWEEP})",
        out_path=PLOTS_DIR / "finetune_hdbscan_min_samples_condensed_tree.png",
    )


async def run_min_cluster_size_experiment_async(
    executor: ProcessPoolExecutor,
    embeddings_2d: np.ndarray,
) -> None:
    """Run 9 HDBSCAN fits (varying min_cluster_size), then save scatter 3x3 and condensed tree 3x3."""
    print("\n--- min_cluster_size experiment (9 runs concurrently) ---")
    tasks = [
        run_hdbscan_async(
            executor,
            _hdbscan_params_tuple(
                embeddings_2d,
                min_cluster_size,
                FIXED_MIN_SAMPLES_FOR_CLUSTER_SIZE_SWEEP,
            ),
        )
        for min_cluster_size in MIN_CLUSTER_SIZE_VALUES
    ]
    clusterers = list(await asyncio.gather(*tasks))

    plot_3x3_scatter_clusters(
        embeddings_2d,
        clusterers,
        MIN_CLUSTER_SIZE_VALUES,
        param_name="min_cluster_size",
        title_prefix=f"HDBSCAN 2D scatter (min_samples={FIXED_MIN_SAMPLES_FOR_CLUSTER_SIZE_SWEEP})",
        out_path=PLOTS_DIR / "finetune_hdbscan_min_cluster_size_scatter.png",
    )
    plot_3x3_condensed_trees(
        clusterers,
        MIN_CLUSTER_SIZE_VALUES,
        param_name="min_cluster_size",
        title_prefix=f"HDBSCAN condensed tree (min_samples={FIXED_MIN_SAMPLES_FOR_CLUSTER_SIZE_SWEEP})",
        out_path=PLOTS_DIR / "finetune_hdbscan_min_cluster_size_condensed_tree.png",
    )


async def main_async(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    max_docs: int = DEFAULT_MAX_DOCS,
    run_min_samples: bool = True,
    run_min_cluster_size: bool = True,
    max_workers: int = 9,
) -> None:
    documents, embeddings = load_documents_and_embeddings(
        data_dir=data_dir,
        text_column=text_column,
        min_length=min_length,
        dedupe=dedupe,
        max_docs=max_docs,
    )
    print("Running UMAP 2D (single run for all HDBSCAN experiments)...")
    umap_2d_params = (
        embeddings,
        UMAP_2D_N_NEIGHBORS,
        UMAP_2D_MIN_DIST,
        UMAP_2D_RANDOM_STATE,
    )
    with ProcessPoolExecutor(max_workers=1) as exec_umap:
        loop = asyncio.get_event_loop()
        embeddings_2d = await loop.run_in_executor(exec_umap, _umap_2d_worker, umap_2d_params)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if run_min_samples:
            await run_min_samples_experiment_async(executor, embeddings_2d)
        if run_min_cluster_size:
            await run_min_cluster_size_experiment_async(executor, embeddings_2d)
    print("Done.")


def main(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    max_docs: int = DEFAULT_MAX_DOCS,
    run_min_samples: bool = True,
    run_min_cluster_size: bool = True,
    max_workers: int = 9,
) -> None:
    asyncio.run(
        main_async(
            data_dir=data_dir,
            text_column=text_column,
            min_length=min_length,
            dedupe=dedupe,
            max_docs=max_docs,
            run_min_samples=run_min_samples,
            run_min_cluster_size=run_min_cluster_size,
            max_workers=max_workers,
        )
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Finetune HDBSCAN: min_samples and min_cluster_size visualisation (scatter + condensed tree)"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Covid-19 Twitter CSV directory")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Raw text column")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH, help="Min document length")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="Max documents (default: 20000)")
    parser.add_argument("--samples-only", action="store_true", help="Run only min_samples experiment")
    parser.add_argument("--cluster-size-only", action="store_true", help="Run only min_cluster_size experiment")
    parser.add_argument("--max-workers", type=int, default=9, help="Concurrent HDBSCAN runs per sweep (default: 9)")
    args = parser.parse_args()

    run_min_samples = not args.cluster_size_only
    run_min_cluster_size = not args.samples_only
    if args.samples_only:
        run_min_cluster_size = False
    if args.cluster_size_only:
        run_min_samples = False

    main(
        data_dir=args.data_dir,
        text_column=args.text_column,
        min_length=args.min_length,
        dedupe=not args.no_dedupe,
        max_docs=args.max_docs,
        run_min_samples=run_min_samples,
        run_min_cluster_size=run_min_cluster_size,
        max_workers=args.max_workers,
    )
