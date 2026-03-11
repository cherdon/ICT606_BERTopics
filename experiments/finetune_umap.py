"""
Finetune UMAP by visualising 2D projections for different n_neighbors and min_dist.

Loads documents, embeds them once, then runs UMAP with varying n_neighbors (or min_dist)
and saves 3x3 scatterplot grids to compare how points are grouped.
"""

from pathlib import Path
import sys

# Project root for imports
_root = Path(__file__).resolve().parents[1]
if _root not in sys.path:
    sys.path.insert(0, str(_root))

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from utils.bertopic_pipeline import get_embedding_model
from utils import prepare_for_bertopic

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = _root / "data" / "covid19_twitter_dataset"
DEFAULT_TEXT_COLUMN = "clean_tweet"
DEFAULT_MIN_LENGTH = 10
DEFAULT_DEDUPE = True

# n_neighbors values to sweep (9 for 3x3 grid)
N_NEIGHBORS_VALUES = [2, 5, 10, 15, 20, 30, 50, 100, 150]

# min_dist values to sweep (9 for 3x3 grid)
MIN_DIST_VALUES = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]

# Fixed UMAP params when sweeping the other
FIXED_MIN_DIST = 0.0
FIXED_N_NEIGHBORS = 15

PLOTS_NEIGHBOURS_DIR = _root / "experiments" / "plots"
PLOTS_DIST_DIR = _root / "experiments" / "plots"


def load_documents_and_embeddings(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
):
    """Load documents and compute embeddings once (shared across all UMAP runs)."""
    print("Loading and preprocessing tweets...")
    documents, _ = prepare_for_bertopic(
        data_dir=data_dir,
        text_column=text_column,
        lang="en",
        min_length=min_length,
        dedupe=dedupe,
    )
    if not documents:
        raise SystemExit("No documents after preprocessing. Exiting.")
    print(f"Documents: {len(documents)}")

    print("Computing embeddings...")
    embedding_model = get_embedding_model()
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    return documents, np.array(embeddings)


def run_umap_2d(embeddings: np.ndarray, n_neighbors: int, min_dist: float, random_state: int = 42) -> np.ndarray:
    """Run UMAP with n_components=2 for scatter plot; returns (n_samples, 2)."""
    umap = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    return umap.fit_transform(embeddings)


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


def run_n_neighbors_experiment(embeddings: np.ndarray) -> None:
    """Sweep n_neighbors; fixed min_dist. Save 3x3 scatter grid."""
    print("\n--- n_neighbors experiment ---")
    coords_list = []
    for n in N_NEIGHBORS_VALUES:
        print(f"  UMAP n_neighbors={n} ...")
        coords = run_umap_2d(embeddings, n_neighbors=n, min_dist=FIXED_MIN_DIST)
        coords_list.append(coords)

    plot_3x3_scattergrid(
        coords_list,
        N_NEIGHBORS_VALUES,
        param_name="n_neighbors",
        title_prefix=f"UMAP 2D (min_dist={FIXED_MIN_DIST})",
        out_path=PLOTS_NEIGHBOURS_DIR / "finetune_umap_n_neighbors.png",
    )


def run_min_dist_experiment(embeddings: np.ndarray) -> None:
    """Sweep min_dist; fixed n_neighbors. Save 3x3 scatter grid."""
    print("\n--- min_dist experiment ---")
    coords_list = []
    for d in MIN_DIST_VALUES:
        print(f"  UMAP min_dist={d} ...")
        coords = run_umap_2d(embeddings, n_neighbors=FIXED_N_NEIGHBORS, min_dist=d)
        coords_list.append(coords)

    plot_3x3_scattergrid(
        coords_list,
        MIN_DIST_VALUES,
        param_name="min_dist",
        title_prefix=f"UMAP 2D (n_neighbors={FIXED_N_NEIGHBORS})",
        out_path=PLOTS_DIST_DIR / "finetune_umap_min_dist.png",
    )


def main(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    run_neighbours: bool = True,
    run_dist: bool = True,
) -> None:
    documents, embeddings = load_documents_and_embeddings(
        data_dir=data_dir,
        text_column=text_column,
        min_length=min_length,
        dedupe=dedupe,
    )
    if run_neighbours:
        run_n_neighbors_experiment(embeddings)
    if run_dist:
        run_min_dist_experiment(embeddings)
    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune UMAP: n_neighbors and min_dist visualisation")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Covid-19 Twitter CSV directory")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Document text column")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH, help="Min document length")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    parser.add_argument("--neighbours-only", action="store_true", help="Run only n_neighbors experiment")
    parser.add_argument("--dist-only", action="store_true", help="Run only min_dist experiment")
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
        run_neighbours=run_neighbours,
        run_dist=run_dist,
    )
