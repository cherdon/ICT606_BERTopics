"""
Dynamic Topic Modeling with BERTopic for COVID-19 Twitter data.

Same pipeline as detailed_bertopic.py, plus:
  - Topics over time: which topics appear when and how they change (word emphasis, frequency).
  - Requires a timestamp column (e.g. created_at) in the data so documents can be binned by time.

Use case: View what topics have been present and how they evolve over time (e.g. pandemic phases).
"""

from pathlib import Path
import sys

# Project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

from utils import prepare_for_bertopic


# ---------------------------------------------------------------------------
# Paths and data config
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "covid19_twitter_dataset"
DEFAULT_TEXT_COLUMN = "clean_tweet"
DEFAULT_TIMESTAMP_COLUMN = "created_at"  # column in metadata with document timestamps
DEFAULT_MIN_LENGTH = 10
DEFAULT_DEDUPE = True
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


# ---------------------------------------------------------------------------
# Dynamic topic modeling: time binning and tuning
# ---------------------------------------------------------------------------
# nr_bins: number of time bins for topics_over_time. Fewer = coarser view; more = finer but slower.
#   BERTopic recommends <100 unique timestamps. Use nr_bins=10–30 for COVID timeline.
# TODO (dynamic): Experiment with nr_bins (e.g. 12 for monthly, 20–30 for bi-weekly).
# datetime_format: str or None. If None, pandas infers (e.g. ISO). Set if your created_at format fails.
#   e.g. "%Y-%m-%d %H:%M:%S" or "%a %b %d %H:%M:%S %z %Y" for Twitter-style.
#
NR_BINS = 20
DATETIME_FORMAT = None  # auto-detect; set if parsing fails

# evolution_tuning: average c-TF-IDF at t with t-1 so topic words evolve smoothly.
# global_tuning: average time-bin c-TF-IDF with global c-TF-IDF so labels stay interpretable.
# TODO (dynamic): Try evolution_tuning=False for sharper per-period wording; global_tuning=False for pure local.
#
EVOLUTION_TUNING = True
GLOBAL_TUNING = True


# ---------------------------------------------------------------------------
# Pipeline components (same as detailed_bertopic.py)
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    low_memory=False,
    random_state=42,
)
hdbscan_model = HDBSCAN(
    min_cluster_size=10,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
    min_samples=1,
)
vectorizer_model = CountVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
)
ctfidf_model = ClassTfidfTransformer(
    bm25_weighting=True,
    reduce_frequent_words=True,
)
representation_model = KeyBERTInspired(
    top_n_words=10,
    nr_repr_docs=5,
    nr_samples=500,
    nr_candidate_words=100,
    random_state=42,
)
NR_TOPICS = None
ASSIGN_OUTLIERS = False


def get_model() -> BERTopic:
    """Build BERTopic with the same pipeline as detailed_bertopic."""
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=10,
        min_topic_size=10,
        nr_topics=NR_TOPICS,
        verbose=True,
        calculate_probabilities=False,
    )


def _get_timestamps_and_mask(
    metadata: pd.DataFrame | None,
    timestamp_column: str,
    n_docs: int,
):
    """
    Get timestamps aligned to documents and a mask of valid rows.
    Returns (timestamps_list, valid_mask) or (None, None) if column missing.
    valid_mask is True where timestamp is present so we can subset docs/topics/timestamps.
    """
    if metadata is None or metadata.empty or timestamp_column not in metadata.columns:
        return None, None
    if len(metadata) != n_docs:
        return None, None
    ts = pd.to_datetime(metadata[timestamp_column], errors="coerce")
    valid = ts.notna()
    if not valid.any():
        return None, None
    return ts.astype(str).tolist(), valid.values


def run(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    nr_bins: int = NR_BINS,
    datetime_format: str | None = DATETIME_FORMAT,
    evolution_tuning: bool = EVOLUTION_TUNING,
    global_tuning: bool = GLOBAL_TUNING,
    save_viz: bool = True,
) -> None:
    """Run BERTopic, then dynamic topic modeling over time; save model, assignments, and over-time table/viz."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing tweets...")
    documents, metadata = prepare_for_bertopic(
        data_dir=data_dir,
        text_column=text_column,
        lang="en",
        min_length=min_length,
        dedupe=dedupe,
    )
    print(f"Documents for BERTopic: {len(documents)}")

    if not documents:
        print("No documents after preprocessing. Exiting.")
        return

    # Timestamps must align 1:1 with documents (metadata is aligned from prepare_for_bertopic)
    timestamps_list, valid_mask = _get_timestamps_and_mask(
        metadata, timestamp_column, len(documents)
    )
    can_do_over_time = timestamps_list is not None and valid_mask is not None

    print("Fitting BERTopic (embeddings → UMAP → HDBSCAN → vectorizer → c-TF-IDF → representation)...")
    model = get_model()
    topics, probs = model.fit_transform(documents)

    if ASSIGN_OUTLIERS and -1 in model.topic_sizes_:
        print("Reducing outliers (assigning to nearest topic)...")
        topics = model.reduce_outliers(documents, topics, strategy="embeddings")

    # Save model
    model_path = output_dir / "dynamic_bertopic_model"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Save topic assignments
    out_meta = metadata.copy() if metadata is not None else pd.DataFrame()
    out_meta["topic_id"] = topics
    out_meta["document"] = documents
    results_path = output_dir / "dynamic_topic_assignments.csv"
    out_meta.to_csv(results_path, index=False)
    print(f"Topic assignments saved to {results_path}")

    n_topics = len([t for t in model.topic_sizes_ if t != -1])
    n_outliers = model.topic_sizes_.get(-1, 0)
    print(f"Topics: {n_topics}, Outliers (-1): {n_outliers}")

    # --- Dynamic: topics over time ---
    if can_do_over_time:
        # Subset to documents with valid timestamps so lengths match
        docs_sub = [d for d, v in zip(documents, valid_mask) if v]
        topics_sub = [t for t, v in zip(topics, valid_mask) if v]
        ts_sub = [t for t, v in zip(timestamps_list, valid_mask) if v]
        print(f"Computing topics over time for {len(docs_sub)} documents with valid timestamps...")
        topics_over_time_df = model.topics_over_time(
            docs_sub,
            ts_sub,
            topics=topics_sub,
            nr_bins=nr_bins,
            datetime_format=datetime_format,
            evolution_tuning=evolution_tuning,
            global_tuning=global_tuning,
        )
        over_time_path = output_dir / "dynamic_topics_over_time.csv"
        topics_over_time_df.to_csv(over_time_path, index=False)
        print(f"Topics over time saved to {over_time_path}")

        if save_viz:
            try:
                fig = model.visualize_topics_over_time(
                    topics_over_time_df,
                    top_n_topics=min(15, n_topics) if n_topics else 15,
                    title="<b>COVID-19 Twitter topics over time</b>",
                )
                viz_path = output_dir / "dynamic_topics_over_time.html"
                fig.write_html(str(viz_path))
                print(f"Topics-over-time visualization saved to {viz_path}")
            except Exception as e:
                print(f"Could not save topics-over-time visualization: {e}")
    else:
        print("Skipping topics over time and visualization (no timestamps column or no valid timestamps).")

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run BERTopic + dynamic topic modeling (topics over time) on COVID-19 tweets"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Covid-19 Twitter CSV directory")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN, help="Document text column")
    parser.add_argument(
        "--timestamp-column",
        default=DEFAULT_TIMESTAMP_COLUMN,
        help="Metadata column with document timestamps (e.g. created_at)",
    )
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH, help="Min document length")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable deduplication")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--nr-bins",
        type=int,
        default=NR_BINS,
        help="Number of time bins for topics over time (recommended <100)",
    )
    parser.add_argument(
        "--no-evolution-tuning",
        action="store_true",
        help="Disable evolution_tuning in topics_over_time",
    )
    parser.add_argument(
        "--no-global-tuning",
        action="store_true",
        help="Disable global_tuning in topics_over_time",
    )
    parser.add_argument("--no-viz", action="store_true", help="Do not save topics-over-time HTML visualization")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        text_column=args.text_column,
        timestamp_column=args.timestamp_column,
        min_length=args.min_length,
        dedupe=not args.no_dedupe,
        output_dir=args.output_dir,
        nr_bins=args.nr_bins,
        datetime_format=DATETIME_FORMAT,
        evolution_tuning=not args.no_evolution_tuning,
        global_tuning=not args.no_global_tuning,
        save_viz=not args.no_viz,
    )
