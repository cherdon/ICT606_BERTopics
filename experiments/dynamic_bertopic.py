"""
Dynamic topic modeling with BERTopic for COVID-19 Twitter data.

Uses the same preprocessing and finalized pipeline as run_bertopic.py, on the
entire dataset. Splits data by month (April–June 2020, August–October 2020,
April–June 2021), fits one BERTopic model on all documents, assigns topics,
then measures how topic frequency changes across months via topics_over_time
and saves a final graph (HTML).
"""

from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
if _root not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

from utils.bertopic_pipeline import (
    run_pipeline,
    build_bertopic_pipeline,
    get_umap_model,
    get_hdbscan_model,
    get_vectorizer_model,
    get_ctfidf_model,
)
from utils.constants import COVID_STOPWORDS, COLUMNS_TO_DROP
from utils import (
    filter_language,
    get_documents,
    load_covid_tweets,
    preprocess_tweets,
    remove_duplicates,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = _root / "data" / "covid19_twitter_dataset"
DEFAULT_TEXT_COLUMN = "original_text"
DEFAULT_PROCESSED_COLUMN = "processed_text"
DEFAULT_TIMESTAMP_COLUMN = "created_at"
DEFAULT_MIN_LENGTH = 10
RESULTS_DIR = _root / "experiments" / "results"
TOPICS_OVER_TIME_CSV = RESULTS_DIR / "dynamic_topics_over_time.csv"
TOPICS_OVER_TIME_HTML = RESULTS_DIR / "dynamic_topics_over_time.html"
TOP_N_TOPICS_VIZ = 15


def main(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    text_column: str = DEFAULT_TEXT_COLUMN,
    processed_column: str = DEFAULT_PROCESSED_COLUMN,
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    save_csv: bool = True,
    save_html: bool = True,
) -> None:
    data_dir = Path(data_dir)
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full dataset (no sampling)
    print("Loading full CSV data...")
    df = load_covid_tweets(data_dir)
    print(f"Loaded {len(df)} rows.")

    # 2. English only, dedupe by text
    df = filter_language(df, lang="en")
    df = remove_duplicates(df, text_column=text_column)
    print(f"After lang filter and dedupe: {len(df)} rows.")

    # 3. Drop columns not needed
    existing_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")
    print(f"Dropped columns: {existing_drop}")

    # 4. Preprocess tweets (same as run_bertopic)
    print("Preprocessing tweet text...")
    df = preprocess_tweets(
        df,
        text_column=text_column,
        output_column=processed_column,
    )
    print(f"Preprocessed {len(df)} tweets.")

    # 5. Add month_year for grouping by month (YYYY-MM); keep before get_documents so metadata is aligned
    if timestamp_column not in df.columns:
        print(f"Missing column '{timestamp_column}'. Cannot compute topics over time. Exiting.")
        sys.exit(1)
    df["month_year"] = (
        pd.to_datetime(df[timestamp_column], errors="coerce")
        .dt.to_period("M")
        .astype(str)
    )

    # 6. Extract documents and metadata (aligned; metadata includes created_at and month_year)
    documents, metadata = get_documents(
        df,
        text_column=processed_column,
        min_length=min_length,
        return_metadata=True,
    )
    print(f"Documents for BERTopic: {len(documents)}")

    if not documents:
        print("No documents. Exiting.")
        sys.exit(1)

    # 7. Finalized pipeline (same as run_bertopic)
    pipeline = build_bertopic_pipeline(
        umap_model=get_umap_model(n_neighbors=25, min_dist=0.01),
        hdbscan_model=get_hdbscan_model(min_cluster_size=25, min_samples=5),
        vectorizer_model=get_vectorizer_model(
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.85,
            extra_stop_words=COVID_STOPWORDS,
        ),
        ctfidf_model=get_ctfidf_model(
            bm25_weighting=False,
            reduce_frequent_words=False,
        ),
    )

    # 8. Fit one model on all documents and assign topics
    print("Fitting BERTopic on all documents...")
    model, topics, probs = run_pipeline(documents, build_fn=pipeline)
    n_topics = len([t for t in set(topics) if t != -1])
    print(f"Found {n_topics} topics (+ outliers -1).")
    print("Topic info (head):", model.get_topic_info().head(10).to_string())

    # 9. Restrict to documents with valid month_year for topics_over_time
    if "month_year" not in metadata.columns:
        print("No month_year in metadata. Skipping topics over time.")
        return
    valid = metadata["month_year"].notna()
    if not valid.any():
        print("No valid month_year. Skipping topics over time.")
        return

    docs_sub = [d for d, v in zip(documents, valid) if v]
    topics_sub = [t for t, v in zip(topics, valid) if v]
    timestamps_sub = metadata.loc[valid, "month_year"].tolist()

    n_unique_months = len(pd.Series(timestamps_sub).unique())
    print(f"Computing topics over time for {len(docs_sub)} docs across {n_unique_months} months...")

    # 10. Topic frequency over time (by month)
    topics_over_time_df = model.topics_over_time(
        docs=docs_sub,
        timestamps=timestamps_sub,
        topics=topics_sub,
        nr_bins=n_unique_months,
        evolution_tuning=True,
        global_tuning=True,
    )

    if save_csv:
        csv_path = results_dir / "dynamic_topics_over_time.csv"
        topics_over_time_df.to_csv(csv_path, index=False)
        print(f"Topics over time table saved to {csv_path}")

    # 11. Final graph: how topic frequency changes across months
    if save_html:
        try:
            fig = model.visualize_topics_over_time(
                topics_over_time_df,
                top_n_topics=min(TOP_N_TOPICS_VIZ, n_topics) if n_topics else TOP_N_TOPICS_VIZ,
                title="<b>COVID-19 Twitter topics over time (by month)</b>",
            )
            html_path = results_dir / "dynamic_topics_over_time.html"
            fig.write_html(str(html_path))
            print(f"Topics-over-time graph saved to {html_path}")
        except Exception as e:
            print(f"Could not save topics-over-time visualization: {e}")

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run BERTopic on full COVID-19 tweets and plot topics over time by month"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Covid-19 Twitter CSV directory",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Raw text column name",
    )
    parser.add_argument(
        "--timestamp-column",
        default=DEFAULT_TIMESTAMP_COLUMN,
        help="Column with tweet timestamps (e.g. created_at)",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help="Min document length",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not save topics_over_time CSV",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Do not save topics_over_time HTML graph",
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        text_column=args.text_column,
        timestamp_column=args.timestamp_column,
        min_length=args.min_length,
        save_csv=not args.no_csv,
        save_html=not args.no_html,
    )
