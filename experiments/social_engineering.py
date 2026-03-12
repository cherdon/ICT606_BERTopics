"""
Social engineering analysis: topic rankings by prevalence, retweet impact,
favourite impact, and log-transformed engagement (to reduce skew from viral tweets).

Copies the full run_bertopic pipeline, then aggregates by topic using
retweet_count and favorite_count from the data. Writes rankings to
experiments/results/social_engineering.txt.
"""

from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
if _root not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from utils.bertopic_pipeline import run_pipeline, build_bertopic_pipeline
from utils.bertopic_pipeline import (
    get_umap_model,
    get_hdbscan_model,
    get_vectorizer_model,
    get_ctfidf_model,
)
from utils.constants import COVID_STOPWORDS, COLUMNS_TO_DROP
from utils.metrics import topic_coherence_gensim, get_topic_overview

RESULTS_DIR = _root / "experiments" / "results"
BERTOPIC_RESULTS_FILE = RESULTS_DIR / "bertopic_results.txt"
SOCIAL_ENGINEERING_FILE = RESULTS_DIR / "social_engineering.txt"
TOP_N_TOPICS_FOR_RESULTS = 20

# Column names in the data (as in CSV)
RETWEET_COL = "retweet_count"
FAVOURITE_COL = "favorite_count"


def write_bertopic_results(
    model,
    documents: list,
    topics: list,
    coherence: float,
    out_path: Path,
) -> None:
    """Write coherence, topic count, and top topics (with words) to a text file."""
    overview, nr_topics = get_topic_overview(model, top_n=10, exclude_outliers=True)
    info = model.get_topic_info()
    info_no_out = info[info["Topic"] != -1].sort_values("Count", ascending=False)
    top_topic_ids = info_no_out.head(TOP_N_TOPICS_FOR_RESULTS)["Topic"].tolist()

    lines = [
        "BERTopic run results (finalized pipeline)",
        "=" * 60,
        f"Coherence (c_v): {coherence:.4f}",
        f"Number of topics (excl. outliers): {nr_topics}",
        f"Total documents: {len(documents)}",
        "",
        "Topic sizes (top topics by count):",
        info_no_out.head(TOP_N_TOPICS_FOR_RESULTS).to_string(),
        "",
        "Top topics (by size) – top 10 words each:",
    ]
    for tid in top_topic_ids:
        if tid in overview and overview[tid]:
            words = [w for w, _ in overview[tid]]
            lines.append(f"  Topic {tid}: {words}")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Results written to {out_path}")


def _safe_series(metadata: pd.DataFrame, col: str) -> pd.Series:
    """Get numeric series from metadata, coerce and fill NaN with 0."""
    if col not in metadata.columns:
        return pd.Series(0, index=metadata.index)
    return pd.to_numeric(metadata[col], errors="coerce").fillna(0)


def write_social_engineering_results(
    model,
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Write topic rankings: prevalence, retweet impact, favourite impact,
    and log-transformed retweets/favourites (to reduce skew).
    """
    overview, _ = get_topic_overview(model, top_n=10, exclude_outliers=True)

    # 1. Topic prevalence – count tweets per topic (what people talk about the most)
    topic_counts = df.groupby("topic").size().sort_values(ascending=False)

    # 2. Topic retweet impact – sum of retweets per topic (which topics spread the most)
    topic_retweets = (
        df.groupby("topic")[RETWEET_COL].sum().sort_values(ascending=False)
    )

    # 3. Topic favourite impact – sum of favourites per topic (which topics people liked the most)
    topic_favourites = (
        df.groupby("topic")[FAVOURITE_COL].sum().sort_values(ascending=False)
    )

    # 4. Log-transformed engagement (reduces skew from viral tweets)
    log_retweets = np.log1p(df[RETWEET_COL])
    log_favourites = np.log1p(df[FAVOURITE_COL])
    topic_log_retweets = (
        df.assign(log_retweets=log_retweets)
        .groupby("topic")["log_retweets"]
        .sum()
        .sort_values(ascending=False)
    )
    topic_log_favourites = (
        df.assign(log_favourites=log_favourites)
        .groupby("topic")["log_favourites"]
        .sum()
        .sort_values(ascending=False)
    )

    def rank_lines(title: str, description: str, series: pd.Series) -> list:
        lines = ["", "=" * 60, title, "=" * 60, description, ""]
        for rank, (topic_id, value) in enumerate(series.items(), start=1):
            words = ""
            if topic_id in overview and overview[topic_id]:
                words = "  " + ", ".join(w for w, _ in overview[topic_id][:5])
            lines.append(f"  {rank}. Topic {topic_id}: {value:,.0f}{words}")
        return lines

    lines = [
        "Social engineering topic rankings",
        "Based on BERTopic assignments and retweet_count / favorite_count.",
        "",
    ]

    lines.extend(
        rank_lines(
            "1. Topic prevalence (tweet count per topic)",
            "What people talk about the most.",
            topic_counts,
        )
    )
    lines.extend(
        rank_lines(
            "2. Topic retweet impact (sum of retweet_count per topic)",
            "Which topics spread the most.",
            topic_retweets,
        )
    )
    lines.extend(
        rank_lines(
            "3. Topic favourite impact (sum of favorite_count per topic)",
            "Which topics people liked the most.",
            topic_favourites,
        )
    )
    lines.extend(
        rank_lines(
            "4a. Topic log-retweet impact (sum of log1p(retweet_count) per topic)",
            "Which topics spread the most (log scale; reduces skew from viral tweets).",
            topic_log_retweets,
        )
    )
    lines.extend(
        rank_lines(
            "4b. Topic log-favourite impact (sum of log1p(favorite_count) per topic)",
            "Which topics people liked the most (log scale; reduces skew).",
            topic_log_favourites,
        )
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Social engineering rankings written to {out_path}")


if __name__ == "__main__":
    from utils import (
        filter_language,
        get_documents,
        load_covid_tweets,
        preprocess_tweets,
        remove_duplicates,
    )

    data_dir = Path(__file__).resolve().parents[1] / "data" / "covid19_twitter_dataset"
    text_column = "original_text"
    processed_column = "processed_text"

    # 1. Load CSV data as DataFrame
    print("Loading CSV data...")
    df = load_covid_tweets(data_dir)
    print(f"Loaded {len(df)} rows.")

    # 2. Keep only English rows and remove duplicates by original_text
    df = filter_language(df, lang="en")
    df = remove_duplicates(df, text_column=text_column)
    print(f"After lang filter and dedupe: {len(df)} rows.")

    # 3. Drop columns not needed for this use case
    existing_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")
    print(f"Dropped columns: {existing_drop}")

    # 4. Run full text preprocessing on the tweet column
    print("Preprocessing tweet text...")
    df = preprocess_tweets(
        df,
        text_column=text_column,
        output_column=processed_column,
    )
    print(f"Preprocessed {len(df)} tweets.")

    # 5. Extract documents and metadata for BERTopic (metadata keeps retweet_count, favorite_count)
    documents, metadata = get_documents(
        df,
        text_column=processed_column,
        min_length=10,
        return_metadata=True,
    )
    print(f"Documents for BERTopic: {len(documents)}")

    if not documents:
        print("No documents. Exiting.")
        sys.exit(1)

    # ---------------------------------------------------------------------------
    # CUSTOM PIPELINE (same as run_bertopic)
    # ---------------------------------------------------------------------------
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

    print("Building and fitting BERTopic pipeline...")
    model, topics, probs = run_pipeline(documents, build_fn=pipeline)
    n_topics = len(set(topics) - {-1})
    print(f"Found {n_topics} topics (+ outliers -1).")
    print("Topic info:", model.get_topic_info().head(10).to_string())

    # Coherence and BERTopic results file
    print("Computing topic coherence (c_v)...")
    coherence = topic_coherence_gensim(
        model, documents, top_n=10, coherence="c_v"
    )
    print(f"Coherence (c_v): {coherence:.4f}")
    write_bertopic_results(
        model, documents, topics, coherence, BERTOPIC_RESULTS_FILE
    )

    # ---------------------------------------------------------------------------
    # Social engineering: build df with topic + engagement columns
    # ---------------------------------------------------------------------------
    out_df = metadata.copy()
    out_df["topic"] = topics
    # Ensure numeric and handle missing column names
    out_df[RETWEET_COL] = _safe_series(metadata, RETWEET_COL)
    out_df[FAVOURITE_COL] = _safe_series(metadata, FAVOURITE_COL)

    write_social_engineering_results(model, out_df, SOCIAL_ENGINEERING_FILE)
