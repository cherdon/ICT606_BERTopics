from pathlib import Path
import sys

# Ensure project root is on path when run as script (e.g. from experiments/)
_root = Path(__file__).resolve().parents[1]
if _root not in sys.path:
    sys.path.insert(0, str(_root))

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
TOP_N_TOPICS_FOR_RESULTS = 20


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

    # 4. Run full text preprocessing on the tweet column (applies preprocess_tweet to each row)
    print("Preprocessing tweet text...")
    df = preprocess_tweets(
        df,
        text_column=text_column,
        output_column=processed_column,
    )
    print(f"Preprocessed {len(df)} tweets.")

    # 5. Extract documents and metadata for BERTopic (use processed text)
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
    # CUSTOM PIPELINE
    # ---------------------------------------------------------------------------
    pipeline = build_bertopic_pipeline(
        umap_model=get_umap_model(n_neighbors=25, min_dist=0.01),
        hdbscan_model=get_hdbscan_model(min_cluster_size=25, min_samples=5),
        vectorizer_model=get_vectorizer_model(
            ngram_range=(1,2),
            min_df=5,
            max_df=0.85,
            extra_stop_words=COVID_STOPWORDS
        ),
        ctfidf_model=get_ctfidf_model(bm25_weighting=False, reduce_frequent_words=False),
    )

    print("Building and fitting BERTopic pipeline...")
    model, topics, probs = run_pipeline(documents, build_fn=pipeline)
    n_topics = len(set(topics) - {-1})
    print(f"Found {n_topics} topics (+ outliers -1).")
    print("Topic info:", model.get_topic_info().head(10).to_string())

    # Coherence and results file
    print("Computing topic coherence (c_v)...")
    coherence = topic_coherence_gensim(model, documents, top_n=10, coherence="c_v")
    print(f"Coherence (c_v): {coherence:.4f}")
    write_bertopic_results(
        model, documents, topics, coherence, BERTOPIC_RESULTS_FILE
    )