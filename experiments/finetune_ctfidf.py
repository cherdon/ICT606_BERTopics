"""
Finetune ClassTfidfTransformer (c-TF-IDF) by sweeping bm25_weighting and reduce_frequent_words.

Runs all four combinations in separate processes (ProcessPoolExecutor), computes topic coherence
(gensim c_v) and get_topic overview for interpretability. Writes results to
experiments/results/ctfidf.txt.

Uses fixed finetuned pipeline: UMAP, HDBSCAN, and vectorizer (ngram_range=(1,2), min_df=5, max_df=0.85).

Experiments:
- (bm25_weighting, reduce_frequent_words): (False, False), (True, False), (False, True), (True, True)
"""
import os

# Avoid tokenizers parallelism warning when forking (ProcessPoolExecutor)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

_root = Path(__file__).resolve().parents[1]
if _root not in sys.path:
    sys.path.insert(0, str(_root))

from utils.bertopic_pipeline import (
    build_bertopic_pipeline,
    get_vectorizer_model,
    get_umap_model,
    get_hdbscan_model,
    get_ctfidf_model,
    run_pipeline,
)
from utils.metrics import topic_coherence_gensim, get_topic_overview
from utils import prepare_documents_for_finetuning
from utils.constants import COVID_STOPWORDS

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

# Finetuned vectorizer (from vectorizer experiments)
FINETUNED_VECTORIZER_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.85,
    "extra_stop_words": COVID_STOPWORDS,
}

# Finetuned UMAP / HDBSCAN (same as vectorizer pipeline)
FINETUNED_UMAP = get_umap_model(n_neighbors=25, min_dist=0.01)
FINETUNED_HDBSCAN = get_hdbscan_model(min_cluster_size=25, min_samples=5)

# (bm25_weighting, reduce_frequent_words)
CTFIDF_EXPERIMENTS = [
    (False, False),
    (True, False),
    (False, True),
    (True, True),
]

RESULTS_DIR = _root / "experiments" / "results"
RESULTS_FILE = RESULTS_DIR / "ctfidf.txt"

TOP_N_TOPICS_FOR_OVERVIEW = 15


def _run_single_experiment(
    documents: list,
    bm25_weighting: bool,
    reduce_frequent_words: bool,
) -> dict:
    """Run one BERTopic fit with the given c-TF-IDF params; return coherence and topic overview."""
    finetuned_vectorizer = get_vectorizer_model(**FINETUNED_VECTORIZER_PARAMS)
    ctfidf = get_ctfidf_model(
        bm25_weighting=bm25_weighting,
        reduce_frequent_words=reduce_frequent_words,
    )
    pipeline = build_bertopic_pipeline(
        umap_model=FINETUNED_UMAP,
        hdbscan_model=FINETUNED_HDBSCAN,
        vectorizer_model=finetuned_vectorizer,
        ctfidf_model=ctfidf,
    )
    model, topics, _ = run_pipeline(documents, build_fn=pipeline)

    coherence = topic_coherence_gensim(model, documents, top_n=10, coherence="c_v")
    overview, nr_topics = get_topic_overview(model, top_n=10, exclude_outliers=True)

    info = model.get_topic_info()
    info_no_out = info[info["Topic"] != -1].sort_values("Count", ascending=False)
    top_topic_ids = info_no_out.head(TOP_N_TOPICS_FOR_OVERVIEW)["Topic"].tolist()

    top_topics_repr = {}
    for tid in top_topic_ids:
        if tid in overview:
            top_topics_repr[tid] = overview[tid]

    return {
        "bm25_weighting": bm25_weighting,
        "reduce_frequent_words": reduce_frequent_words,
        "coherence": coherence,
        "nr_topics": nr_topics,
        "top_topics": top_topics_repr,
    }


def _format_result(r: dict) -> str:
    """Format a single experiment result for the text file."""
    lines = [
        "",
        "=" * 60,
        f"bm25_weighting={r['bm25_weighting']}  reduce_frequent_words={r['reduce_frequent_words']}",
        "=" * 60,
        f"Coherence (c_v): {r['coherence']:.4f}",
        f"Number of topics (excl. outliers): {r['nr_topics']}",
        "",
        "Top topics (by size) – top 10 words each:",
    ]
    for tid, words_scores in r["top_topics"].items():
        words = [w for w, _ in words_scores]
        lines.append(f"  Topic {tid}: {words}")
    return "\n".join(lines)


def run_experiments(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_workers: int = 4,
) -> list:
    """Load documents once, then run all c-TF-IDF combinations in parallel processes."""
    print("Loading and preprocessing documents...")
    documents, _ = prepare_documents_for_finetuning(
        data_dir=data_dir,
        text_column=DEFAULT_TEXT_COLUMN,
        processed_column=DEFAULT_PROCESSED_COLUMN,
        min_length=DEFAULT_MIN_LENGTH,
        dedupe=DEFAULT_DEDUPE,
        max_docs=max_docs,
        sample_per_file=SAMPLE_PER_FILE,
        random_state=42,
    )
    if not documents:
        raise SystemExit("No documents after preprocessing. Exiting.")
    print(f"Using {len(documents)} documents.")

    params = [
        (documents, bm25, reduce_freq)
        for (bm25, reduce_freq) in CTFIDF_EXPERIMENTS
    ]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_experiment, doc, bm25, reduce_freq): (bm25, reduce_freq)
            for (doc, bm25, reduce_freq) in params
        }
        for i, future in enumerate(as_completed(futures)):
            bm25, reduce_freq = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(
                    f"  [{i + 1}/{len(params)}] bm25={bm25} reduce_frequent_words={reduce_freq} -> "
                    f"coherence={r['coherence']:.4f} topics={r['nr_topics']}"
                )
            except Exception as e:
                print(
                    f"  [{i + 1}/{len(params)}] bm25={bm25} reduce_frequent_words={reduce_freq} FAILED: {e}"
                )
                raise

    results.sort(key=lambda x: (x["bm25_weighting"], x["reduce_frequent_words"]))
    return results


def write_results(results: list, out_path: Path) -> None:
    """Write experiment results to a text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "c-TF-IDF finetuning results (bm25_weighting, reduce_frequent_words)\n"
        )
        f.write("Coherence: gensim c_v. Top topics: up to 10 words per topic.\n")
        for r in results:
            f.write(_format_result(r))
        f.write("\n")
    print(f"Results written to {out_path}")


def main(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_workers: int = 4,
    out_path: Path = RESULTS_FILE,
) -> None:
    results = run_experiments(
        data_dir=data_dir, max_docs=max_docs, max_workers=max_workers
    )
    write_results(results, out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Finetune c-TF-IDF: bm25_weighting and reduce_frequent_words"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Covid-19 Twitter CSV directory",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=DEFAULT_MAX_DOCS,
        help="Max documents (default: 20000)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Process pool size (default: 4)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_FILE,
        help="Output text file path",
    )
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        max_docs=args.max_docs,
        max_workers=args.max_workers,
        out_path=args.out,
    )
