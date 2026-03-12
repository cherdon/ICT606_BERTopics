"""
Finetune CountVectorizer for BERTopic by sweeping ngram_range, min_df, and max_df.

Runs all permutations in separate processes (ProcessPoolExecutor), computes topic coherence (gensim c_v)
and get_topic overview for interpretability. Writes results to
experiments/results/vectorizer.txt.

Sweep:
- ngram_range: (1,1), (1,2)
- min_df: 5, 10, 20
- max_df: 0.85, 0.9
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

from utils.bertopic_pipeline import build_bertopic_pipeline, get_vectorizer_model, run_pipeline
from utils.metrics import topic_coherence_gensim, get_topic_overview
from utils import prepare_documents_for_finetuning

# Same COVID stopwords as run_bertopic
COVID_STOPWORDS = [
    "covid", "covid19", "covid_19", "covid-19", "covid__19",
    "coronavirus", "pandemic", "covid19pandemic",
]

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

NGRAM_RANGE_VALUES = [(1, 1), (1, 2)]
MIN_DF_VALUES = [5, 10, 20]
MAX_DF_VALUES = [0.85, 0.9]

RESULTS_DIR = _root / "experiments" / "results"
RESULTS_FILE = RESULTS_DIR / "vectorizer.txt"

TOP_N_TOPICS_FOR_OVERVIEW = 15  # number of top topics to show in results (by size)


def _run_single_experiment(
    documents: list,
    ngram_range: tuple,
    min_df: int,
    max_df: float,
) -> dict:
    """Run one BERTopic fit with the given vectorizer params; return coherence and topic overview."""
    vectorizer = get_vectorizer_model(
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        extra_stop_words=COVID_STOPWORDS,
    )
    pipeline = build_bertopic_pipeline(vectorizer_model=vectorizer)
    model, topics, _ = run_pipeline(documents, build_fn=pipeline)

    coherence = topic_coherence_gensim(model, documents, top_n=10, coherence="c_v")
    overview, nr_topics = get_topic_overview(model, top_n=10, exclude_outliers=True)

    # Topic sizes for "top" topics (by count, excluding -1)
    info = model.get_topic_info()
    info_no_out = info[info["Topic"] != -1].sort_values("Count", ascending=False)
    top_topic_ids = info_no_out.head(TOP_N_TOPICS_FOR_OVERVIEW)["Topic"].tolist()

    top_topics_repr = {}
    for tid in top_topic_ids:
        if tid in overview:
            top_topics_repr[tid] = overview[tid]

    return {
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_df": max_df,
        "coherence": coherence,
        "nr_topics": nr_topics,
        "top_topics": top_topics_repr,
    }


def _format_result(r: dict) -> str:
    """Format a single experiment result for the text file."""
    lines = [
        "",
        "=" * 60,
        f"ngram_range={r['ngram_range']}  min_df={r['min_df']}  max_df={r['max_df']}",
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
    """Load documents once, then run all vectorizer permutations in parallel processes."""
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
        (documents, ngram, min_df, max_df)
        for ngram in NGRAM_RANGE_VALUES
        for min_df in MIN_DF_VALUES
        for max_df in MAX_DF_VALUES
    ]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_experiment, doc, ng, mi, mx): (ng, mi, mx)
            for (doc, ng, mi, mx) in params
        }
        for i, future in enumerate(as_completed(futures)):
            ngram, min_df, max_df = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(f"  [{i + 1}/{len(params)}] ngram={ngram} min_df={min_df} max_df={max_df} -> coherence={r['coherence']:.4f} topics={r['nr_topics']}")
            except Exception as e:
                print(f"  [{i + 1}/{len(params)}] ngram={ngram} min_df={min_df} max_df={max_df} FAILED: {e}")
                raise

    # Sort by (ngram, min_df, max_df) for consistent output
    results.sort(key=lambda x: (x["ngram_range"], x["min_df"], x["max_df"]))
    return results


def write_results(results: list, out_path: Path) -> None:
    """Write experiment results to a text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Vectorizer finetuning results (ngram_range, min_df, max_df)\n")
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
    results = run_experiments(data_dir=data_dir, max_docs=max_docs, max_workers=max_workers)
    write_results(results, out_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune vectorizer: ngram_range, min_df, max_df")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Covid-19 Twitter CSV directory")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="Max documents (default: 20000)")
    parser.add_argument("--max-workers", type=int, default=4, help="Process pool size (default: 4)")
    parser.add_argument("--out", type=Path, default=RESULTS_FILE, help="Output text file path")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        max_docs=args.max_docs,
        max_workers=args.max_workers,
        out_path=args.out,
    )
