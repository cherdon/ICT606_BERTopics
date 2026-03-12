"""
Finetune representation models by comparing KeyBERTInspired only vs KeyBERTInspired + MMR.

Runs three experiments in separate processes (ProcessPoolExecutor), computes topic coherence
(gensim c_v) and get_topic overview for interpretability. Writes results to
experiments/results/representation.txt.

Uses the full optimized pipeline: UMAP, HDBSCAN, vectorizer (ngram_range=(1,2), min_df=5,
max_df=0.85), and c-TF-IDF (bm25_weighting=False, reduce_frequent_words=False).

Experiments:
1. KeyBERTInspired only
2. KeyBERTInspired + MaximalMarginalRelevance (diversity=0.3)
3. KeyBERTInspired + MaximalMarginalRelevance (diversity=0.6)
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

from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from utils.bertopic_pipeline import (
    build_bertopic_pipeline,
    get_vectorizer_model,
    get_umap_model,
    get_hdbscan_model,
    get_ctfidf_model,
    get_representation_model,
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

# Optimized pipeline (same as finetune_ctfidf)
FINETUNED_VECTORIZER_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.85,
    "extra_stop_words": COVID_STOPWORDS,
}
FINETUNED_UMAP = get_umap_model(n_neighbors=25, min_dist=0.01)
FINETUNED_HDBSCAN = get_hdbscan_model(min_cluster_size=25, min_samples=5)
FINETUNED_CTFIDF = get_ctfidf_model(bm25_weighting=False, reduce_frequent_words=False)

# (experiment_label, representation_model)
# representation_model can be a single model or a list (BERTopic combines them)
REPRESENTATION_EXPERIMENTS = [
    ("KeyBERTInspired_only", get_representation_model()),
    (
        "KeyBERTInspired_MMR_diversity_0.3",
        [KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.3)],
    ),
    (
        "KeyBERTInspired_MMR_diversity_0.6",
        [KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.6)],
    ),
]

RESULTS_DIR = _root / "experiments" / "results"
RESULTS_FILE = RESULTS_DIR / "representation.txt"

TOP_N_TOPICS_FOR_OVERVIEW = 15


def _run_single_experiment(
    documents: list,
    experiment_label: str,
    representation_model,
) -> dict:
    """Run one BERTopic fit with the given representation model; return coherence and topic overview."""
    finetuned_vectorizer = get_vectorizer_model(**FINETUNED_VECTORIZER_PARAMS)
    pipeline = build_bertopic_pipeline(
        umap_model=FINETUNED_UMAP,
        hdbscan_model=FINETUNED_HDBSCAN,
        vectorizer_model=finetuned_vectorizer,
        ctfidf_model=FINETUNED_CTFIDF,
        representation_model=representation_model,
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
        "experiment_label": experiment_label,
        "coherence": coherence,
        "nr_topics": nr_topics,
        "top_topics": top_topics_repr,
    }


def _format_result(r: dict) -> str:
    """Format a single experiment result for the text file."""
    lines = [
        "",
        "=" * 60,
        f"Representation: {r['experiment_label']}",
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
    """Load documents once, then run all representation experiments in parallel processes."""
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
        (documents, label, repr_model)
        for (label, repr_model) in REPRESENTATION_EXPERIMENTS
    ]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_single_experiment, doc, label, repr_model): label
            for (doc, label, repr_model) in params
        }
        for i, future in enumerate(as_completed(futures)):
            label = futures[future]
            try:
                r = future.result()
                results.append(r)
                print(
                    f"  [{i + 1}/{len(params)}] {label} -> "
                    f"coherence={r['coherence']:.4f} topics={r['nr_topics']}"
                )
            except Exception as e:
                print(f"  [{i + 1}/{len(params)}] {label} FAILED: {e}")
                raise

    # Keep order: KeyBERTInspired_only, then MMR 0.3, then MMR 0.6
    order = {label: i for i, (label, _) in enumerate(REPRESENTATION_EXPERIMENTS)}
    results.sort(key=lambda x: order[x["experiment_label"]])
    return results


def write_results(results: list, out_path: Path) -> None:
    """Write experiment results to a text file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Representation finetuning results (KeyBERTInspired, + MMR)\n")
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
        description="Finetune representation: KeyBERTInspired vs KeyBERTInspired + MMR"
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
