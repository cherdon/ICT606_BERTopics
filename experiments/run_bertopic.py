"""Run BERTopic on Covid-19 Twitter data using utils.preprocessing."""

import argparse
from pathlib import Path

from bertopic import BERTopic

# Optional: add project root to path when running as script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import prepare_for_bertopic
from utils import visualisation


# Config: override via CLI or edit here
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "covid19_twitter_dataset"
DEFAULT_TEXT_COLUMN = "clean_tweet"
DEFAULT_MIN_LENGTH = 10
DEFAULT_DEDUPE = True
DEFAULT_TOP_N_TOPICS = 10
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def run(
    data_dir: Path | str,
    text_column: str = DEFAULT_TEXT_COLUMN,
    min_length: int = DEFAULT_MIN_LENGTH,
    dedupe: bool = DEFAULT_DEDUPE,
    top_n_topics: int = DEFAULT_TOP_N_TOPICS,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    run_visualisations: bool = False,
) -> None:
    """Load data, fit BERTopic, save model and results; optionally run visualisations."""
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

    print("Fitting BERTopic...")
    model = BERTopic(
        verbose=True,
        calculate_probabilities=False,
    )
    topics, probs = model.fit_transform(documents)

    # Save model
    model_path = output_dir / "bertopic_model"
    model.save(str(model_path))
    print(f"Model saved to {model_path}")

    # Save topic assignments (aligned to documents)
    if metadata is not None:
        metadata = metadata.copy()
    else:
        import pandas as pd
        metadata = pd.DataFrame()
    metadata["topic_id"] = topics
    metadata["document"] = documents
    results_path = output_dir / "topic_assignments.csv"
    metadata.to_csv(results_path, index=False)
    print(f"Topic assignments saved to {results_path}")

    # Optional visualisations (no-op until utils.visualisation is implemented)
    if run_visualisations:
        try:
            visualisation.visualise_topics(model, documents)
        except NotImplementedError:
            pass
        try:
            visualisation.plot_topic_barchart(model, top_n=top_n_topics)
        except NotImplementedError:
            pass
        if metadata is not None and "created_at" in metadata.columns:
            try:
                visualisation.plot_topic_over_time(
                    model, documents, metadata["created_at"].tolist()
                )
            except NotImplementedError:
                pass

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERTopic on Covid-19 Twitter data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing Covid-19 Twitter CSV files",
    )
    parser.add_argument(
        "--text-column",
        default=DEFAULT_TEXT_COLUMN,
        help="Column to use as document text",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_LENGTH,
        help="Minimum character length per document",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable deduplication by text",
    )
    parser.add_argument(
        "--top-n-topics",
        type=int,
        default=DEFAULT_TOP_N_TOPICS,
        help="Top N topics for barchart (when viz implemented)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model and results",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Run visualisation stubs when implemented",
    )
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        text_column=args.text_column,
        min_length=args.min_length,
        dedupe=not args.no_dedupe,
        top_n_topics=args.top_n_topics,
        output_dir=args.output_dir,
        run_visualisations=args.viz,
    )


if __name__ == "__main__":
    main()
