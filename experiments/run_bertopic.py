from pathlib import Path
from utils.bertopic_pipeline import run_pipeline

# Columns to drop from the raw CSV (not needed for topic modeling)
COLUMNS_TO_DROP = [
    "id",
    "source",
    "hashtags",
    "user_mentions",
    "clean_tweet",
    "compound",
    "neg",
    "neu",
    "pos",
]

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

    print("Building and fitting BERTopic pipeline...")
    model, topics, probs = run_pipeline(documents)
    print(f"Found {len(set(topics) - {-1})} topics (+ outliers -1).")
    print("Topic info:", model.get_topic_info().head(10).to_string())