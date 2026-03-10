"""Preprocessing functions for Covid-19 Twitter data before BERTopic."""

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


def load_covid_tweets(
    data_dir: str | Path,
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load all matching CSV files from data_dir into one DataFrame.

    Args:
        data_dir: Directory containing Covid-19 Twitter CSV files.
        pattern: Glob pattern for CSV files (default "*.csv").

    Returns:
        Combined DataFrame of all loaded CSVs. Malformed rows are skipped.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(
            f,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="skip",
        )
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def filter_language(df: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Restrict to rows where the language column equals the given code.

    Args:
        df: DataFrame with a 'lang' column.
        lang: Language code to keep (default "en").

    Returns:
        Filtered DataFrame (copy).
    """
    if "lang" not in df.columns:
        return df.copy()
    return df.loc[df["lang"].astype(str).str.strip().str.lower() == lang.lower()].copy()


def remove_duplicates(
    df: pd.DataFrame,
    text_column: str = "clean_tweet",
) -> pd.DataFrame:
    """Drop duplicate rows by the given text column, keeping first occurrence.

    Args:
        df: DataFrame with the text column.
        text_column: Column name used for deduplication.

    Returns:
        Deduplicated DataFrame (copy).
    """
    if text_column not in df.columns:
        return df.copy()
    return df.drop_duplicates(subset=[text_column], keep="first").copy()


def get_documents(
    df: pd.DataFrame,
    text_column: str = "clean_tweet",
    min_length: int = 10,
    return_metadata: bool = True,
) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """Extract document list and optional metadata for BERTopic.

    Drops NaN/empty text, strips whitespace, and filters by minimum length
    (number of characters). Returns aligned documents and metadata.

    Args:
        df: DataFrame with the text column and optional metadata columns.
        text_column: Column containing document text.
        min_length: Minimum character length for a document to be included.
        return_metadata: If True, return a metadata DataFrame aligned to documents.

    Returns:
        (documents, metadata). documents is a list of strings. metadata is a
        DataFrame with same length as documents (created_at, sentiment, etc.)
        or None if return_metadata is False.
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not in DataFrame")

    # Drop rows with missing text and strip
    out = df[[text_column]].copy()
    out[text_column] = out[text_column].astype(str).str.strip()
    out = out[out[text_column].str.len() >= min_length]

    documents = out[text_column].tolist()

    metadata = None
    if return_metadata:
        meta_cols = [c for c in df.columns if c != text_column]
        if meta_cols:
            # Align by index after dropna/length filter
            meta = df.loc[out.index, meta_cols].copy()
            meta.reset_index(drop=True, inplace=True)
            metadata = meta
        else:
            metadata = pd.DataFrame(index=range(len(documents)))

    return documents, metadata


def prepare_for_bertopic(
    data_dir: str | Path,
    text_column: str = "clean_tweet",
    lang: str = "en",
    min_length: int = 10,
    dedupe: bool = True,
    pattern: str = "*.csv",
) -> Tuple[List[str], Optional[pd.DataFrame]]:
    """Load Covid-19 tweets, filter, optionally dedupe, and return documents for BERTopic.

    Pipeline: load_covid_tweets → filter_language → (optional) remove_duplicates
    → get_documents.

    Args:
        data_dir: Directory containing Covid-19 Twitter CSV files.
        text_column: Column to use as document text (default "clean_tweet").
        lang: Language code to keep (default "en").
        min_length: Minimum character length per document.
        dedupe: Whether to remove duplicate texts.
        pattern: Glob pattern for CSV files.

    Returns:
        (documents, metadata) for use with BERTopic.fit_transform(documents).
    """
    df = load_covid_tweets(data_dir, pattern=pattern)
    df = filter_language(df, lang=lang)
    if dedupe:
        df = remove_duplicates(df, text_column=text_column)
    return get_documents(
        df,
        text_column=text_column,
        min_length=min_length,
        return_metadata=True,
    )
