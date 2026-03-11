"""Preprocessing functions for Covid-19 Twitter data before BERTopic."""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

try:
    import emoji
except ImportError:
    emoji = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# URL and artifact patterns (compiled once)
# ---------------------------------------------------------------------------
_URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    re.IGNORECASE,
)
_RT_PATTERN = re.compile(
    r"\bRT\s*@\s*[\w]+:\s*",
    re.IGNORECASE,
)


def remove_urls(text: str) -> str:
    """Remove URLs (http, https, www) from text.

    Args:
        text: Input string.

    Returns:
        String with URLs removed.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    return _URL_PATTERN.sub(" ", text)


def remove_special_chars_preserve_text(text: str) -> str:
    """Remove @ and # but keep the text that follows (e.g. @user -> user, #tag -> tag).

    Args:
        text: Input string.

    Returns:
        String with @ and # removed; mentions and hashtag text preserved.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    # Replace @mention with mention, #hashtag with hashtag (preserve the word)
    out = re.sub(r"@(\w+)", r"\1", text)
    out = re.sub(r"#(\w+)", r"\1", out)
    return out


def remove_artifacts(text: str) -> str:
    """Remove non-text artifacts such as &amp;, \\n, \\t, &lt;, &gt;, &quot;, etc.

    Args:
        text: Input string.

    Returns:
        String with HTML entities and control characters normalized/removed.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    out = text.replace("&amp;", " ")
    out = out.replace("&lt;", " ")
    out = out.replace("&gt;", " ")
    out = out.replace("&quot;", " ")
    out = out.replace("&#39;", "'")
    out = re.sub(r"&#\d+;", " ", out)
    out = out.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return out


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines to a single space and strip.

    Args:
        text: Input string.

    Returns:
        String with normalized whitespace.
    """
    if not isinstance(text, str):
        return text
    out = re.sub(r"\s+", " ", text)
    return out.strip()


def remove_rt_markers(text: str) -> str:
    """Remove retweet markers like 'RT @username: ' while keeping the rest of the text.

    Args:
        text: Input string.

    Returns:
        String with RT @user: prefix removed.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    return _RT_PATTERN.sub(" ", text)


def emojis_to_text(text: str) -> str:
    """Convert emojis to words using emoji.demojize(), then split by underscore.

    e.g. 😷 -> :face_with_medical_mask: -> 'face with medical mask'.

    Requires the 'emoji' package. If not installed, returns text unchanged.

    Args:
        text: Input string possibly containing emojis.

    Returns:
        String with emojis replaced by space-separated words.
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if emoji is None:
        return text
    # Standard demojize gives e.g. ":face_with_medical_mask:"
    demojized = emoji.demojize(text)
    # Replace each :shortcode: with words separated by spaces (underscore -> space)
    def shortcode_to_words(match: re.Match) -> str:
        return match.group(1).replace("_", " ")
    out = re.sub(r":([a-z0-9_]+):", shortcode_to_words, demojized)
    return out


def to_lowercase(text: str) -> str:
    """Convert text to lowercase.

    Args:
        text: Input string.

    Returns:
        Lowercase string.
    """
    if not isinstance(text, str):
        return text
    return text.lower()


def preprocess_tweet(
    text: str,
    *,
    remove_urls_flag: bool = True,
    remove_artifacts_flag: bool = True,
    remove_rt_flag: bool = True,
    remove_special_chars_flag: bool = True,
    emojis_to_text_flag: bool = True,
    normalize_whitespace: bool = True,
    lowercase: bool = True,
) -> str:
    """Apply a full preprocessing pipeline to a single tweet string.

    Order: URLs -> artifacts -> RT markers -> @/# (preserve text) -> emojis -> whitespace -> lowercase.

    Args:
        text: Raw tweet text.
        remove_urls_flag: Whether to remove URLs.
        remove_artifacts_flag: Whether to remove &amp;, \\n, etc.
        remove_rt_flag: Whether to remove 'RT @user:'.
        remove_special_chars_flag: Whether to strip @ and # but keep following text.
        emojis_to_text_flag: Whether to convert emojis to words.
        normalize_whitespace: Whether to collapse and trim whitespace.
        lowercase: Whether to lowercase.

    Returns:
        Preprocessed tweet string.
    """
    if not isinstance(text, str):
        return str(text)
    s = text
    if remove_urls_flag:
        s = remove_urls(s)
    if remove_artifacts_flag:
        s = remove_artifacts(s)
    if remove_rt_flag:
        s = remove_rt_markers(s)
    if remove_special_chars_flag:
        s = remove_special_chars_preserve_text(s)
    if emojis_to_text_flag:
        s = emojis_to_text(s)
    if normalize_whitespace:
        s = remove_extra_whitespace(s)
    if lowercase:
        s = to_lowercase(s)
    return s


def preprocess_tweet_series(
    series: "pd.Series",
    show_progress: bool = True,
    progress_desc: str = "Preprocessing tweets",
    **kwargs: bool,
) -> "pd.Series":
    """Apply preprocess_tweet to every element of a pandas Series.

    Args:
        series: Series of tweet strings.
        show_progress: If True and tqdm is installed, show a progress bar.
        progress_desc: Label for the progress bar when show_progress is True.
        **kwargs: Passed to preprocess_tweet.

    Returns:
        New Series of preprocessed strings.
    """
    s = series.astype(str)
    if show_progress:
        try:
            from tqdm import tqdm
            tqdm.pandas(desc=progress_desc)
            return s.progress_apply(lambda x: preprocess_tweet(x, **kwargs))
        except ImportError:
            pass
    return s.apply(lambda x: preprocess_tweet(x, **kwargs))


def preprocess_tweets(
    df: pd.DataFrame,
    text_column: str = "original_text",
    output_column: Optional[str] = None,
    show_progress: bool = True,
    progress_desc: str = "Preprocessing tweets",
    **kwargs: bool,
) -> pd.DataFrame:
    """Apply preprocess_tweet to the given text column of a DataFrame.

    Uses preprocess_tweet_series under the hood. Returns a copy of the
    DataFrame with the (possibly new) column set to the preprocessed text.

    Args:
        df: DataFrame containing a column of raw tweet text.
        text_column: Column name containing raw tweet strings.
        output_column: If set, preprocessed text is written here; otherwise
            text_column is overwritten.
        show_progress: If True and tqdm is installed, show a progress bar.
        progress_desc: Label for the progress bar when show_progress is True.
        **kwargs: Passed to preprocess_tweet (e.g. remove_urls_flag=True).

    Returns:
        New DataFrame with preprocessed text in output_column or text_column.
    """
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not in DataFrame")
    out = df.copy()
    target = output_column if output_column is not None else text_column
    out[target] = preprocess_tweet_series(
        out[text_column],
        show_progress=show_progress,
        progress_desc=progress_desc,
        **kwargs,
    )
    return out


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
