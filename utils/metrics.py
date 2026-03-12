"""
Metrics and interpretability utilities for BERTopic models.

Includes:
- Gensim CoherenceModel for topic coherence (e.g. c_v).
- get_topic_overview() for interpretability (top words per topic, topic count).
- Stubs for visualize_heatmap() and reduce_topics() (for redundant topic analysis).
"""

from typing import Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Topic coherence (Gensim)
# ---------------------------------------------------------------------------


def topic_coherence_gensim(
    topic_model: Any,
    documents: List[str],
    *,
    top_n: int = 10,
    coherence: str = "c_v",
    tokenizer: Optional[Any] = None,
) -> float:
    """
    Compute topic coherence using Gensim's CoherenceModel.

    Uses the fitted BERTopic model's topic–term lists and the document corpus
    to score interpretability (e.g. c_v). Excludes topic -1 (outliers).

    Args:
        topic_model: Fitted BERTopic model (must have get_topic() and topic_labels_).
        documents: List of document strings (same as used for fitting).
        top_n: Number of top terms per topic to use for coherence.
        coherence: Coherence measure ('c_v', 'u_mass', 'c_uci', 'c_npmi'). c_v is default.
        tokenizer: Optional callable(doc: str) -> list of tokens. If None, split on whitespace.

    Returns:
        Coherence score (float). Higher is better for c_v; interpretation is measure-dependent.
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models import CoherenceModel
    except ImportError:
        raise ImportError("topic_coherence_gensim requires gensim. Install with: pip install gensim")

    def default_tokenizer(doc: str) -> List[str]:
        return [t.strip().lower() for t in doc.split() if t.strip()]

    tokenize = tokenizer or default_tokenizer
    texts = [tokenize(d) for d in documents]
    dictionary = Dictionary(texts)

    topic_ids = [t for t in topic_model.get_topic_info()["Topic"].tolist() if t != -1]
    vocab = set(dictionary.token2id)
    topics = []
    for tid in topic_ids:
        tup = topic_model.get_topic(tid)
        if tup:
            # CoherenceModel requires topic words to be in the dictionary (token2id)
            words = [w for w, _ in tup[:top_n] if w in vocab]
            if words:
                topics.append(words)

    if not topics:
        return 0.0

    cm = CoherenceModel(
        topics=topics,
        texts=texts,
        dictionary=dictionary,
        coherence=coherence,
    )
    return cm.get_coherence()


# ---------------------------------------------------------------------------
# Topic interpretability: top words and topic count (redundancy proxy)
# ---------------------------------------------------------------------------


def get_topic_overview(
    topic_model: Any,
    *,
    top_n: int = 10,
    exclude_outliers: bool = True,
) -> Tuple[dict, int]:
    """
    Get top words per topic for interpretability and the number of topics
    (excluding outliers), as a proxy for redundancy (many small/redundant topics
    vs fewer, distinct ones).

    Args:
        topic_model: Fitted BERTopic model.
        top_n: Number of top terms to return per topic.
        exclude_outliers: If True, do not include topic -1 in the overview.

    Returns:
        (topic_overview, nr_topics):
        - topic_overview: dict mapping topic_id -> list of (word, score) for top_n terms.
        - nr_topics: Number of topics (excluding -1 if exclude_outliers).
    """
    info = topic_model.get_topic_info()
    topic_ids = info["Topic"].tolist()
    if exclude_outliers:
        topic_ids = [t for t in topic_ids if t != -1]

    overview = {}
    for tid in topic_ids:
        tup = topic_model.get_topic(tid)
        if tup:
            overview[tid] = tup[:top_n]
        else:
            overview[tid] = []

    return overview, len(topic_ids)


# ---------------------------------------------------------------------------
# Heatmap and topic reduction (for redundant topic analysis)
# ---------------------------------------------------------------------------


def visualize_topic_heatmap(
    topic_model: Any,
    docs: List[str],
    **kwargs: Any,
) -> Any:
    """
    Visualise topic similarity heatmap via BERTopic's visualize_heatmap().

    Helps identify redundant topics (high similarity). Requires a fitted model
    and the documents used for fitting.

    Args:
        topic_model: Fitted BERTopic model.
        docs: List of documents used for fitting.
        **kwargs: Passed to topic_model.visualize_heatmap().

    Returns:
        Plotly figure from topic_model.visualize_heatmap().
    """
    return topic_model.visualize_heatmap(docs=docs, **kwargs)


def reduce_topics(
    topic_model: Any,
    docs: List[str],
    nr_topics: int = 20,
    **kwargs: Any,
) -> Any:
    """
    Reduce the number of topics by merging similar ones.

    Wrapper around topic_model.reduce_topics(docs, nr_topics=nr_topics).
    Use after inspecting the heatmap to set a target number of topics.

    Args:
        topic_model: Fitted BERTopic model.
        docs: List of documents used for fitting.
        nr_topics: Target number of topics after reduction.
        **kwargs: Passed to topic_model.reduce_topics().

    Returns:
        The model (possibly in-place updated) or new topics; see BERTopic docs.
    """
    return topic_model.reduce_topics(docs, nr_topics=nr_topics, **kwargs)
