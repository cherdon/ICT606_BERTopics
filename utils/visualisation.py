"""Visualisation helpers for BERTopic results (stubs for later implementation)."""

from typing import Any, List, Optional


def visualise_topics(
    model: Any,
    docs: List[str],
    **kwargs: Any,
) -> Any:
    """Visualise topic distribution or inter-topic distances.

    To be implemented: use BERTopic's built-in visualisation or custom plots
    for topic overview (e.g. intertopic distance map).

    Args:
        model: Fitted BERTopic model.
        docs: List of documents used for fitting.
        **kwargs: Optional arguments for the visualisation.

    Returns:
        Plot object or figure (implementation-dependent).
    """
    raise NotImplementedError("visualise_topics is not yet implemented")


def plot_topic_barchart(
    model: Any,
    topic_id: Optional[int] = None,
    top_n: int = 10,
    **kwargs: Any,
) -> Any:
    """Plot bar chart of top terms per topic (or for a single topic).

    To be implemented: bar chart of c-TF-IDF or top words per topic.

    Args:
        model: Fitted BERTopic model.
        topic_id: Specific topic ID, or None for all topics.
        top_n: Number of top terms to show per topic.
        **kwargs: Optional arguments for the plot.

    Returns:
        Plot object or figure (implementation-dependent).
    """
    raise NotImplementedError("plot_topic_barchart is not yet implemented")


def plot_topic_over_time(
    model: Any,
    docs: List[str],
    timestamps: List[Any],
    **kwargs: Any,
) -> Any:
    """Plot topic prevalence over time (e.g. by date).

    To be implemented: aggregate topic assignments by time bin and plot
    temporal trends for use with created_at from tweet metadata.

    Args:
        model: Fitted BERTopic model.
        docs: List of documents (aligned to timestamps).
        timestamps: Timestamps or dates for each document.
        **kwargs: Optional arguments for the plot.

    Returns:
        Plot object or figure (implementation-dependent).
    """
    raise NotImplementedError("plot_topic_over_time is not yet implemented")
