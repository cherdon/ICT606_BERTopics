"""
BERTopic pipeline configuration for COVID-19 Twitter topic modeling.

This module defines all pipeline steps (embedding, dimensionality reduction,
clustering, vectorizer, c-TF-IDF, representation) with model choices and
comments to support tuning for short, informal tweet text.
"""

import time
from pathlib import Path
from typing import Any, List, Optional

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


# ---------------------------------------------------------------------------
# 1. EMBEDDING MODEL
# ---------------------------------------------------------------------------

def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    **kwargs: Any,
) -> SentenceTransformer:
    """
    Embedding model: maps each tweet (document) to a dense vector.

    Choice: all-MiniLM-L6-v2 — Fast, good quality on short text, and the default
    in BERTopic for English. Sentence-transformers are well-suited for social
    media text; this model balances speed and semantic quality for large tweet
    corpora. It handles informal language and short sentences better than
    generic BERT-style encoders.

    TODO: Finetune this step:
    - Try paraphrase-MiniLM-L3-v2 for faster runs on very large datasets.
    - Try all-mpnet-base-v2 for higher quality (slower, more RAM).
    - If you have multilingual tweets, use paraphrase-multilingual-MiniLM-L12-v2.
    - Consider domain-adapted embeddings (e.g. fine-tuned on COVID/twitter data)
      if you have labelled data.
    """
    return SentenceTransformer(model_name, device=device, **kwargs)


# ---------------------------------------------------------------------------
# 2. DIMENSIONALITY REDUCTION (UMAP)
# ---------------------------------------------------------------------------

def get_umap_model(
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
    **kwargs: Any,
) -> UMAP:
    """
    UMAP reduces high-dimensional embeddings to a low-dimensional space
    where HDBSCAN clusters. BERTopic typically uses 5 components for the
    topic space.

    Choice: n_components=5 (BERTopic default), n_neighbors=15, min_dist=0.0.
    Cosine metric matches normalized sentence embeddings. Lower min_dist
    keeps local structure tighter, which helps topic coherence on short texts.

    TODO: Finetune this step:
    - Increase n_neighbors (e.g. 20–50) for broader, fewer topics; decrease for
      more granular topics.
    - Adjust min_dist: higher values (e.g. 0.1) can reduce over-clustering.
    - Fix random_state for reproducibility across runs.
    """
    return UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 3. CLUSTERING (HDBSCAN)
# ---------------------------------------------------------------------------

def get_hdbscan_model(
    min_cluster_size: int = 15,
    min_samples: int = 5,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    **kwargs: Any,
) -> HDBSCAN:
    """
    HDBSCAN clusters the UMAP-reduced embeddings without requiring a fixed
    number of clusters. It finds density-based clusters and assigns outliers
    to topic -1, which fits topic modeling where cluster count is unknown.

    Choice: min_cluster_size=15 (reasonable for tweets; smaller = more topics),
    min_samples=5, cluster_selection_method="eom" (often better than "leaf"
    for topic-like clusters). Euclidean metric is used in UMAP space.

    TODO: Finetune this step:
    - Tune min_cluster_size: decrease (e.g. 10) for more fine-grained topics;
      increase (e.g. 25–50) for fewer, broader topics. Scale with corpus size.
    - Tune min_samples: higher values can reduce noise but merge small topics.
    - Try cluster_selection_method="leaf" if "eom" yields too few clusters.
    """
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 4. VECTORIZER (for c-TF-IDF input)
# ---------------------------------------------------------------------------

def get_vectorizer_model(
    ngram_range: tuple[int, int] = (1, 2),
    stop_words: str = "english",
    max_features: Optional[int] = 10_000,
    min_df: int = 2,
    max_df: float = 0.95,
    **kwargs: Any,
) -> CountVectorizer:
    """
    CountVectorizer builds the document-term matrix per topic, which is then
    weighted by c-TF-IDF. For tweets we want unigrams and bigrams to capture
    phrases (e.g. "vaccine hesitancy", "wear a mask").

    Choice: ngram_range=(1, 2), stop_words="english", min_df=2 to drop
    rare terms, max_df=0.95 to drop corpus-wide buzzwords. max_features
    caps vocabulary size for speed and to avoid noise.

    TODO: Finetune this step:
    - Adjust ngram_range: (1, 1) for more generic topics; (1, 3) if
      multi-word phrases are important.
    - Tune min_df / max_df for your corpus size and noise level.
    - Consider max_features=5000 for smaller corpora or 20000 for very large.
    - Add custom stop words (e.g. COVID-related generic terms) via stop_words
      or a list.
    """
    return CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 5. c-TF-IDF (Class-based TF-IDF)
# ---------------------------------------------------------------------------

def get_ctfidf_model(
    bm25_weighting: bool = False,
    reduce_frequent_words: bool = False,
    **kwargs: Any,
) -> ClassTfidfTransformer:
    """
    ClassTfidfTransformer applies class-based TF-IDF: each "class" is a
    topic, so terms are weighted by how discriminative they are for that
    topic vs the rest. This produces the topic-term matrix used for
    labels and representations.

    Choice: Default c-TF-IDF (bm25_weighting=False). For short documents
    like tweets, standard c-TF-IDF is often sufficient; BM25 can help if
    you see many very short "documents" per topic.

    TODO: Finetune this step:
    - Set bm25_weighting=True if topic keywords look too dominated by
      frequent words; BM25 dampens term frequency.
    - Set reduce_frequent_words=True to downweight terms that appear in
      many topics (e.g. "covid", " pandemic").
    """
    return ClassTfidfTransformer(
        bm25_weighting=bm25_weighting,
        reduce_frequent_words=reduce_frequent_words,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 6. REPRESENTATION MODEL (topic labels)
# ---------------------------------------------------------------------------

def get_representation_model(
    top_n_words: int = 10,
    nr_repr_docs: int = 5,
    nr_samples: int = 500,
    nr_candidate_words: int = 100,
    random_state: int = 42,
    **kwargs: Any,
) -> KeyBERTInspired:
    """
    KeyBERTInspired refines topic labels by comparing candidate keywords
    (from c-TF-IDF) with the topic embedding via cosine similarity,
    yielding more interpretable, semantically aligned labels. It uses
    the same embedding model as the main pipeline (set in BERTopic).

    Choice: top_n_words=10 (BERTopic default; keep below ~20 for coherence),
    nr_repr_docs=5, nr_samples=500. Good balance for short tweet topics.

    TODO: Finetune this step:
    - Adjust top_n_words: 5–15 for concise labels; up to 20–30 if you need
      more context per topic (recommended to stay ≤20 for coherence).
    - Increase nr_repr_docs / nr_samples if topics have many documents and
      you want more representative candidates.
    - You can pass a list of representation models (e.g. [KeyBERTInspired(),
      MaximalMarginalRelevance()]) to BERTopic and it will combine them.
    """
    return KeyBERTInspired(
        top_n_words=top_n_words,
        nr_repr_docs=nr_repr_docs,
        nr_samples=nr_samples,
        nr_candidate_words=nr_candidate_words,
        random_state=random_state,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# BUILD FULL PIPELINE
# ---------------------------------------------------------------------------

def build_bertopic_pipeline(
    embedding_model: Optional[SentenceTransformer] = None,
    umap_model: Optional[UMAP] = None,
    hdbscan_model: Optional[HDBSCAN] = None,
    vectorizer_model: Optional[CountVectorizer] = None,
    ctfidf_model: Optional[ClassTfidfTransformer] = None,
    representation_model: Optional[KeyBERTInspired] = None,
    verbose: bool = True,
    calculate_probabilities: bool = False,
    **bertopic_kwargs: Any,
) -> BERTopic:
    """
    Assemble a BERTopic model with all pipeline steps configured for
    COVID-19 tweet topic modeling. Any component left as None uses the
    default from the get_* functions above (called internally here).
    """
    embedding_model = embedding_model or get_embedding_model()
    umap_model = umap_model or get_umap_model()
    hdbscan_model = hdbscan_model or get_hdbscan_model()
    vectorizer_model = vectorizer_model or get_vectorizer_model()
    ctfidf_model = ctfidf_model or get_ctfidf_model()
    representation_model = representation_model or get_representation_model()

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        verbose=verbose,
        calculate_probabilities=calculate_probabilities,
        **bertopic_kwargs,
    )


# ---------------------------------------------------------------------------
# CONVENIENCE: fit and return model + topic assignments
# ---------------------------------------------------------------------------

def _timed_step(step_name: str):
    """Return a wrapper that prints and times a method call."""
    def wrapper(fn):
        def timed_fn(*args, **kwargs):
            print(f"  {step_name} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            print(f"{time.perf_counter() - t0:.1f}s")
            return out
        return timed_fn
    return wrapper


def run_pipeline(
    documents: List[str],
    build_fn=None,
) -> tuple[BERTopic, List[int], Optional[Any]]:
    """
    Run the full pipeline on a list of documents (e.g. cleaned tweets).
    Returns (model, topic_ids, probabilities).
    """
    if build_fn is None:
        build_fn = build_bertopic_pipeline

    print("1. Building pipeline ...")
    t0 = time.perf_counter()
    model = build_fn()
    print(f"   Done in {time.perf_counter() - t0:.1f}s")

    # Wrap internal steps so we can time each (step 1 = build above; 2–6 = fit)
    steps = [
        ("2. Embedding", "_extract_embeddings"),
        ("3. UMAP", "_reduce_dimensionality"),
        ("4. HDBSCAN", "_cluster_embeddings"),
        ("5. c-TF-IDF & topics", "_extract_topics"),
        ("6. Topic vectors", "_create_topic_vectors"),
    ]
    originals = {}
    for label, method_name in steps:
        fn = getattr(model, method_name, None)
        if fn is not None:
            originals[method_name] = fn
            setattr(model, method_name, _timed_step(label)(fn))

    print(f"Fitting on {len(documents)} documents (steps 2–6):")
    t0 = time.perf_counter()
    try:
        topics, probs = model.fit_transform(documents)
    finally:
        for method_name, fn in originals.items():
            setattr(model, method_name, fn)
    print(f"  Total fit: {time.perf_counter() - t0:.1f}s")

    return model, topics, probs
