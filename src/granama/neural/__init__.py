"""Neural ranking of anagrams: fluency from a language model, relevance from an embedding model.

``NeuralRanker`` works with any ``Fluency`` and ``Relevance`` implementation. The Hugging
Face ones in ``granama.neural.models`` need the ``lm`` extra, imported only when created.
"""

from granama.neural.models import (
    DEFAULT_FLUENCY_MODEL,
    DEFAULT_RELEVANCE_MODEL,
    CausalLMFluency,
    EmbeddingRelevance,
)
from granama.neural.ranker import Fluency, NeuralRanker, RankedAnagram, Relevance

__all__ = [
    "DEFAULT_FLUENCY_MODEL",
    "DEFAULT_RELEVANCE_MODEL",
    "CausalLMFluency",
    "EmbeddingRelevance",
    "Fluency",
    "NeuralRanker",
    "RankedAnagram",
    "Relevance",
]
