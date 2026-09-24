"""Ranking of generated anagrams, most plausible first.

Anagrams are ranked by their words' rarity tiers (see ``Vocabulary.rarity``): first by the
rarest word, then by the sum of tiers. Ties keep the solver's order (longest words first).
The key was chosen against the known anagrams in ``benchmarks/eval_ranking.py``.
"""

import heapq
import math
from collections.abc import Iterable

from granama.vocabulary import Rarity, Vocabulary


def rarity_key(words: tuple[str, ...], rarity: Rarity) -> tuple[float, float]:
    """Sort key of an anagram: (rarest tier, sum of tiers). A word without a tier counts
    as rarer than any tier."""
    tiers = [rarity.get(word, math.inf) for word in words]
    return max(tiers), sum(tiers)


def rank_anagrams(
    anagrams: Iterable[tuple[str, ...]], vocabulary: Vocabulary, limit: int | None = None
) -> list[tuple[str, ...]]:
    """Return ``anagrams`` sorted most plausible first, keeping only the best ``limit``.

    All anagrams are consumed, so pass a search without ``SearchOptions.limit``; with
    ``limit`` only the best ones are kept in memory."""
    rarity = vocabulary.rarity
    if not rarity:
        raise ValueError("ranking requires a vocabulary with rarity data (a bundled dictionary)")

    def key(words: tuple[str, ...]) -> tuple[float, float]:
        return rarity_key(words, rarity)

    if limit is None:
        return sorted(anagrams, key=key)
    return heapq.nsmallest(limit, anagrams, key=key)
