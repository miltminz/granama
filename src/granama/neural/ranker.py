"""Rank anagrams by fluency (a language model) and relevance to the input (an embedding model).

The pipeline, measured on the known anagrams of ``benchmarks/eval_lm.py``:

1. First pass, over every anagram: z(relevance) + z(commonness of the rarest word). It is
   cheap enough for large outputs; with the default models it keeps the known answers in
   the top ~400.
2. The best ``pool`` anagrams are scored for fluency in up to ``max_orders`` word orders
   each; the most fluent order is kept. Anagrams with more orders than that (n words have
   up to n! of them) get their most promising orders under word-pair scores: the language
   model scores each word as a start and each ordered pair of words once per input, and an
   order's estimate is its first word's score plus the scores of its consecutive pairs.
3. Final score over the pool: z(fluency) + ``relevance_weight`` * z(relevance).

z-scores are taken over the anagrams of one input, so the weights do not depend on the
scale of each model's scores. The models are pluggable: anything implementing ``Fluency``
and ``Relevance`` works (see ``granama.neural.models`` for Hugging Face implementations).
This module is pure Python; only the model implementations need the ``lm`` extra.
"""

import heapq
import itertools
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from granama.normalize import normalize
from granama.vocabulary import Vocabulary


class Fluency(Protocol):
    def log_prob(
        self, phrases: Sequence[str], context: str = "", complete: bool = True
    ) -> Sequence[float]:
        """Log-probability of each phrase as English text (higher is more fluent),
        optionally conditioned on a ``context`` that precedes it. With ``complete=False``
        a phrase is scored as the start of a text rather than a whole one."""
        ...


class Relevance(Protocol):
    def similarity(self, source: str, phrases: Sequence[str]) -> Sequence[float]:
        """Similarity in meaning of each phrase to ``source`` (higher is closer)."""
        ...


@dataclass(frozen=True)
class RankedAnagram:
    words: tuple[str, ...]  # as generated
    phrase: str  # the most fluent order of ``words``
    score: float  # z(fluency) + relevance_weight * z(relevance)
    fluency: float  # log-probability of ``phrase``
    relevance: float  # similarity to the input


@dataclass
class NeuralRanker:
    fluency: Fluency
    relevance: Relevance
    relevance_weight: float = 0.5
    pool: int = 1000
    rarity_weight: float = 1.0  # how much rare words count against an anagram in the
    # first pass (0 ignores rarity, negative favors rare words); needs rarity data
    max_orders: int = 20  # word orders scored per anagram (all of them up to 3 words)

    def rank(
        self,
        text: str,
        anagrams: Iterable[tuple[str, ...]],
        vocabulary: Vocabulary,
        limit: int | None = None,
    ) -> list[RankedAnagram]:
        """Return the best anagrams of ``text``, best first: at most ``pool`` of them, or
        ``limit``. The input's own words (in any order) are left out."""
        own = sorted(
            w for w in (normalize(t, vocabulary.language.alphabet) for t in text.split()) if w
        )
        candidates = [words for words in anagrams if sorted(words) != own]
        if not candidates:
            return []

        relevance = list(self.relevance.similarity(text, [" ".join(w) for w in candidates]))
        first = zscores(relevance)
        if vocabulary.rarity and self.rarity_weight:
            rarity = vocabulary.rarity
            unknown = max(rarity.values()) + 1
            common = [-max(rarity.get(w, unknown) for w in words) for words in candidates]
            first = [
                f + self.rarity_weight * c for f, c in zip(first, zscores(common), strict=True)
            ]
        pool = sorted(range(len(candidates)), key=lambda i: -first[i])[: self.pool]

        orders, owners = [], []
        for i, phrases in zip(pool, self._orders([candidates[i] for i in pool]), strict=True):
            orders += phrases
            owners += [i] * len(phrases)
        best: dict[int, tuple[float, str]] = {}
        for order, i, lp in zip(orders, owners, self.fluency.log_prob(orders), strict=True):
            if i not in best or lp > best[i][0]:
                best[i] = (lp, order)

        fluency = zscores([best[i][0] for i in pool])
        pooled_relevance = zscores([relevance[i] for i in pool])
        ranked = [
            RankedAnagram(
                words=candidates[i],
                phrase=best[i][1],
                score=f + self.relevance_weight * r,
                fluency=best[i][0],
                relevance=relevance[i],
            )
            for i, f, r in zip(pool, fluency, pooled_relevance, strict=True)
        ]
        ranked.sort(key=lambda a: -a.score)  # stable: ties keep the first-pass order
        return ranked[:limit]

    def _orders(self, anagrams: list[tuple[str, ...]]) -> list[list[str]]:
        """The word orders to score for each anagram: all of them, or the ``max_orders``
        best under word-pair scores."""
        long = [words for words in anagrams if count_orders(words) > self.max_orders]
        start: dict[str, float] = {}
        follow: dict[tuple[str, str], float] = {}
        if long:
            singles = sorted({w for words in long for w in words})
            pairs = sorted({p for words in long for p in itertools.permutations(words, 2)})
            lps = self.fluency.log_prob(singles + [f"{a} {b}" for a, b in pairs], complete=False)
            start = dict(zip(singles, lps[: len(singles)], strict=True))
            follow = {
                (a, b): lp - start[a] for (a, b), lp in zip(pairs, lps[len(singles) :], strict=True)
            }
        return [
            best_orders(words, start, follow, self.max_orders)
            if count_orders(words) > self.max_orders
            else list(dict.fromkeys(" ".join(p) for p in itertools.permutations(words)))
            for words in anagrams
        ]


def count_orders(words: Sequence[str]) -> int:
    """Number of distinct orders of ``words``."""
    return math.factorial(len(words)) // math.prod(map(math.factorial, Counter(words).values()))


ORDER_BEAM = 1000  # partial orders kept per step: exact up to 6 words (720 orders)


def best_orders(
    words: Sequence[str],
    start: dict[str, float],
    follow: dict[tuple[str, str], float],
    n: int,
) -> list[str]:
    """The ``n`` orders of ``words`` with the best word-pair scores, best first, found by a
    beam search over prefixes (exact while no step has more than ``ORDER_BEAM`` of them)."""
    beam: dict[tuple[str, ...], float] = {(): 0.0}
    for _ in words:
        grown = {}
        for prefix, score in beam.items():
            rest = Counter(words)
            rest.subtract(prefix)
            for w in sorted(w for w, left in rest.items() if left):
                step = follow[prefix[-1], w] if prefix else start[w]
                grown[(*prefix, w)] = score + step
        beam = dict(heapq.nlargest(ORDER_BEAM, grown.items(), key=lambda item: item[1]))
    return [" ".join(order) for order in heapq.nlargest(n, beam, key=beam.__getitem__)]


def zscores(values: Sequence[float]) -> list[float]:
    """Standardize ``values`` to mean 0 and standard deviation 1 (all 0 if constant)."""
    mean = statistics.fmean(values)
    std = statistics.pstdev(values, mean)
    if not std or not math.isfinite(std):
        return [0.0] * len(values)
    return [(v - mean) / std for v in values]
