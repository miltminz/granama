"""Anagram search.

The search runs over letter *signatures* (groups of mutual anagrams) rather than
individual words, and signatures are chosen in a fixed canonical order so each
unordered combination is produced exactly once. Concrete word tuples are only
materialized at the end, by expanding each signature into its words.

Signatures are encoded with ``LetterPacking`` so the hot fit test is a single int
operation (see ``benchmarks/`` for measurements).
"""

from collections.abc import Iterator
from itertools import combinations_with_replacement, groupby, islice, product

from granama.letters import LetterCounts, LetterPacking
from granama.normalize import normalize
from granama.options import SearchOptions
from granama.vocabulary import Vocabulary

# (packed letters, letter count, signature)
type _Candidate = tuple[int, int, LetterCounts]


def generate_anagrams(
    text: str, vocabulary: Vocabulary, options: SearchOptions | None = None
) -> Iterator[tuple[str, ...]]:
    """Yield anagrams of ``text`` as tuples of words, each unordered set once."""
    options = options or SearchOptions()
    alphabet = vocabulary.language.alphabet
    target = LetterCounts.from_word(normalize(text, alphabet), alphabet)
    if target.is_empty():
        return

    max_len = options.max_word_len or target.total
    signatures = [
        sig
        for sig in vocabulary.entries
        if options.min_word_len <= sig.total <= max_len and sig.fits_in(target)
    ]
    # Longest first: big words consume letters fastest and prune the tree early.
    # The max_words bound in _Search also relies on this order.
    signatures.sort(key=lambda sig: (-sig.total, vocabulary.entries[sig]))

    packing = LetterPacking.for_target(target)
    candidates = [(packing.pack(sig), sig.total, sig) for sig in signatures]
    search = _Search(packing.guard, options, max_len)
    solutions = search.run(packing.pack(target), target.total, candidates)
    results = (words for sigs in solutions for words in _expand(sigs, vocabulary.entries))
    yield from islice(results, options.limit)


class _Search:
    def __init__(self, guard: int, options: SearchOptions, max_len: int) -> None:
        self.guard = guard
        self.options = options
        self.max_len = max_len
        self.path: list[LetterCounts] = []

    def run(
        self, remaining: int, total: int, candidates: list[_Candidate]
    ) -> Iterator[tuple[LetterCounts, ...]]:
        options, path = self.options, self.path
        if total == 0:
            if len(path) >= options.min_words:
                yield tuple(path)
            return
        if total < options.min_word_len:
            return
        words_left = None if options.max_words is None else options.max_words - len(path)
        if words_left is not None and total > words_left * self.max_len:
            return

        # Inlined LetterPacking.fits: this is the hot loop.
        guard = self.guard
        with_guard = remaining | guard
        fitting = [c for c in candidates if (with_guard - c[0]) & guard == guard]
        for i, (packed, length, sig) in enumerate(fitting):
            # Later words come from fitting[i:], so none is longer than `length`.
            # Lengths only decrease along the list, so once this fails it fails for all.
            if words_left is not None and total - length > (words_left - 1) * length:
                break
            path.append(sig)
            # Only signatures at index >= i: canonical order, no permuted duplicates.
            yield from self.run(remaining - packed, total - length, fitting[i:])
            path.pop()


def _expand(
    signatures: tuple[LetterCounts, ...], entries: dict[LetterCounts, tuple[str, ...]]
) -> Iterator[tuple[str, ...]]:
    """Turn a signature combination into every distinct word combination."""
    # Canonical order keeps repeated signatures adjacent.
    per_group = [
        list(combinations_with_replacement(entries[sig], len(list(run))))
        for sig, run in groupby(signatures)
    ]
    for choice in product(*per_group):
        yield tuple(word for group in choice for word in group)
