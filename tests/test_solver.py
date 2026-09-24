from collections import Counter

import pytest

from granama.options import SearchOptions
from granama.solver import generate_anagrams
from granama.vocabulary import Vocabulary


def run(text, vocab, **kwargs):
    return list(generate_anagrams(text, vocab, SearchOptions(**kwargs)))


def as_sets(results):
    return {tuple(sorted(r)) for r in results}


def test_finds_known_anagrams(vocab):
    results = as_sets(run("Dirty Room!", vocab))
    assert ("dormitory",) in results
    assert ("dirty", "room") in results


def test_single_word_anagrams(vocab):
    results = as_sets(run("tinsel", vocab, max_words=1))
    assert results == {("enlist",), ("listen",), ("silent",)}


def test_every_result_uses_exactly_the_input_letters(vocab):
    results = run("listen noon", vocab)
    assert results
    for words in results:
        assert Counter("".join(words)) == Counter("listennoon")


def test_no_duplicates_up_to_word_order(vocab):
    results = run("listen noon", vocab)
    keys = [tuple(sorted(r)) for r in results]
    assert len(keys) == len(set(keys))


def test_repeated_signature_is_allowed(vocab):
    assert as_sets(run("aa", vocab)) == {("a", "a")}


def test_repeated_signature_with_several_words(vocab):
    # "ten"/"net" share a signature; picking it twice yields unordered pairs.
    assert as_sets(run("tenten", vocab, max_words=2)) == {
        ("net", "net"),
        ("net", "ten"),
        ("ten", "ten"),
    }


def test_min_and_max_words(vocab):
    for words in run("listen noon", vocab, min_words=2, max_words=3):
        assert 2 <= len(words) <= 3
    assert as_sets(run("dirtyroom", vocab, min_words=2)) == {("dirty", "room")}
    assert as_sets(run("dirtyroom", vocab, max_words=1)) == {("dormitory",)}


def test_min_and_max_word_len(vocab):
    results = run("noon one ten", vocab, min_word_len=3, max_word_len=4)
    assert results
    for words in results:
        assert all(3 <= len(w) <= 4 for w in words)
    assert as_sets(run("noon neon", vocab, min_word_len=3, max_word_len=4)) == {("neon", "noon")}
    assert run("dirtyroom", vocab, min_word_len=6) == [("dormitory",)]


def test_min_word_len_prunes_leftover(vocab):
    # "sit a" would need a 1-letter word, excluded by min_word_len=2
    assert run("sita", vocab, min_word_len=2) == []


def test_limit(vocab):
    assert len(run("listen noon", vocab)) > 2
    assert len(run("listen noon", vocab, limit=2)) == 2


@pytest.mark.parametrize("text", ["", "   ", "123 !?"])
def test_empty_input(vocab, text):
    assert run(text, vocab) == []


def test_impossible_input(vocab):
    assert run("xyz", vocab) == []


def test_deterministic(vocab, english):
    words = ["noon", "listen", "silent", "enlist", "on", "no", "neon", "le", "el", "tin", "one"]
    other = Vocabulary.from_words(list(reversed(words)), english)
    first = Vocabulary.from_words(words, english)
    assert run("listen noon", first) == run("listen noon", other)


def test_longer_words_come_first(vocab):
    first = run("dirtyroom", vocab)[0]
    assert first == ("dormitory",)


def test_default_options(vocab):
    assert ("dormitory",) in list(generate_anagrams("dormitory", vocab))


def _reference(text, vocab, options):
    """Independent reference: plain recursion over individual words with Counters,
    no signatures, packing or word-count bounds; only skips words that don't fit."""
    max_len = options.max_word_len or len(text)
    words = sorted(
        w
        for group in vocab.entries.values()
        for w in group
        if options.min_word_len <= len(w) <= max_len
    )
    found = set()

    def rec(remaining, start, path):
        if not remaining:
            if options.min_words <= len(path) <= (options.max_words or len(path)):
                found.add(tuple(path))
            return
        for i in range(start, len(words)):
            need = Counter(words[i])
            if all(remaining[ch] >= n for ch, n in need.items()):
                rec(remaining - need, i, [*path, words[i]])

    rec(Counter(text), 0, [])
    return found


@pytest.mark.parametrize("text", ["listennoon", "dirtyroom", "tenten", "noonneon", "sitaa"])
@pytest.mark.parametrize(
    ("min_words", "max_words"), [(1, None), (1, 1), (1, 2), (1, 3), (2, None), (2, 2), (2, 3)]
)
@pytest.mark.parametrize(("min_len", "max_len"), [(1, None), (2, None), (1, 3), (3, 6)])
def test_matches_reference(vocab, text, min_words, max_words, min_len, max_len):
    options = SearchOptions(
        min_words=min_words, max_words=max_words, min_word_len=min_len, max_word_len=max_len
    )
    results = list(generate_anagrams(text, vocab, options))
    assert len(results) == len(as_sets(results))
    assert as_sets(results) == _reference(text, vocab, options)
