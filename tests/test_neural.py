import os
import sys

import pytest

from granama.neural import CausalLMFluency, EmbeddingRelevance, NeuralRanker
from granama.neural.models import MISSING_EXTRA
from granama.neural.ranker import count_orders, zscores
from granama.vocabulary import Vocabulary, load_vocabulary


class FakeFluency:
    """log_prob from a table; unknown phrases are very unlikely."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def log_prob(self, phrases, context="", complete=True):
        self.calls.append(list(phrases))
        return [self.table.get(p, -100.0) for p in phrases]


class FakeRelevance:
    def __init__(self, table):
        self.table = table

    def similarity(self, source, phrases):
        return [self.table.get(p, 0.0) for p in phrases]


FLUENCY = {"dirty room": -5, "room dirty": -9, "dirty moor": -8, "moor dirty": -10}
ANAGRAMS = [("dormitory",), ("dirty", "room"), ("dirty", "moor")]


def ranker(fluency=FLUENCY, relevance=None, **kwargs):
    return NeuralRanker(FakeFluency(fluency), FakeRelevance(relevance or {}), **kwargs)


def test_ranks_by_fluency_in_the_best_word_order(vocab):
    ranked = ranker().rank("dormitory", ANAGRAMS, vocab)
    assert [a.phrase for a in ranked] == ["dirty room", "dirty moor"]
    assert ranked[0].words == ("dirty", "room")
    assert ranked[0].fluency == -5


def test_leaves_out_the_input_words(vocab):
    ranked = ranker().rank("Room, DIRTY!", [("room", "dirty"), ("dormitory",)], vocab)
    assert [a.phrase for a in ranked] == ["dormitory"]


def test_relevance_weight(vocab):
    relevance = {"dirty room": 0.1, "dirty moor": 0.9}
    fluent_first = ranker(relevance=relevance, relevance_weight=0)
    assert fluent_first.rank("dormitory", ANAGRAMS, vocab)[0].phrase == "dirty room"
    relevant_first = ranker(relevance=relevance, relevance_weight=10)
    assert relevant_first.rank("dormitory", ANAGRAMS, vocab)[0].phrase == "dirty moor"


def test_pool_keeps_the_best_first_pass(vocab):
    r = ranker(relevance={"dirty moor": 0.9}, pool=1)
    assert [a.phrase for a in r.rank("dormitory", ANAGRAMS, vocab)] == ["dirty moor"]
    assert r.fluency.calls == [["dirty moor", "moor dirty"]]


def test_first_pass_prefers_common_words(english):
    rarity = {"dirty": 35, "room": 35, "moor": 60, "dormitory": 50}
    vocab = Vocabulary.from_words(rarity, english, rarity=rarity)
    ranked = ranker(pool=1).rank("dormitory", ANAGRAMS, vocab)
    assert [a.phrase for a in ranked] == ["dirty room"]


def test_rarity_weight(english):
    rarity = {"dirty": 35, "room": 35, "moor": 60, "dormitory": 50}
    vocab = Vocabulary.from_words(rarity, english, rarity=rarity)
    relevance = {"dirty moor": 0.9}
    ignored = ranker(relevance=relevance, pool=1, rarity_weight=0)
    assert [a.phrase for a in ignored.rank("dormitory", ANAGRAMS, vocab)] == ["dirty moor"]
    heavy = ranker(relevance=relevance, pool=1, rarity_weight=2)
    assert [a.phrase for a in heavy.rank("dormitory", ANAGRAMS, vocab)] == ["dirty room"]


def test_limit(vocab):
    assert len(ranker().rank("dormitory", ANAGRAMS, vocab, limit=1)) == 1


def test_max_orders_keeps_the_best_orders_under_pair_scores(vocab):
    # word starts and pairs: "c" starts best, then "c a", then "a b"
    pairs = {"a": -3, "b": -3, "c": -1, "c a": -2, "a b": -3, "c b": -5, "b a": -6}
    r = ranker({**pairs, "c a b": -4, "d e": -1}, max_orders=2)
    ranked = r.rank("x", [("a", "b", "c"), ("d", "e")], vocab)
    prefixes, full = r.fluency.calls  # words and pairs are scored in one call
    assert prefixes == ["a", "b", "c", "a b", "a c", "b a", "b c", "c a", "c b"]
    assert full == ["c a b", "c b a", "d e", "e d"]  # every order of the 2-word anagram
    assert ranked[1].phrase == "c a b"


def test_repeated_words(vocab):
    r = ranker({"a": -1, "a b": -1, "b a": -9, "a a": -2, "a b a": -1}, max_orders=2)
    assert r.rank("x", [("a", "b", "a")], vocab)[0].phrase == "a b a"
    assert r.fluency.calls[-1] == ["a b a", "a a b"]


@pytest.mark.parametrize(
    ("words", "count"), [((), 1), (("a", "b", "c"), 6), (("a", "b", "a"), 3), (("a", "a"), 1)]
)
def test_count_orders(words, count):
    assert count_orders(words) == count


def test_no_candidates(vocab):
    assert ranker().rank("dormitory", [("dormitory",)], vocab) == []


def test_zscores():
    assert zscores([1.0, 3.0]) == [-1.0, 1.0]
    assert zscores([2.0, 2.0]) == [0.0, 0.0]


@pytest.mark.parametrize("model", [CausalLMFluency, EmbeddingRelevance])
def test_models_need_the_lm_extra(monkeypatch, model):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"granama\[lm\]"):
        model()
    assert "granama[lm]" in MISSING_EXTRA


@pytest.mark.skipif(
    not os.environ.get("GRANAMA_TEST_MODELS"),
    reason="downloads models; set GRANAMA_TEST_MODELS=1 (needs the lm extra)",
)
def test_default_models_find_dirty_room():
    from granama import generate_anagrams

    vocabulary = load_vocabulary("en")
    ranked = NeuralRanker(CausalLMFluency(), EmbeddingRelevance()).rank(
        "dormitory", generate_anagrams("dormitory", vocabulary), vocabulary, limit=3
    )
    assert "dirty room" in [a.phrase for a in ranked]
