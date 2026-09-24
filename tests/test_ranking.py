import math

import pytest

from granama.ranking import rank_anagrams, rarity_key
from granama.vocabulary import Vocabulary, load_vocabulary

RARITY = {"dirty": 35, "room": 35, "moor": 60, "dormitory": 50, "rotor": 35, "id": 50, "my": 35}


@pytest.fixture
def ranked_vocab(english):
    return Vocabulary.from_words(RARITY, english, rarity=RARITY)


def test_rarity_key_is_rarest_then_sum():
    assert rarity_key(("dirty", "room"), RARITY) == (35, 70)
    assert rarity_key(("rotor", "id", "my"), RARITY) == (50, 120)
    assert rarity_key(("dirty", "unknown"), RARITY) == (math.inf, math.inf)


def test_ranks_by_rarest_word_then_sum(ranked_vocab):
    anagrams = [("dirty", "moor"), ("rotor", "id", "my"), ("dormitory",), ("dirty", "room")]
    assert rank_anagrams(anagrams, ranked_vocab) == [
        ("dirty", "room"),
        ("dormitory",),  # (50, 50) before (50, 120)
        ("rotor", "id", "my"),
        ("dirty", "moor"),
    ]


def test_ties_keep_input_order(ranked_vocab):
    anagrams = [("room", "dirty"), ("dirty", "room")]
    assert rank_anagrams(anagrams, ranked_vocab) == anagrams


def test_limit_keeps_the_best(ranked_vocab):
    anagrams = iter([("dirty", "moor"), ("dormitory",), ("dirty", "room")])
    assert rank_anagrams(anagrams, ranked_vocab, limit=2) == [("dirty", "room"), ("dormitory",)]


def test_requires_rarity_data(vocab):
    with pytest.raises(ValueError, match="rarity data"):
        rank_anagrams([("dirty", "room")], vocab)


def test_bundled_vocabulary_has_rarity():
    vocab = load_vocabulary("en")
    assert vocab.rarity["room"] == 35
    assert set(vocab.rarity) == {w for ws in vocab.entries.values() for w in ws}
