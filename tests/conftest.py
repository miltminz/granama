import pytest

from granama.languages import get_language
from granama.vocabulary import Vocabulary

FIXTURE_WORDS = [
    "a",
    "i",
    "dirty",
    "room",
    "dormitory",
    "listen",
    "silent",
    "enlist",
    "tin",
    "net",
    "ten",
    "sit",
    "its",
    "le",
    "el",
    "on",
    "no",
    "one",
    "neon",
    "noon",
]


@pytest.fixture
def english():
    return get_language("en")


@pytest.fixture
def vocab(english):
    return Vocabulary.from_words(FIXTURE_WORDS, english)
