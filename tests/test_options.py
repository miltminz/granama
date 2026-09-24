import pytest

from granama.options import SearchOptions


def test_defaults_are_valid():
    opts = SearchOptions()
    assert (opts.min_words, opts.max_words, opts.limit) == (1, None, None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_words": 0}, "min_words must be >= 1"),
        ({"min_words": 3, "max_words": 2}, "max_words must be >= min_words"),
        ({"min_word_len": 0}, "min_word_len must be >= 1"),
        ({"min_word_len": 4, "max_word_len": 3}, "max_word_len must be >= min_word_len"),
        ({"limit": 0}, "limit must be >= 1"),
    ],
)
def test_invalid(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SearchOptions(**kwargs)


def test_equal_bounds_allowed():
    SearchOptions(min_words=2, max_words=2, min_word_len=3, max_word_len=3)
