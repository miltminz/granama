import pytest

from granama.letters import LetterCounts, LetterPacking


def lc(word, alphabet="abc"):
    return LetterCounts.from_word(word, alphabet)


def test_from_word_counts_and_total():
    counts = lc("abca")
    assert counts.counts == (2, 1, 1)
    assert counts.total == 4


def test_from_word_rejects_unknown_char():
    with pytest.raises(ValueError, match="'z'"):
        lc("az")


def test_anagrams_are_equal_and_hash_equal():
    assert lc("abc") == lc("cab")
    assert hash(lc("abc")) == hash(lc("cab"))
    assert lc("abc") != lc("abcc")


def test_fits_in():
    assert lc("ab").fits_in(lc("abc"))
    assert lc("").fits_in(lc("a"))
    assert lc("abc").fits_in(lc("abc"))
    assert not lc("aa").fits_in(lc("abc"))


def test_subtract():
    rest = lc("aabc") - lc("ab")
    assert rest == lc("ac")
    assert rest.total == 2


def test_subtract_unavailable_raises():
    with pytest.raises(ValueError):
        lc("a") - lc("b")


def test_is_empty():
    assert lc("").is_empty()
    assert (lc("ab") - lc("ba")).is_empty()
    assert not lc("a").is_empty()


def test_packing_width_fits_max_count():
    packing = LetterPacking.for_target(lc("aaab"))  # max count 3 -> 2 bits + guard
    assert packing.width == 3
    assert packing.guard == 0b100_100_100


def test_packing_for_empty_target():
    packing = LetterPacking.for_target(lc(""))
    assert packing.width == 1
    assert packing.pack(lc("")) == 0


def test_pack_layout():
    packing = LetterPacking.for_target(lc("aaab"))
    assert packing.pack(lc("aaabcc")) == 0b010_001_011


def test_pack_rejects_overflowing_count():
    packing = LetterPacking.for_target(lc("ab"))  # 1 bit per count
    with pytest.raises(ValueError, match="does not fit"):
        packing.pack(lc("aa"))


@pytest.mark.parametrize(
    ("word", "remaining"),
    [("", ""), ("", "abc"), ("a", "a"), ("ab", "abc"), ("abc", "cab"), ("aa", "aaab")],
)
def test_packed_fits(word, remaining):
    packing = LetterPacking.for_target(lc("aaabbbccc"))
    assert packing.fits(packing.pack(lc(word)), packing.pack(lc(remaining)))


@pytest.mark.parametrize(
    ("word", "remaining"),
    [("a", ""), ("aa", "abc"), ("c", "ab"), ("abcc", "abc"), ("aaa", "aab")],
)
def test_packed_does_not_fit(word, remaining):
    packing = LetterPacking.for_target(lc("aaabbbccc"))
    assert not packing.fits(packing.pack(lc(word)), packing.pack(lc(remaining)))


def test_packed_fits_matches_fits_in_exhaustively():
    target = lc("aaabbc")
    packing = LetterPacking.for_target(target)
    counts = [LetterCounts((a, b, c)) for a in range(4) for b in range(3) for c in range(2)]
    for word in counts:
        for remaining in counts:
            assert packing.fits(packing.pack(word), packing.pack(remaining)) == word.fits_in(
                remaining
            ), (word, remaining)
            if word.fits_in(remaining):
                assert packing.pack(remaining) - packing.pack(word) == packing.pack(
                    remaining - word
                )
