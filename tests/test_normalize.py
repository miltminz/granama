from granama.normalize import normalize

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def test_lowercases():
    assert normalize("Dirty ROOM", ALPHABET) == "dirtyroom"


def test_strips_diacritics():
    assert normalize("Café Naïve", ALPHABET) == "cafenaive"


def test_drops_punctuation_digits_and_spaces():
    assert normalize("it's 4 you, ok?!", ALPHABET) == "itsyouok"


def test_drops_letters_outside_alphabet():
    assert normalize("abcxyz", "abc") == "abc"


def test_empty():
    assert normalize("", ALPHABET) == ""
