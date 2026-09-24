import pytest

from granama.languages import get_language, read_resource, supported_languages


def test_english_registered():
    en = get_language("en")
    assert en.alphabet == "abcdefghijklmnopqrstuvwxyz"
    assert "en" in supported_languages()


def test_unknown_language():
    with pytest.raises(ValueError, match="unsupported language 'xx'"):
        get_language("xx")


def test_default_dictionary():
    en = get_language("en")
    assert en.get_dictionary().name == en.default_dictionary == "scowl-60"
    assert en.get_dictionary("scowl-60") is en.get_dictionary()


def test_unknown_dictionary():
    with pytest.raises(ValueError, match=r"unknown dictionary 'nope' for 'en'.*scowl-60"):
        get_language("en").get_dictionary("nope")


@pytest.mark.parametrize("dictionary", get_language("en").dictionaries, ids=lambda d: d.name)
def test_bundled_resources_exist(dictionary):
    words = read_resource(dictionary.words_resource).split()
    assert len(words) > 10_000
    assert {"a", "i", "the", "dirty", "room", "listen"} <= set(words)
    assert words == sorted(set(words))
    if dictionary.tags_resource:
        assert read_resource(dictionary.tags_resource)


def test_extra_tags_resource_exists():
    assert "offensive-1" in read_resource(get_language("en").extra_tags_resource)
