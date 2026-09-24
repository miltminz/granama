import pytest

from granama.languages import Dictionary, Language
from granama.letters import LetterCounts
from granama.vocabulary import (
    DEFAULT_EXCLUDED_TAGS,
    Vocabulary,
    _read_source,
    is_excluded,
    load_vocabulary,
    parse_rarity,
    parse_tags,
)


def words_of(vocab):
    return {w for ws in vocab.entries.values() for w in ws}


def test_groups_anagrams_under_one_signature(vocab, english):
    sig = LetterCounts.from_word("listen", english.alphabet)
    assert vocab.entries[sig] == ("enlist", "listen", "silent")


def test_normalizes_dedupes_and_skips_empty(english):
    vocab = Vocabulary.from_words(["Room", "room", "rööm", "123", ""], english)
    assert list(vocab.entries.values()) == [("room",)]
    assert len(vocab) == 1


def test_len_counts_words(vocab):
    assert len(vocab) == 20


# --- tags -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word_tags", "excluded", "expected"),
    [
        ({"vulgar-3"}, {"vulgar"}, True),  # family covers every level
        ({"vulgar-3"}, {"vulgar-3"}, True),
        ({"vulgar-3"}, {"vulgar-1"}, False),
        ({"offensive-1"}, {"vulgar"}, False),
        ({"informal", "vulgar-1"}, {"vulgar"}, True),
        (set(), {"vulgar"}, False),
        ({"vulgar-1"}, set(), False),
    ],
)
def test_is_excluded(word_tags, excluded, expected):
    assert is_excluded(word_tags, frozenset(excluded)) is expected


def test_parse_tags():
    text = "# comment\n\nfoo\tvulgar-3\nbar\tinformal, vulgar-1\nfoo\tinformal\nbaz\tnone\n"
    assert parse_tags(text) == {
        "foo": {"vulgar-3", "informal"},
        "bar": {"informal", "vulgar-1"},
        "baz": frozenset(),
    }


def test_parse_tags_rejects_line_without_tags():
    with pytest.raises(ValueError, match="without tags"):
        parse_tags("foo\n")


def test_from_words_excludes_tagged_and_listed_words(english):
    tags = {"foo": frozenset({"vulgar-3"}), "bar": frozenset({"informal"})}
    words = ["foo", "bar", "baz", "Qux"]
    assert words_of(Vocabulary.from_words(words, english, tags)) == {"foo", "bar", "baz", "qux"}
    kept = Vocabulary.from_words(words, english, tags, exclude_tags={"vulgar"})
    assert words_of(kept) == {"bar", "baz", "qux"}
    kept = Vocabulary.from_words(words, english, tags, exclude_words=["BAZ", "qux"])
    assert words_of(kept) == {"foo", "bar"}


# --- loading ----------------------------------------------------------------


def test_load_custom_file(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("tin\nnit\nfoo\n", encoding="utf-8")
    vocab = load_vocabulary("en", path=path)
    assert sorted(vocab.entries.values()) == [("foo",), ("nit", "tin")]


def test_custom_file_gets_curated_language_tags(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("tosser\ntin\n", encoding="utf-8")  # tosser: vulgar-3 in extra-tags
    assert words_of(load_vocabulary("en", path=path)) == {"tin"}
    assert words_of(load_vocabulary("en", path=path, exclude_tags=())) == {"tosser", "tin"}


def test_dictionary_and_path_are_exclusive(tmp_path):
    with pytest.raises(ValueError, match="either dictionary or path"):
        load_vocabulary("en", dictionary="scowl-60", path=tmp_path / "x.txt")


def test_unknown_dictionary():
    with pytest.raises(ValueError, match="unknown dictionary"):
        load_vocabulary("en", dictionary="nope")


def test_load_bundled_is_cached():
    assert load_vocabulary("en") is load_vocabulary("en", dictionary="scowl-60")
    assert load_vocabulary("en") is load_vocabulary("en", exclude_tags=DEFAULT_EXCLUDED_TAGS)
    assert load_vocabulary("en") is not load_vocabulary("en", exclude_tags=())


def test_bundled_default_filters_offensive_and_vulgar():
    everything = words_of(load_vocabulary("en", exclude_tags=()))
    default = words_of(load_vocabulary("en"))
    only_offensive = words_of(load_vocabulary("en", exclude_tags=["offensive"]))
    assert default < only_offensive < everything
    # ESDB tag (vulgar-3) and curated tag (vulgar-3) are both applied
    assert {"crap", "fart", "tosser"} <= everything - default
    assert {"crap", "fart", "tosser"} <= only_offensive
    # curated "none" overrides an ESDB false positive
    assert {"dicker", "dickens"} <= default
    assert {"dirty", "room"} <= default


def test_read_source_without_tag_resources(tmp_path):
    bare = Language(
        code="en",
        alphabet="abcdefghijklmnopqrstuvwxyz",
        dictionaries=(Dictionary("plain", "no tags", "en/scowl-60.txt"),),
        default_dictionary="plain",
    )
    words, tags, rarity = _read_source(bare, "plain", None)
    assert "room" in words
    assert tags == {}
    assert rarity == {}
    path = tmp_path / "words.txt"
    path.write_text("foo bar\n", encoding="utf-8")
    assert _read_source(bare, None, str(path)) == (("foo", "bar"), {}, {})


# --- rarity --------------------------------------------------------------------


def test_parse_rarity():
    text = "# comment\n\nfoo\t35\nbar\t60\n"
    assert parse_rarity(text) == {"foo": 35, "bar": 60}


def test_from_words_drops_words_above_max_rarity(english):
    rarity = {"foo": 35, "bar": 60}
    words = ["foo", "bar", "baz"]  # baz: missing from rarity
    assert words_of(Vocabulary.from_words(words, english, rarity=rarity)) == {
        "foo",
        "bar",
        "baz",
    }
    kept = Vocabulary.from_words(words, english, rarity=rarity, max_rarity=60)
    assert words_of(kept) == {"foo", "bar"}  # baz dropped: missing tier treated as too rare
    kept = Vocabulary.from_words(words, english, rarity=rarity, max_rarity=35)
    assert words_of(kept) == {"foo"}


def test_from_words_keeps_rarity_of_kept_words(english):
    rarity = {"foo": 35, "bar": 60}
    vocab = Vocabulary.from_words(["foo", "bar", "baz"], english, rarity=rarity, max_rarity=35)
    assert vocab.rarity == {"foo": 35}
    assert Vocabulary.from_words(["foo"], english).rarity == {}


def test_max_rarity_requires_bundled_dictionary_with_rarity_data(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("room\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires a bundled dictionary"):
        load_vocabulary("en", path=path, max_rarity=35)


def test_max_rarity_requires_rarity_resource(monkeypatch):
    bare = Language(
        code="en",
        alphabet="abcdefghijklmnopqrstuvwxyz",
        dictionaries=(Dictionary("plain", "no rarity data", "en/scowl-60.txt"),),
        default_dictionary="plain",
    )
    monkeypatch.setattr("granama.vocabulary.get_language", lambda code: bare)
    with pytest.raises(ValueError, match="no rarity data"):
        load_vocabulary("en", dictionary="plain", max_rarity=35)


def test_max_rarity_filters_bundled_vocabulary():
    everything = words_of(load_vocabulary("en", exclude_tags=()))
    common = words_of(load_vocabulary("en", exclude_tags=(), max_rarity=35))
    assert 0 < len(common) < len(everything)
    assert {"dirty", "room"} <= common
