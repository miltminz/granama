import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_wordlist.py"
spec = importlib.util.spec_from_file_location("build_wordlist", SCRIPT)
assert spec and spec.loader
build_wordlist = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_wordlist)


def test_clean_word():
    assert build_wordlist.clean_word(" café\n") == "cafe"
    assert build_wordlist.clean_word("a") == "a"
    for raw in ["Paris", "don't", "cat's", "e-mail", "ice cream", "etc.", "b", "'tis", ""]:
        assert build_wordlist.clean_word(raw) is None, raw


def test_filter_words_keeps_plain_lowercase_words():
    raw = ["room", "Paris", "don't", "e-mail", "  dirty \n", "room", "b", "a", "café"]
    assert build_wordlist.filter_words(raw) == ["a", "cafe", "dirty", "i", "room"]


def test_collect_tags_merges_notes_and_skips_untagged_or_dropped():
    rows = [
        ("crap", "vulgar-3"),
        ("crap", ""),
        ("crap", "informal"),
        ("room", ""),
        ("Crap", "vulgar-3"),  # proper noun, dropped
    ]
    assert build_wordlist.collect_tags(rows) == {"crap": ["informal", "vulgar-3"]}


def test_collect_tiers_keeps_the_smallest_size_and_skips_dropped():
    rows = [("room", 60), ("room", 35), ("room", 50), ("Paris", 35), ("dirty", 40)]
    assert build_wordlist.collect_tiers(rows) == {"dirty": 40, "room": 35}
