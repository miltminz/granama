"""Word vocabulary grouped by letter signature, with optional tag-based filtering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from granama.languages import Language, get_language, read_resource
from granama.letters import LetterCounts
from granama.normalize import normalize

# A tag family ("vulgar") excludes all its levels ("vulgar-1", "vulgar-3", ...).
DEFAULT_EXCLUDED_TAGS = frozenset({"offensive", "vulgar"})

type Tags = Mapping[str, frozenset[str]]
type Rarity = Mapping[str, int]


@dataclass(frozen=True)
class Vocabulary:
    """Words of a language, grouped so that mutual anagrams share one signature.

    ``rarity`` maps kept words to their rarity tier (lower = more common); it is empty
    when the source has no rarity data."""

    language: Language
    entries: dict[LetterCounts, tuple[str, ...]]
    rarity: Rarity = field(default_factory=dict)

    @classmethod
    def from_words(
        cls,
        words: Iterable[str],
        language: Language,
        tags: Tags | None = None,
        exclude_tags: Iterable[str] = (),
        exclude_words: Iterable[str] = (),
        rarity: Rarity | None = None,
        max_rarity: int | None = None,
    ) -> Vocabulary:
        """Build a vocabulary, dropping words carrying an excluded tag (see
        ``is_excluded``), listed in ``exclude_words``, or whose ``rarity`` tier
        (lower = more common) exceeds ``max_rarity`` — a word missing from
        ``rarity`` is treated as rarer than any tier. Words are normalized before
        lookup, so ``tags`` and ``rarity`` keys must already be normalized."""
        tags = tags or {}
        rarity = rarity or {}
        exclude_tags = frozenset(exclude_tags)
        banned = {normalize(w, language.alphabet) for w in exclude_words}
        groups: dict[LetterCounts, set[str]] = {}
        kept_rarity: dict[str, int] = {}
        for raw in words:
            word = normalize(raw, language.alphabet)
            if not word or word in banned or is_excluded(tags.get(word, ()), exclude_tags):
                continue
            if max_rarity is not None and rarity.get(word, max_rarity + 1) > max_rarity:
                continue
            signature = LetterCounts.from_word(word, language.alphabet)
            groups.setdefault(signature, set()).add(word)
            if word in rarity:
                kept_rarity[word] = rarity[word]
        entries = {sig: tuple(sorted(ws)) for sig, ws in groups.items()}
        return cls(language, entries, kept_rarity)

    def __len__(self) -> int:
        return sum(len(words) for words in self.entries.values())


def is_excluded(word_tags: Iterable[str], exclude_tags: frozenset[str]) -> bool:
    """True if any tag equals an excluded tag or belongs to an excluded family
    (``"vulgar"`` matches ``"vulgar-3"``)."""
    return any(tag in exclude_tags or tag.split("-", 1)[0] in exclude_tags for tag in word_tags)


def parse_tags(text: str) -> dict[str, frozenset[str]]:
    """Parse ``word<TAB>tag,tag`` lines; ``#`` comments and blank lines are ignored.

    The tag ``none`` gives the word an empty tag set (used to override other sources)."""
    tags: dict[str, set[str]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, _, tag_list = line.partition("\t")
        new = {t.strip() for t in tag_list.split(",") if t.strip()}
        if not new:
            raise ValueError(f"tag line without tags: {line!r}")
        tags.setdefault(word.strip(), set()).update(new - {"none"})
    return {word: frozenset(ts) for word, ts in tags.items()}


def parse_rarity(text: str) -> dict[str, int]:
    """Parse ``word<TAB>tier`` lines (see ``scripts/build_wordlist.py``); ``#`` comments
    and blank lines are ignored. Lower tiers are more common."""
    rarity: dict[str, int] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        word, _, tier = line.partition("\t")
        rarity[word.strip()] = int(tier)
    return rarity


def load_vocabulary(
    lang: str = "en",
    *,
    dictionary: str | None = None,
    path: str | Path | None = None,
    exclude_tags: Iterable[str] = DEFAULT_EXCLUDED_TAGS,
    exclude_words: Iterable[str] = (),
    max_rarity: int | None = None,
) -> Vocabulary:
    """Load a vocabulary for ``lang``.

    ``dictionary`` picks a bundled dictionary (default: the language's default one);
    ``path`` instead loads a custom file with one word per line. By default offensive
    and vulgar words are excluded; pass ``exclude_tags=()`` to keep everything.
    ``max_rarity`` keeps only words at or below that SCOWL rarity tier (see
    ``Dictionary.rarity_resource``; lower is more common) — it requires a bundled
    dictionary that has rarity data, not ``path``.
    Results are cached per argument combination.
    """
    if dictionary is not None and path is not None:
        raise ValueError("pass either dictionary or path, not both")
    if max_rarity is not None and path is not None:
        raise ValueError("max_rarity requires a bundled dictionary, not a custom wordlist")
    language = get_language(lang)
    if path is None:
        # resolve the default so equivalent calls share one cache entry
        resolved = language.get_dictionary(dictionary)
        dictionary = resolved.name
        if max_rarity is not None and not resolved.rarity_resource:
            raise ValueError(f"dictionary {dictionary!r} has no rarity data for max_rarity")
    return _load(
        language,
        dictionary,
        str(path) if path is not None else None,
        frozenset(exclude_tags),
        frozenset(exclude_words),
        max_rarity,
    )


@lru_cache(maxsize=32)
def _load(
    language: Language,
    dictionary: str | None,
    path: str | None,
    exclude_tags: frozenset[str],
    exclude_words: frozenset[str],
    max_rarity: int | None = None,
) -> Vocabulary:
    words, tags, rarity = _read_source(language, dictionary, path)
    return Vocabulary.from_words(
        words, language, tags, exclude_tags, exclude_words, rarity, max_rarity
    )


@lru_cache(maxsize=8)
def _read_source(
    language: Language, dictionary: str | None, path: str | None
) -> tuple[tuple[str, ...], Tags, Rarity]:
    """Words, tags and rarity of a source: the dictionary's tags/rarity, tags
    overridden word by word by the language's curated extra tags."""
    tags: dict[str, frozenset[str]] = {}
    rarity: dict[str, int] = {}
    if path is None:
        source = language.get_dictionary(dictionary)
        words = read_resource(source.words_resource).split()
        if source.tags_resource:
            tags.update(parse_tags(read_resource(source.tags_resource)))
        if source.rarity_resource:
            rarity.update(parse_rarity(read_resource(source.rarity_resource)))
    else:
        words = Path(path).read_text(encoding="utf-8").split()
    if language.extra_tags_resource:
        tags.update(parse_tags(read_resource(language.extra_tags_resource)))
    return tuple(words), tags, rarity
