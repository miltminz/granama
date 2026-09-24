"""Registry of supported languages and their bundled dictionaries."""

from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Dictionary:
    """A bundled word list. Resource paths are relative to the ``granama.data`` package."""

    name: str
    description: str
    words_resource: str
    tags_resource: str | None = None
    rarity_resource: str | None = None


@dataclass(frozen=True)
class Language:
    code: str
    alphabet: str
    dictionaries: tuple[Dictionary, ...]
    default_dictionary: str
    # Curated tags applied on top of any dictionary of this language, custom files included.
    extra_tags_resource: str | None = None

    def get_dictionary(self, name: str | None = None) -> Dictionary:
        """Return the dictionary called ``name``, or the default one if ``name`` is None."""
        name = name or self.default_dictionary
        for dictionary in self.dictionaries:
            if dictionary.name == name:
                return dictionary
        available = ", ".join(d.name for d in self.dictionaries)
        raise ValueError(f"unknown dictionary {name!r} for {self.code!r} (available: {available})")


def read_resource(path: str) -> str:
    return resources.files("granama.data").joinpath(path).read_text(encoding="utf-8")


_LANGUAGES: dict[str, Language] = {
    "en": Language(
        code="en",
        alphabet="abcdefghijklmnopqrstuvwxyz",
        dictionaries=(
            Dictionary(
                name="scowl-60",
                description="SCOWL v2 (ESDB) size 60, American spelling, ~79k words",
                words_resource="en/scowl-60.txt",
                tags_resource="en/scowl-60.tags.tsv",
                rarity_resource="en/scowl-60.rarity.tsv",
            ),
        ),
        default_dictionary="scowl-60",
        extra_tags_resource="en/extra-tags.tsv",
    ),
}


def get_language(code: str) -> Language:
    try:
        return _LANGUAGES[code]
    except KeyError:
        supported = ", ".join(sorted(_LANGUAGES))
        raise ValueError(f"unsupported language {code!r} (supported: {supported})") from None


def supported_languages() -> list[str]:
    return sorted(_LANGUAGES)
