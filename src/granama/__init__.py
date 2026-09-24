"""granama: fast multi-word anagram generator."""

from granama.languages import Dictionary, Language, get_language, supported_languages
from granama.options import SearchOptions
from granama.ranking import rank_anagrams
from granama.solver import generate_anagrams
from granama.vocabulary import DEFAULT_EXCLUDED_TAGS, Vocabulary, load_vocabulary

__all__ = [
    "DEFAULT_EXCLUDED_TAGS",
    "Dictionary",
    "Language",
    "SearchOptions",
    "Vocabulary",
    "generate_anagrams",
    "get_language",
    "load_vocabulary",
    "rank_anagrams",
    "supported_languages",
]
