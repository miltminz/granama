"""Search constraints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchOptions:
    """Constraints on generated anagrams. ``None`` means unbounded."""

    min_words: int = 1
    max_words: int | None = None
    min_word_len: int = 1
    max_word_len: int | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        _check_range("words", self.min_words, self.max_words)
        _check_range("word_len", self.min_word_len, self.max_word_len)
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be >= 1")


def _check_range(name: str, low: int, high: int | None) -> None:
    if low < 1:
        raise ValueError(f"min_{name} must be >= 1")
    if high is not None and high < low:
        raise ValueError(f"max_{name} must be >= min_{name}")
