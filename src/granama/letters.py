"""Immutable letter multisets over a fixed alphabet."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class LetterCounts:
    """Count of each alphabet letter, stored as a tuple indexed by alphabet position.

    Two words are anagrams of each other iff their ``LetterCounts`` are equal.
    """

    counts: tuple[int, ...]
    total: int = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total", sum(self.counts))

    @classmethod
    def from_word(cls, word: str, alphabet: str) -> LetterCounts:
        """Count letters of an already-normalized ``word``.

        Raises ``ValueError`` if ``word`` contains a character outside ``alphabet``.
        """
        index = {ch: i for i, ch in enumerate(alphabet)}
        counts = [0] * len(alphabet)
        for ch in word:
            try:
                counts[index[ch]] += 1
            except KeyError:
                raise ValueError(f"character {ch!r} not in alphabet") from None
        return cls(tuple(counts))

    def fits_in(self, other: LetterCounts) -> bool:
        """True if every letter of ``self`` is available in ``other``."""
        return all(a <= b for a, b in zip(self.counts, other.counts, strict=True))

    def __sub__(self, other: LetterCounts) -> LetterCounts:
        if not other.fits_in(self):
            raise ValueError("cannot subtract: letters not available")
        return LetterCounts(tuple(a - b for a, b in zip(self.counts, other.counts, strict=True)))

    def is_empty(self) -> bool:
        return self.total == 0


@dataclass(frozen=True, slots=True)
class LetterPacking:
    """Encodes ``LetterCounts`` as a single int, for fast fit tests in the solver.

    Each letter gets a ``width``-bit field: the low bits hold the count, the top bit is
    a guard. Subtracting ``word`` from ``remaining | guard`` borrows from (clears) a
    field's guard bit exactly when that letter is short, so a fit test is one
    subtraction and one mask instead of a per-letter loop.
    """

    width: int
    guard: int

    @classmethod
    def for_target(cls, target: LetterCounts) -> LetterPacking:
        """Packing wide enough for ``target`` and anything that fits in it."""
        width = max(target.counts, default=0).bit_length() + 1
        guard = 0
        for i in range(len(target.counts)):
            guard |= 1 << (i * width + width - 1)
        return cls(width, guard)

    def pack(self, letters: LetterCounts) -> int:
        limit = 1 << (self.width - 1)
        value = 0
        for i, count in enumerate(letters.counts):
            if count >= limit:
                raise ValueError(f"count {count} does not fit in a {self.width}-bit field")
            value |= count << (i * self.width)
        return value

    def fits(self, word: int, remaining: int) -> bool:
        """Packed equivalent of ``LetterCounts.fits_in``: every letter of ``word`` is
        available in ``remaining``."""
        return ((remaining | self.guard) - word) & self.guard == self.guard
