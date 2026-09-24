"""Text normalization: reduce arbitrary text to the letters of an alphabet."""

import unicodedata


def normalize(text: str, alphabet: str) -> str:
    """Return ``text`` lowercased, stripped of diacritics and of any character
    not in ``alphabet`` (spaces, punctuation, digits, ...)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    allowed = set(alphabet)
    return "".join(ch for ch in decomposed if ch in allowed)
