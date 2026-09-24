"""Build a bundled English dictionary from SCOWL v2 / ESDB (https://wordlist.aspell.net/).

Usage: uv run python scripts/build_wordlist.py [--size 60] [--spelling A] [--variant-level 1]
                                               [--esdb-dir DIR]

Clones https://github.com/en-wl/wordlist at a pinned commit, builds its SQLite database
(needs git, make, python3 with sqlite3; about a minute) and writes into
src/granama/data/en/:

  scowl-<size>.txt        one word per line: plain lowercase words, accents stripped;
                          proper nouns, abbreviations, possessives and hyphenated or
                          multi-word entries are dropped
  scowl-<size>.tags.tsv   ESDB usage notes (offensive-1, vulgar-3, informal, ...) as
                          "word<TAB>tag,tag"
  SCOWL-COPYRIGHT.txt     upstream copyright and licence

A new dictionary must also be registered in src/granama/languages.py.
"""

import argparse
import re
import shutil
import sqlite3
import subprocess
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path

ESDB_REPO = "https://github.com/en-wl/wordlist.git"
ESDB_REF = "1e5b7d3a72f47a71da5d28686c1dd4b397178485"  # master, 2026-06-24
SIZES = (35, 40, 50, 55, 60, 70, 80)
# Single letters that are real words; other one-letter entries are dropped.
SINGLE_LETTER_WORDS = frozenset({"a", "i"})
EXTRA_WORDS = frozenset({"i"})  # ESDB lists it capitalized as "I"

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "granama" / "data" / "en"
_WORD_RE = re.compile(r"[a-z]+")

QUERY = """
select word, usage_note from scowl_v0
where size <= :size and variant_level <= :variant_level
  and spelling in (:spelling, '_') and region in ('', :region)
  and base_pos != 'abbr' and category = ''
"""
# No upper bound on size: this ranks every word by the smallest SCOWL size it appears at,
# regardless of which size ends up bundled as the dictionary.
SIZE_QUERY = """
select word, min(size) as tier from scowl_v0
where variant_level <= :variant_level
  and spelling in (:spelling, '_') and region in ('', :region)
  and base_pos != 'abbr' and category = ''
group by word
"""
REGIONS = {"A": "US", "B": "GB", "Z": "GB", "C": "CA", "D": "AU"}
# EXTRA_WORDS aren't necessarily rows in scowl_v0 under every spelling/region; treat them
# as maximally common since they're added by hand as core words.
FALLBACK_TIER = 10


def clean_word(raw: str) -> str | None:
    """Strip accents; return the word if it is plain lowercase letters, else None."""
    decomposed = unicodedata.normalize("NFKD", raw.strip())
    word = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    if not _WORD_RE.fullmatch(word):
        return None
    if len(word) == 1 and word not in SINGLE_LETTER_WORDS:
        return None
    return word


def filter_words(raw_words: Iterable[str]) -> list[str]:
    """Keep plain lowercase words, add ``EXTRA_WORDS``; return sorted and deduplicated."""
    words = set(EXTRA_WORDS)
    words.update(w for w in map(clean_word, raw_words) if w is not None)
    return sorted(words)


def collect_tags(rows: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    """Map each kept word to its sorted non-empty usage notes (a word may have several)."""
    tags: dict[str, set[str]] = {}
    for raw, note in rows:
        word = clean_word(raw)
        if word is not None and note:
            tags.setdefault(word, set()).add(note)
    return {word: sorted(notes) for word, notes in sorted(tags.items())}


def collect_tiers(rows: Iterable[tuple[str, int]]) -> dict[str, int]:
    """Map each kept word to the smallest SCOWL size it appears at (lower = more common)."""
    tiers: dict[str, int] = {}
    for raw, size in rows:
        word = clean_word(raw)
        if word is None:
            continue
        if word not in tiers or size < tiers[word]:
            tiers[word] = size
    return dict(sorted(tiers.items()))


def build_esdb(workdir: Path) -> Path:
    """Clone ESDB at ``ESDB_REF`` into ``workdir`` and build scowl.db; return the checkout."""
    repo = workdir / "esdb"
    repo.mkdir()
    for cmd in (
        ["git", "init", "-q"],
        ["git", "fetch", "-q", "--depth", "1", ESDB_REPO, ESDB_REF],
        ["git", "checkout", "-q", "FETCH_HEAD"],
        ["make"],
    ):
        subprocess.run(cmd, cwd=repo, check=True)
    return repo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=60, choices=SIZES)
    parser.add_argument("--spelling", default="A", choices=sorted(REGIONS))
    parser.add_argument("--variant-level", type=int, default=1)
    parser.add_argument("--esdb-dir", type=Path, help="existing ESDB checkout with scowl.db")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        repo = args.esdb_dir or build_esdb(Path(tmp))
        db = sqlite3.connect(repo / "scowl.db")
        params = {
            "size": args.size,
            "variant_level": args.variant_level,
            "spelling": args.spelling,
            "region": REGIONS[args.spelling],
        }
        rows = db.execute(QUERY, params).fetchall()
        size_rows = db.execute(SIZE_QUERY, params).fetchall()
        db.close()
        shutil.copyfile(repo / "Copyright", DATA_DIR / "SCOWL-COPYRIGHT.txt")

    words = filter_words(word for word, _ in rows)
    tags = collect_tags(rows)
    tiers = collect_tiers(size_rows)
    name = f"scowl-{args.size}"
    (DATA_DIR / f"{name}.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    header = f"# ESDB usage notes for {name} (generated by scripts/build_wordlist.py)\n"
    tag_lines = "".join(f"{word}\t{','.join(notes)}\n" for word, notes in tags.items())
    (DATA_DIR / f"{name}.tags.tsv").write_text(header + tag_lines, encoding="utf-8")
    rarity_header = (
        f"# SCOWL rarity tier for {name}: smallest SCOWL size each word appears at "
        "(lower = more common; generated by scripts/build_wordlist.py)\n"
    )
    word_tiers = {word: tiers.get(word, FALLBACK_TIER) for word in words}
    rarity_lines = "".join(f"{word}\t{tier}\n" for word, tier in word_tiers.items())
    (DATA_DIR / f"{name}.rarity.tsv").write_text(rarity_header + rarity_lines, encoding="utf-8")
    print(f"wrote {len(words)} words, {len(tags)} tagged, {len(word_tiers)} tiered")
    print(f"to {DATA_DIR}/{name}.*")


if __name__ == "__main__":
    main()
