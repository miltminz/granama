"""Evaluate anagram rankers on well-known anagrams.

Usage:
  uv run python benchmarks/eval_ranking.py                   # unranked vs the current ranker
  uv run python benchmarks/eval_ranking.py -r new=path/new.py:rank_anagrams

A ranker is ``module:function`` or ``path/to/file.py:function`` with the signature of
``granama.rank_anagrams`` (called without ``limit``). For each case, every anagram of the
input is generated (no constraints) and ranked; the table shows where the known answer
lands (1 = first) plus summary rows. The geometric mean is the headline number: unlike
the median, it moves when a ranker helps or hurts the long inputs with huge outputs.
"""

import argparse
import math
import statistics

from bench import load_solver

from granama import generate_anagrams, load_vocabulary

# input -> known meaningful anagram
CASES: dict[str, str] = {
    "dormitory": "dirty room",
    "clint eastwood": "old west action",
    "astronomer": "moon starer",
    "the eyes": "they see",
    "conversation": "voices rant on",
    "a gentleman": "elegant man",
    "eleven plus two": "twelve plus one",
    "slot machines": "cash lost in me",
    "the morse code": "here come dots",
    "desperation": "a rope ends it",
    "school master": "the classroom",
    "listen": "silent",
    "funeral": "real fun",
    "new york times": "monkeys write",
    "macdonalds": "clam and sod",
    "madam curie": "radium came",
    "albert einstein": "ten elite brains",
    # "asteroid threats": "disaster to earth",
    # "statue of liberty": "built to stay free",
}
DEFAULT_RANKERS = ["unranked", "current=granama:rank_anagrams"]


def rank_of(answer: str, ranked: list[tuple[str, ...]]) -> int | None:
    target = sorted(answer.split())
    return next((i for i, words in enumerate(ranked, 1) if sorted(words) == target), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-r",
        "--ranker",
        action="append",
        metavar="NAME=SPEC",
        help="ranker to evaluate ('unranked' keeps solver order); repeatable",
    )
    args = parser.parse_args()
    specs = args.ranker or DEFAULT_RANKERS

    vocabulary = load_vocabulary("en")
    results = {text: list(generate_anagrams(text, vocabulary)) for text in CASES}
    columns: dict[str, list[int | None]] = {}
    for spec in specs:
        name, _, target = spec.partition("=")
        rank = None if name == "unranked" else load_solver(target)
        columns[name] = [
            rank_of(answer, results[text] if rank is None else rank(results[text], vocabulary))
            for text, answer in CASES.items()
        ]

    width = max(len(answer) for answer in CASES.values())
    print(f"{'answer':<{width}} {'results':>8} " + " ".join(f"{n:>10}" for n in columns))
    for i, (text, answer) in enumerate(CASES.items()):
        cells = " ".join(f"{_fmt(ranks[i]):>10}" for ranks in columns.values())
        print(f"{answer:<{width}} {len(results[text]):>8} {cells}")
    sizes = [len(results[text]) for text in CASES]
    for label, summary in [
        ("geomean", lambda r: f"{math.exp(statistics.fmean(map(math.log, r))):.0f}"),
        ("median", lambda r: f"{statistics.median(r):g}"),
        ("top 10", lambda r: f"{sum(x <= 10 for x in r)}/{len(r)}"),
        ("top 100", lambda r: f"{sum(x <= 100 for x in r)}/{len(r)}"),
    ]:
        cells = " ".join(f"{summary(_filled(ranks, sizes)):>10}" for ranks in columns.values())
        print(f"{label:<{width}} {'':>8} {cells}")


def _filled(ranks: list[int | None], sizes: list[int]) -> list[int]:
    """Count a missing answer as ranked last among its case's results."""
    return [size if rank is None else rank for rank, size in zip(ranks, sizes, strict=True)]


def _fmt(rank: int | None) -> str:
    return "missing" if rank is None else str(rank)


if __name__ == "__main__":
    main()
