"""Rank anagram candidates for a list of seed phrases (puzzle pipeline, steps 1-2).

Reads seeds (one per line, ``#`` comments), ranks the anagrams of each with the neural ranker
and appends one JSON line per seed to the output. Seeds already in the output are skipped,
so an interrupted run resumes where it stopped.

    uv run --extra lm python -u scripts/build_candidates.py seeds/people.txt \
        -o out/people.jsonl --top 30
"""

import argparse
import json
import sys
import time
from pathlib import Path

from granama import SearchOptions, generate_anagrams, load_vocabulary
from granama.neural import NeuralRanker
from granama.neural.models import CausalLMFluency, EmbeddingRelevance
from granama.normalize import normalize


def read_seeds(path: Path) -> list[str]:
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            seeds.append(line)
    return list(dict.fromkeys(seeds))


def done_seeds(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["seed"] for line in f if line.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("seeds", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=30, help="candidates kept per seed")
    parser.add_argument("--max-words", type=int, default=4)
    parser.add_argument("--min-len", type=int, default=2, help="letters per word")
    parser.add_argument("--min-letters", type=int, default=8, help="skip shorter seeds")
    parser.add_argument("--max-letters", type=int, default=18, help="skip longer seeds")
    parser.add_argument(
        "--max-anagrams",
        type=int,
        default=300_000,
        help="skip seeds with more anagrams than this (ranking cost)",
    )
    parser.add_argument("--pool", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="process at most N new seeds")
    args = parser.parse_args(argv)

    vocab = load_vocabulary("en", exclude_tags={"offensive", "vulgar"})
    alphabet = vocab.language.alphabet
    options = SearchOptions(max_words=args.max_words, min_word_len=args.min_len)

    seeds = read_seeds(args.seeds)
    done = done_seeds(args.output)
    todo = [s for s in seeds if s not in done][: args.limit]
    print(f"{len(seeds)} seeds, {len(done)} done, {len(todo)} to do", flush=True)

    ranker = NeuralRanker(CausalLMFluency(), EmbeddingRelevance(), pool=args.pool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.output.open("a", encoding="utf-8") as out:
        for n, seed in enumerate(todo, 1):
            t = time.time()
            tokens = {normalize(w, alphabet) for w in seed.split()} - {""}
            letters = sum(len(w) for w in tokens)
            record = {"seed": seed, "letters": letters, "candidates": []}
            if not args.min_letters <= letters <= args.max_letters:
                record["skipped"] = "length"
            else:
                # an anagram that reuses a word of the seed is no puzzle
                anagrams = [
                    words
                    for words in generate_anagrams(seed, vocab, options)
                    if not tokens.intersection(words)
                ]
                record["anagrams"] = len(anagrams)
                if len(anagrams) > args.max_anagrams:
                    record["skipped"] = "too many anagrams"
                elif anagrams:
                    ranked = ranker.rank(seed, anagrams, vocab, args.top)
                    record["candidates"] = [
                        {
                            "phrase": a.phrase,
                            "score": round(a.score, 3),
                            "fluency": round(a.fluency, 2),
                            "relevance": round(a.relevance, 4),
                        }
                        for a in ranked
                    ]
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            best = record["candidates"][0]["phrase"] if record["candidates"] else "-"
            note = record.get("skipped", f"{record.get('anagrams', 0)} anagrams")
            elapsed = time.time() - start
            eta = elapsed / n * (len(todo) - n)
            print(
                f"[{n}/{len(todo)}] {seed!r}: {note}, best {best!r} "
                f"({time.time() - t:.1f}s, eta {eta / 60:.0f} min)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
