"""Command-line interface."""

import argparse
import dataclasses
import sys

from granama.languages import get_language, supported_languages
from granama.neural import (
    DEFAULT_FLUENCY_MODEL,
    DEFAULT_RELEVANCE_MODEL,
    CausalLMFluency,
    EmbeddingRelevance,
    NeuralRanker,
)
from granama.options import SearchOptions
from granama.ranking import rank_anagrams
from granama.solver import generate_anagrams
from granama.vocabulary import DEFAULT_EXCLUDED_TAGS, load_vocabulary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="granama", description="Generate multi-word anagrams.")
    parser.add_argument("text", nargs="?", help="input text (spaces and punctuation are ignored)")
    parser.add_argument("--lang", default="en", choices=supported_languages())
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dictionary", help="bundled dictionary name (see --list-dictionaries)")
    source.add_argument("--wordlist", help="custom word list file, one word per line")
    parser.add_argument(
        "--list-dictionaries", action="store_true", help="list bundled dictionaries and exit"
    )
    parser.add_argument(
        "--exclude-tags",
        default=",".join(sorted(DEFAULT_EXCLUDED_TAGS)),
        metavar="TAGS",
        help="comma-separated word tags to exclude; a family like 'vulgar' covers "
        "'vulgar-1', 'vulgar-3', ... (default: %(default)s; '' keeps all words)",
    )
    parser.add_argument(
        "--max-rarity",
        type=int,
        metavar="TIER",
        help="keep only words at or below this SCOWL rarity tier (lower = more common, "
        "e.g. 35 for common words, 60 for everything in scowl-60); requires a bundled "
        "dictionary with rarity data",
    )
    parser.add_argument(
        "--rank",
        nargs="?",
        const="rarity",
        choices=["rarity", "neural"],
        help="print the most plausible anagrams first; --limit then keeps the best ones. "
        "'rarity' (the default with a bare --rank) puts common words first and requires a "
        "bundled dictionary with rarity data. 'neural' ranks by fluency and relevance with "
        "language models (needs the lm extra, see the neural ranking options)",
    )
    neural = parser.add_argument_group("neural ranking (--rank neural)")
    neural.add_argument(
        "--fluency-model",
        default=DEFAULT_FLUENCY_MODEL,
        metavar="MODEL",
        help="Hugging Face causal language model (default: %(default)s)",
    )
    neural.add_argument(
        "--relevance-model",
        default=DEFAULT_RELEVANCE_MODEL,
        metavar="MODEL",
        help="sentence-transformers embedding model (default: %(default)s)",
    )
    neural.add_argument(
        "--relevance-prompt", metavar="NAME", help="prompt name of the embedding model"
    )
    neural.add_argument(
        "--relevance-weight",
        type=float,
        default=NeuralRanker.relevance_weight,
        metavar="W",
        help="weight of relevance against fluency (default: %(default)s)",
    )
    neural.add_argument(
        "--rarity-weight",
        type=float,
        default=NeuralRanker.rarity_weight,
        metavar="W",
        help="how much rare words count against an anagram when choosing the --pool "
        "anagrams (default: %(default)s; 0 ignores rarity, negative favors rare words); "
        "needs a bundled dictionary",
    )
    neural.add_argument(
        "--pool",
        type=int,
        default=NeuralRanker.pool,
        metavar="N",
        help="anagrams scored by the language model (default: %(default)s); at most "
        "this many are printed",
    )
    neural.add_argument(
        "--max-orders",
        type=int,
        default=NeuralRanker.max_orders,
        metavar="N",
        help="word orders scored per anagram (default: %(default)s); longer anagrams get "
        "their best orders under word-pair scores",
    )
    neural.add_argument(
        "--capitalize",
        action="store_true",
        help='score phrases as "Old west action" rather than "old west action."',
    )
    neural.add_argument(
        "--scores",
        action="store_true",
        help="print score, fluency (log-probability) and relevance before each anagram",
    )
    neural.add_argument("--device", help="torch device, e.g. cpu (default: GPU if available)")
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--min-len", type=int, default=1, help="minimum letters per word")
    parser.add_argument("--max-len", type=int, help="maximum letters per word")
    parser.add_argument("--limit", type=int, help="stop after this many anagrams")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_dictionaries:
        language = get_language(args.lang)
        for dictionary in language.dictionaries:
            default = " (default)" if dictionary.name == language.default_dictionary else ""
            print(f"{dictionary.name}{default}: {dictionary.description}")
        return 0
    if args.text is None:
        parser.error("the following arguments are required: text")

    try:
        options = SearchOptions(
            min_words=args.min_words,
            max_words=args.max_words,
            min_word_len=args.min_len,
            max_word_len=args.max_len,
            limit=args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))  # exits with status 2

    exclude_tags = [tag.strip() for tag in args.exclude_tags.split(",") if tag.strip()]
    try:
        vocabulary = load_vocabulary(
            args.lang,
            dictionary=args.dictionary,
            path=args.wordlist,
            exclude_tags=exclude_tags,
            max_rarity=args.max_rarity,
        )
    except ValueError as exc:
        parser.error(str(exc))
    except OSError as exc:
        print(f"granama: cannot read word list: {exc}", file=sys.stderr)
        return 1

    if args.scores and args.rank != "neural":
        parser.error("--scores requires --rank neural")
    if args.rank == "neural":
        if args.pool < 1:
            parser.error("--pool must be >= 1")
        if args.max_orders < 1:
            parser.error("--max-orders must be >= 1")
        try:
            ranker = neural_ranker(args)
        except (ImportError, OSError) as exc:  # missing extra, unknown model
            parser.error(str(exc))
        search = dataclasses.replace(options, limit=None)
        anagrams = generate_anagrams(args.text, vocabulary, search)
        for anagram in ranker.rank(args.text, anagrams, vocabulary, options.limit):
            if args.scores:
                print(
                    f"{anagram.score:+6.2f} {anagram.fluency:7.1f} {anagram.relevance:6.3f}  ",
                    end="",
                )
            print(anagram.phrase)
        return 0
    if args.rank:
        if not vocabulary.rarity:
            parser.error("--rank requires a bundled dictionary with rarity data")
        # search everything, then keep the best --limit anagrams
        search = dataclasses.replace(options, limit=None)
        anagrams = generate_anagrams(args.text, vocabulary, search)
        results = rank_anagrams(anagrams, vocabulary, options.limit)
    else:
        results = generate_anagrams(args.text, vocabulary, options)
    for words in results:
        print(" ".join(words))
    return 0


def neural_ranker(args: argparse.Namespace) -> NeuralRanker:
    """Load the models named on the command line (slow: they may be downloaded)."""
    return NeuralRanker(
        fluency=CausalLMFluency(args.fluency_model, device=args.device, capitalize=args.capitalize),
        relevance=EmbeddingRelevance(
            args.relevance_model, device=args.device, prompt=args.relevance_prompt
        ),
        relevance_weight=args.relevance_weight,
        rarity_weight=args.rarity_weight,
        pool=args.pool,
        max_orders=args.max_orders,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
