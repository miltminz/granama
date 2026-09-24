# granama

Fast multi-word anagram generator.

```console
$ uv run granama "dormitory" --max-words 2
dormitory
dirty moor
dirty room
roomy dirt
$ uv run granama "clint eastwood" --min-len 3 --limit 5
```

Options:
- search: `--min-words`, `--max-words`, `--min-len`, `--max-len` (letters per word), `--limit`
- vocabulary: `--lang` (default `en`), `--dictionary NAME` (see `--list-dictionaries`) or
  `--wordlist PATH` (custom list, one word per line), `--exclude-tags TAGS` (default
  `offensive,vulgar`; `--exclude-tags ""` keeps every word), `--max-rarity TIER` (bundled
  dictionaries only; keeps only words at or below that rarity, see below)
- output: `--rank` (most common words first; bundled dictionaries only, see below) or
  `--rank neural` (fluent phrases related to the input, in their best word order; needs the
  `lm` extra, see [Neural ranking](#neural-ranking))

Library:

```python
from granama import SearchOptions, generate_anagrams, get_language, load_vocabulary, rank_anagrams

vocab = load_vocabulary("en", dictionary="scowl-60", exclude_tags={"offensive", "vulgar"})
for words in generate_anagrams("dirty room", vocab, SearchOptions(max_words=2)):
    print(words)

best = rank_anagrams(generate_anagrams("clint eastwood", vocab), vocab, limit=10)

get_language("en").dictionaries  # bundled dictionaries, e.g. to offer a choice in a UI
```

`load_vocabulary` caches one vocabulary per (dictionary, filters) combination, so a server
can call it on every request. `exclude_words=` removes extra words (e.g. a per-site blocklist).

Results are unordered word sets: each combination is produced once, regardless of word order.

## How it works

Words are grouped by letter signature, so mutual anagrams like listen/silent/enlist are
one node. The search backtracks over signatures in a fixed order, which avoids producing
the same set in different orders. At each level it keeps only candidates that still fit
the remaining letters, and it prunes branches using the word-count and word-length bounds.
Signatures are expanded into words only at the end.

Two measured optimizations (see `benchmarks/`):
- **Packed letter counts:** each signature is a single int with a small bit field per
  letter plus a guard bit, so "does this word fit" is one subtraction and one mask
  (`LetterPacking`). This made the solver 6–12× faster.
- **Word-count bound:** candidates are tried longest first, so once a word of length L is
  chosen, no later word is longer than L. If the remaining letters need more than
  `words_left × L`, the rest of that level is skipped. With `--max-words` this was another
  8–14× faster.

## Development

```console
$ uv sync
$ uv run pytest --cov=granama
$ uv run ruff check . && uv run ruff format --check .
```

Measure before optimizing. `benchmarks/bench.py` times solvers on a fixed set of cases
(fresh process per run, best of 3, peak memory), and `--check` verifies that a candidate
gives exactly the same output as the reference:

```console
$ uv run python benchmarks/bench.py                          # current solver
$ uv run python benchmarks/bench.py --check -s ref=granama:generate_anagrams -s new=path/new.py:generate_anagrams
$ uv run python benchmarks/bench.py -s ref=granama:generate_anagrams -s new=path/new.py:generate_anagrams
```

The same goes for ranking quality. `benchmarks/eval_ranking.py` ranks every anagram of 13
inputs with a known good answer ("dormitory" → "dirty room", ...) and shows where that
answer lands, summarized by the geometric mean of its positions (lower is better).
Unranked output scores 103 and the current `rank_anagrams` scores 59:

```console
$ uv run python benchmarks/eval_ranking.py                   # unranked vs current ranker
$ uv run python benchmarks/eval_ranking.py -r new=path/new.py:rank_anagrams
```

`benchmarks/eval_lm.py` runs the same cases through the models of `--rank neural` (or any
other Hugging Face model) and reports where the known answers land for each model and for
combinations of fluency and relevance. Scores are cached in `benchmarks/.eval_lm_cache/`, so
adding a case only scores that case:

```console
$ uv run --extra lm python benchmarks/eval_lm.py embed sentence-transformers/all-mpnet-base-v2
$ uv run --extra lm python benchmarks/eval_lm.py lm HuggingFaceTB/SmolLM2-135M --pool 2000
$ uv run --extra lm python benchmarks/eval_lm.py report --show 5
```

The tests of `granama.neural` use fake models; `GRANAMA_TEST_MODELS=1 uv run --extra lm
pytest` also runs the real default models (downloads them).

## Neural ranking

`--rank neural` looks for anagrams that mean something: phrases that read as English and are
related to the input. It needs the optional `lm` extra (torch, transformers,
sentence-transformers; `pip install 'granama[lm]'`, or `uv run --extra lm granama ...` here),
downloads two small models on first use, and uses a CUDA GPU when there is one.

```console
$ uv run --extra lm granama desperation --rank neural --limit 3 --scores
 +5.86   -25.8  0.391  note despair
 +5.76   -28.0  0.419  tone despair
 +4.76   -31.4  0.403  ore and spite
```

Each anagram gets two scores:
- **fluency**: how likely a language model (default `HuggingFaceTB/SmolLM2-135M`) finds the
  phrase, as log P(words + "."), for its best word order;
- **relevance**: how close in meaning it is to the input, as the cosine similarity of
  sentence embeddings (default `Snowflake/snowflake-arctic-embed-m-v1.5`).

Both are standardized over the anagrams of the input (z-scores, so they can be added) and
combined as `z(fluency) + 0.5 × z(relevance)`. Scoring every word order of every anagram is
too slow, so a first pass, z(relevance) + z(commonness of the rarest word), keeps the best
`--pool` anagrams (default 1000) for the language model; only those are printed. With
`--scores`, each line starts with the combined score, the fluency and the relevance.

The models and weights are options (`--fluency-model`, `--relevance-model`,
`--relevance-prompt`, `--relevance-weight`, `--pool`, `--device`). `--rarity-weight` (default
1) sets how much rare words count against an anagram in the first pass: 0 ignores rarity,
higher values keep more anagrams of common words in the pool, and negative values favor
rare words (for more unusual finds; the language model still finds rare words less fluent).

Each anagram is scored in at most `--max-orders` word orders (default 20; all of them up to 3
words). For longer anagrams, the language model first scores every word as a phrase start and
every ordered pair of words, once per input, and the orders with the best sum of start and
pair scores are the ones scored in full. On the known anagrams this picks the same orders as
trying every order, and it keeps anagrams of 6+ words tractable (n words have up to n! orders).
`--capitalize` scores "Old west action" instead of "old west action.": on 96 phrases with a
known word order it picked the right order more often (75% against 64%), but it made no
difference to the ranking of the known anagrams, so it is off by default. In Python the models are
pluggable: `NeuralRanker` takes any object with a `log_prob(phrases, context="")` method
(`Fluency`) and one with a `similarity(source, phrases)` method (`Relevance`):

```python
from granama import generate_anagrams, load_vocabulary
from granama.neural import CausalLMFluency, EmbeddingRelevance, NeuralRanker

vocab = load_vocabulary("en")
ranker = NeuralRanker(CausalLMFluency(), EmbeddingRelevance(), relevance_weight=0.5)
for anagram in ranker.rank("desperation", generate_anagrams("desperation", vocab), vocab, 5):
    print(anagram.phrase, anagram.score)
```

The defaults were chosen with `benchmarks/eval_lm.py` (see Development): on its known
anagrams, the answer's geometric-mean rank goes from 43 with `--rank` to about 4. Larger
language models (SmolLM2-360M, Qwen2.5-0.5B, Qwen3-0.6B) were no better. Among 14 embedding
models, snowflake-arctic-embed-m-v1.5 had the best first pass (every known answer in its top
400, against 1000 for all-mpnet-base-v2) at the same size; the newest retrieval models
(granite-embedding-r2, gte-modernbert, EmbeddingGemma) did worse on this task. Known limits: the
language model sometimes prefers a worse word order ("action old west"), and large outputs
are slow, since the first pass embeds every anagram. Gated models (e.g. `google/embeddinggemma-300m`) need
`HF_TOKEN` exported and their license accepted.

## Dictionaries and word tags

Bundled dictionaries live in `src/granama/data/<lang>/` and are registered in
`src/granama/languages.py`:

| Name | Source | Words |
|---|---|---|
| `scowl-60` (default) | [SCOWL v2 / ESDB](https://wordlist.aspell.net/) size 60, American spelling | 78,804 |

Only plain lowercase words are kept (no proper nouns, abbreviations, possessives or
hyphenated entries), with accents removed.

Words can carry tags, which `--exclude-tags` / `exclude_tags=` filter on. A tag family
excludes all its levels (`vulgar` covers `vulgar-1` and `vulgar-3`). Tags come from:
- `<dictionary>.tags.tsv`: ESDB usage notes (`offensive-1/2`, `vulgar-1/2/3`, `informal`).
  ESDB only tags the worst offenders.
- `en/extra-tags.tsv`: hand-curated. It adds the slurs and vulgar words ESDB misses and
  fixes false positives with `none`. For a word listed there it replaces the dictionary's
  tags, and it also applies to `--wordlist` files. The file header lists which words were
  deliberately left untagged.

### Rarity: ranking and filtering

Most generated anagrams are meaningless word combinations. As a cheap first pass toward
meaningful ones, each word of a bundled dictionary has a rarity tier in
`<dictionary>.rarity.tsv`, generated alongside the word list: the smallest SCOWL size the
word appears at. SCOWL sizes are cumulative lists from most to least common, so a lower
tier means a more common word. In `scowl-60` the tiers are 35 (about half the words), 40,
50 and 60.

- `--rank` / `rank_anagrams()` prints the anagrams with the most common words first: first
  by the rarest word's tier, then by the sum of tiers. It must see every result before
  printing, so `--limit` then keeps the best N rather than the first N.
- `--max-rarity TIER` / `max_rarity=` removes words rarer than `TIER` from the vocabulary.
  This is faster and gives less output, but a single rare word drops a good anagram
  ("moon starer" is lost at any tier below 60 because of "starer").

Both judge words, not phrases, so plenty of nonsense still ranks high. Both need a bundled
dictionary with rarity data, not `--wordlist`. `benchmarks/eval_ranking.py` measures how
close to the top known good anagrams land (see Development).

To (re)build a dictionary from a pinned ESDB commit (needs git, make and python3; takes
about a minute):

```console
$ uv run python scripts/build_wordlist.py --size 60      # sizes 35 40 50 55 60 70 80
```

To add another dictionary, build it (or drop a word list in `data/<lang>/`) and add a
`Dictionary(...)` entry to the language in `languages.py`.

SCOWL copyright: `src/granama/data/en/SCOWL-COPYRIGHT.txt`.
