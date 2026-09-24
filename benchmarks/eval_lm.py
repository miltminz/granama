"""Evaluate language-model scorers on the known anagrams of eval_ranking.py.

Usage (needs the ``lm`` extra; a CUDA GPU is used when available):
  uv run --extra lm python benchmarks/eval_lm.py embed sentence-transformers/all-mpnet-base-v2
  uv run --extra lm python benchmarks/eval_lm.py lm HuggingFaceTB/SmolLM2-135M --pool 2000
  uv run --extra lm python benchmarks/eval_lm.py report

The models are granama's own (``granama.neural.models``), so this measures what
``granama --rank neural`` uses. ``granama.neural.NeuralRanker`` implements the best pipeline
found here (with snowflake-arctic-embed-m-v1.5 relevance for both passes and a pool of 1000:
``lm HuggingFaceTB/SmolLM2-135M --pool 1000 --first-pass
emb:Snowflake/snowflake-arctic-embed-m-v1.5``).

Every anagram of each case in ``eval_ranking.CASES`` (except the input itself) is scored by:
- an embedding model (``embed``): relevance, the cosine similarity of anagram and input.
- a causal LM (``lm``): fluency, log P(words + ".") for the best of all word orders, and
  the same probability conditioned on "<input>: ".

Scoring every order of every anagram is slow, so ``lm --pool N`` scores only the N best
anagrams per case under a cheap first pass, z(mpnet relevance) + z(commonness of the rarest
word), which keeps every known answer in the top 1000 (``--first-pass`` picks another
embedding scorer). ``--pool-from LM_SCORER`` uses
z(that LM's fluency) + 0.5 z(mpnet relevance) as the first pass instead. Anagrams outside
the pool rank last. The first pass needs mpnet scores, so run ``embed`` with it first.

Scores are cached per (scorer, case) in ``benchmarks/.eval_lm_cache`` and reused while the
case's anagrams are unchanged, so adding a case only scores that case. ``report`` shows
where each known answer lands for every scorer alone and for combinations of z-scores,
fluency + w * relevance + w * commonness, summarized as in eval_ranking.py. It only
includes scorers that have scored every case.
"""

import argparse
import hashlib
import itertools
import math
import re
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from eval_ranking import CASES

from granama import generate_anagrams, load_vocabulary, rank_anagrams
from granama.neural import CausalLMFluency, EmbeddingRelevance

CACHE = Path(__file__).parent / ".eval_lm_cache"
FIRST_PASS = "emb:sentence-transformers/all-mpnet-base-v2"
CONDITION = "{text}: "
RELEVANCE_WEIGHTS = [0.5, 1, 2]
COMMONNESS_WEIGHTS = [0, 0.5]


@dataclass
class Case:
    text: str
    answer: str
    anagrams: list[tuple[str, ...]]
    commonness: np.ndarray  # minus the rarest word's tier: higher is more common
    fingerprint: str = field(init=False)
    answer_index: int = field(init=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha1()
        for words in self.anagrams:
            digest.update(" ".join(words).encode() + b"\n")
        self.fingerprint = digest.hexdigest()
        target = sorted(self.answer.split())
        self.answer_index = next(
            i for i, words in enumerate(self.anagrams) if sorted(words) == target
        )

    def rank(self, scores: np.ndarray) -> int:
        """Position of the known answer (1 = first); last if it was not scored."""
        answer = scores[self.answer_index]
        if not np.isfinite(answer):
            return len(self.anagrams)
        return int((scores > answer).sum()) + 1


def load_cases() -> list[Case]:
    vocabulary = load_vocabulary("en")
    cases = []
    for text, answer in CASES.items():
        own = sorted(text.split())
        anagrams = [w for w in generate_anagrams(text, vocabulary) if sorted(w) != own]
        tiers = [max(vocabulary.rarity.get(w, math.inf) for w in words) for words in anagrams]
        cases.append(Case(text, answer, anagrams, -np.array(tiers, float)))
    return cases


# ------------------------------------------------------------------------------ cache


def _path(scorer: str, case: Case) -> Path:
    return CACHE / re.sub(r"[/:]", "__", scorer) / f"{case.text.replace(' ', '_')}.npz"


NAME_FILE = "scorer.txt"  # each scorer directory records its scorer name


def load_scores(scorer: str, case: Case) -> dict[str, np.ndarray] | None:
    path = _path(scorer, case)
    if not path.exists():
        return None
    with np.load(path) as data:
        if str(data["fingerprint"]) != case.fingerprint:
            return None
        return {key: data[key] for key in data.files if key != "fingerprint"}


def save_scores(scorer: str, case: Case, **scores: np.ndarray) -> None:
    path = _path(scorer, case)
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / NAME_FILE).write_text(scorer)
    np.savez(path, fingerprint=case.fingerprint, **scores)


def cached_scorers() -> list[str]:
    """Scorer names with a cache directory (``emb:...`` or ``lm:...``)."""
    return sorted(path.read_text() for path in CACHE.glob(f"*/{NAME_FILE}"))


def fluency(scorer: str, case: Case, key: str = "fluency") -> np.ndarray:
    """An LM score over all anagrams, -inf outside the scored pool."""
    data = load_scores(scorer, case)
    if data is None:
        raise SystemExit(f"{scorer} has not scored {case.text!r}")
    out = np.full(len(case.anagrams), -np.inf)
    out[data["pool"]] = data[key]
    return out


def relevance(scorer: str, case: Case) -> np.ndarray:
    data = load_scores(scorer, case)
    if data is None:
        raise SystemExit(f"{scorer} has not scored {case.text!r} (run embed first)")
    return data["relevance"].astype(float)


def zscore(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Standardize ``x`` over ``mask`` (default: its finite values); -inf elsewhere."""
    if mask is None:
        mask = np.isfinite(x)
    out = np.full(len(x), -np.inf)
    values = x[mask]
    out[mask] = (values - values.mean()) / (values.std() + 1e-9)
    return out


# ------------------------------------------------------------------------------ scoring


def embed(model_name: str, prompt: str | None, full_precision: bool, batch_size: int) -> None:
    scorer = f"emb:{model_name}" + (f":{prompt}" if prompt else "")
    todo = [case for case in load_cases() if load_scores(scorer, case) is None]
    if not todo:
        print(f"{scorer}: every case already scored")
        return
    model = EmbeddingRelevance(
        model_name, prompt=prompt, half=not full_precision, batch_size=batch_size
    )
    for case in todo:
        similarity = model.similarity(case.text, [" ".join(w) for w in case.anagrams])
        save_scores(scorer, case, relevance=np.array(similarity, np.float32))
        print(f"  {case.text}: {len(case.anagrams)} anagrams", flush=True)


def first_pass(case: Case, pool_from: str | None, embedder: str) -> np.ndarray:
    rel = zscore(relevance(embedder, case))
    if pool_from:
        return zscore(fluency(pool_from, case)) + 0.5 * rel
    return rel + zscore(case.commonness)


def lm(
    model_name: str, pool: int | None, pool_from: str | None, embedder: str, batch_tokens: int
) -> None:
    scorer = f"lm:{model_name}"
    if pool:
        scorer += f":pool{pool}" + (f":from:{pool_from.split('/')[-1]}" if pool_from else "")
        if embedder != FIRST_PASS:
            scorer += f":first:{embedder.split('/')[-1]}"
    elif pool_from:
        raise SystemExit("--pool-from needs --pool")
    todo = [case for case in load_cases() if load_scores(scorer, case) is None]
    if not todo:
        print(f"{scorer}: every case already scored")
        return
    model = CausalLMFluency(model_name, batch_tokens=batch_tokens)
    for case in todo:
        indices = np.arange(len(case.anagrams))
        if pool:
            indices = np.sort(
                np.argsort(-first_pass(case, pool_from, embedder), kind="stable")[:pool]
            )
        texts, owner = [], []
        for i in indices:
            orders = sorted({" ".join(p) for p in itertools.permutations(case.anagrams[i])})
            texts += orders
            owner += [i] * len(orders)
        scores = model.log_prob(texts)
        best: dict[int, tuple[float, str]] = {}
        for text, i, score in zip(texts, owner, scores, strict=True):
            if i not in best or score > best[i][0]:
                best[i] = (score, text)
        orders = [best[i][1] for i in indices]
        save_scores(
            scorer,
            case,
            pool=indices,
            order=np.array(orders),
            fluency=np.array([best[i][0] for i in indices], np.float32),
            conditional=np.array(
                model.log_prob(orders, context=CONDITION.format(text=case.text)), np.float32
            ),
        )
        print(f"  {case.text}: {len(indices)} anagrams, {len(texts)} orders", flush=True)


# ------------------------------------------------------------------------------ report


@dataclass
class Row:
    name: str
    ranks: list[int]
    score: Callable[[Case], np.ndarray] | None = None  # to show the top anagrams
    lm: str | None = None

    @property
    def geomean(self) -> float:
        return math.exp(statistics.fmean(map(math.log, self.ranks)))


def short(scorer: str) -> str:
    return re.sub(r"[\w.-]+/", "", scorer.partition(":")[2])


def report(top: int, show: int) -> None:
    cases = load_cases()
    complete, skipped = [], []
    for scorer in cached_scorers():
        ok = all(load_scores(scorer, case) is not None for case in cases)
        (complete if ok else skipped).append(scorer)
    embedders = [s for s in complete if s.startswith("emb:")]
    lms = [s for s in complete if s.startswith("lm:")]

    vocabulary = load_vocabulary("en")
    baseline = Row(
        "current rank_anagrams",
        [
            next(
                i
                for i, words in enumerate(rank_anagrams(case.anagrams, vocabulary), 1)
                if sorted(words) == sorted(case.answer.split())
            )
            for case in cases
        ],
    )
    rows = [baseline]
    for emb in embedders:
        rows.append(Row(short(emb), [case.rank(relevance(emb, case)) for case in cases]))

    def combination(lm_: str, key: str, emb: str | None, w: float, wc: float):
        def score(case: Case) -> np.ndarray:
            x = fluency(lm_, case, key)
            mask = np.isfinite(x)
            total = zscore(x, mask)
            if wc:
                total = total + wc * zscore(case.commonness, mask)
            if emb:
                total = total + w * zscore(relevance(emb, case), mask)
            return total

        return score

    for lm_ in lms:
        for key in ["fluency", "conditional"]:
            for emb in [None, *embedders]:
                for w in [0] if emb is None else RELEVANCE_WEIGHTS:
                    for wc in COMMONNESS_WEIGHTS:
                        score = combination(lm_, key, emb, w, wc)
                        name = f"{short(lm_)} {key}"
                        name += f" + {w}*{short(emb)}" if emb else ""
                        name += f" + {wc}*commonness" if wc else ""
                        ranks = [case.rank(score(case)) for case in cases]
                        rows.append(Row(name, ranks, score, lm_))

    rows.sort(key=lambda row: row.geomean)
    shown = rows[:top] if any(row is baseline for row in rows[:top]) else [*rows[:top], baseline]
    width = max(len(row.name) for row in shown)
    print(f"{'scorer':<{width}} {'geomean':>7} {'median':>6} {'top10':>5} {'top100':>6}  ranks")
    for row in shown:
        r = row.ranks
        print(
            f"{row.name:<{width}} {row.geomean:>7.1f} {statistics.median(r):>6g}"
            f" {sum(x <= 10 for x in r):>5} {sum(x <= 100 for x in r):>6}  {' '.join(map(str, r))}"
        )
    print("cases:", " | ".join(f"{c.answer} ({len(c.anagrams)})" for c in cases))
    if skipped:
        print("not in the table (missing cases):", ", ".join(skipped))

    best = next((row for row in rows if row.score), None)
    if show and best:
        print(f"\ntop {show} of {best.name}:")
        for case in cases:
            scores = best.score(case)
            data = load_scores(best.lm, case)
            order = dict(zip(data["pool"].tolist(), data["order"].tolist(), strict=True))
            top_ids = np.argsort(-scores, kind="stable")[:show]
            print(f"  {case.text}: " + " | ".join(order[i] for i in top_ids.tolist()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("embed", help="score relevance with a sentence-transformers model")
    p.add_argument("model")
    p.add_argument("--prompt", help="prompt name of the model, e.g. STS for EmbeddingGemma")
    p.add_argument(
        "--full-precision", action="store_true", help="skip fp16 on GPU (needed for Gemma)"
    )
    p.add_argument("--batch-size", type=int, default=1024)
    p = commands.add_parser("lm", help="score fluency with a causal language model")
    p.add_argument("model")
    p.add_argument("--pool", type=int, help="score only the best N anagrams per case")
    p.add_argument(
        "--pool-from", metavar="LM_SCORER", help="e.g. lm:HuggingFaceTB/SmolLM2-135M:pool2000"
    )
    p.add_argument(
        "--first-pass",
        default=FIRST_PASS,
        metavar="EMB_SCORER",
        help="embedding scorer of the first pass (default: %(default)s)",
    )
    p.add_argument("--batch-tokens", type=int, default=20_000, help="lower it on out-of-memory")
    p = commands.add_parser("report", help="rank of the known answers per scorer")
    p.add_argument("--top", type=int, default=20, help="rows to print")
    p.add_argument("--show", type=int, default=0, metavar="N", help="print the best row's top N")
    args = parser.parse_args()
    if args.command == "embed":
        embed(args.model, args.prompt, args.full_precision, args.batch_size)
    elif args.command == "lm":
        lm(args.model, args.pool, args.pool_from, args.first_pass, args.batch_tokens)
    else:
        report(args.top, args.show)


if __name__ == "__main__":
    main()
