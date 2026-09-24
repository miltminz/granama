"""Benchmark anagram solvers.

Usage:
  uv run python benchmarks/bench.py              # time the current solver
  uv run python benchmarks/bench.py -s old=path/old.py:generate_anagrams \
                                    -s new=granama:generate_anagrams
  uv run python benchmarks/bench.py --check -s ...   # verify identical output vs the first solver

A solver is ``module:function`` or ``path/to/file.py:function`` with the signature of
``granama.generate_anagrams``. Each (solver, case) runs in a fresh process; the table shows
the best wall time over ``--repeats`` runs (vocabulary loading excluded) and peak RSS.
"""

import argparse
import importlib
import importlib.util
import json
import resource
import subprocess
import sys
import time

from granama import SearchOptions, load_vocabulary

CASES: dict[str, tuple[str, dict]] = {
    "dormitory": ("dormitory", {}),
    "astronomer": ("astronomer", {}),
    "clint eastwood": ("clint eastwood", {}),
    "clint eastwood len>=3": ("clint eastwood", {"min_word_len": 3}),
    "clint eastwood first100": ("clint eastwood", {"limit": 100}),
    "william shakespeare w<=3": ("william shakespeare", {"max_words": 3}),
    "william shakespeare len>=4": ("william shakespeare", {"min_word_len": 4}),
    "presbyterians len>=3": ("presbyterians", {"min_word_len": 3}),
    "a decimal point w<=3": ("a decimal point", {"max_words": 3}),
    "the country side len>=4 first100": ("the country side", {"min_word_len": 4, "limit": 100}),
}
DEFAULT_SOLVER = "current=granama:generate_anagrams"


def load_solver(spec: str):
    target, func = spec.rsplit(":", 1)
    if target.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location("_bench_solver", target)
        assert module_spec and module_spec.loader
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, func)


def run_case(solver_spec: str, case: str) -> list[tuple[str, ...]]:
    text, kwargs = CASES[case]
    solver = load_solver(solver_spec)
    return list(solver(text, load_vocabulary("en"), SearchOptions(**kwargs)))


def child(solver_spec: str, case: str) -> None:
    load_vocabulary("en")
    start = time.perf_counter()
    n = len(run_case(solver_spec, case))
    elapsed = time.perf_counter() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"s": elapsed, "n": n, "mb": peak_mb}))


def measure(solver_spec: str, case: str, repeats: int) -> dict:
    runs = []
    for _ in range(repeats):
        proc = subprocess.run(
            [sys.executable, __file__, "--child", solver_spec, case],
            capture_output=True,
            text=True,
            check=True,
        )
        runs.append(json.loads(proc.stdout))
    best = min(runs, key=lambda r: r["s"])
    return {**best, "mb": max(r["mb"] for r in runs)}


def time_all(solvers: dict[str, str], cases: list[str], repeats: int) -> None:
    print(f"{'case':36} {'results':>8} " + " ".join(f"{name:>18}" for name in solvers))
    for case in cases:
        results = [measure(spec, case, repeats) for spec in solvers.values()]
        cells = " ".join(f"{r['s']:8.3f}s {r['mb']:5.0f}MB" for r in results)
        print(f"{case:36} {results[0]['n']:>8} {cells}", flush=True)


def check(solvers: dict[str, str], cases: list[str]) -> bool:
    (ref_name, ref_spec), *others = solvers.items()
    ok = True
    for case in cases:
        expected = run_case(ref_spec, case)
        for name, spec in others:
            same = run_case(spec, case) == expected
            ok &= same
            print(f"{'OK  ' if same else 'DIFF'} {name} vs {ref_name}: {case}")
    return ok


def main() -> int:
    if sys.argv[1:2] == ["--child"]:
        child(sys.argv[2], sys.argv[3])
        return 0
    parser = argparse.ArgumentParser(description="Benchmark anagram solvers.")
    parser.add_argument(
        "-s",
        "--solver",
        action="append",
        metavar="NAME=SPEC",
        help=f"solver to benchmark (repeatable, default {DEFAULT_SOLVER})",
    )
    parser.add_argument(
        "-c",
        "--case",
        action="append",
        choices=list(CASES),
        help="run only these cases (repeatable)",
    )
    parser.add_argument("-r", "--repeats", type=int, default=3)
    parser.add_argument(
        "--check", action="store_true", help="verify all solvers give identical output to the first"
    )
    args = parser.parse_args()
    solvers = dict(spec.split("=", 1) for spec in (args.solver or [DEFAULT_SOLVER]))
    cases = args.case or list(CASES)
    if args.check:
        return 0 if check(solvers, cases) else 1
    time_all(solvers, cases, args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
