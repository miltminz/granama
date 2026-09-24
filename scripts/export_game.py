"""Embed the selected anagrams into game/index.html as an obfuscated blob.

The blob is XOR-scrambled then base64-encoded so the answers can't be spotted in the page
source; it is not encryption, since the page has to decode it to score guesses.

Usage: uv run python scripts/export_game.py [--selected out/selected.jsonl]
"""

import argparse
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = b"granama"  # must match KEY in game/index.html


def categories(seeds: Path) -> dict[str, str]:
    """Map each seed to the '# section' heading it appears under."""
    cats, current = {}, None
    for line in seeds.read_text().splitlines():
        line = line.strip()
        if line.startswith("# ") and not line.startswith("# Famous"):
            current = line[2:]
        elif line and not line.startswith("#"):
            cats[line] = current
    return cats


def encode(data: object) -> str:
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode()
    return base64.b64encode(bytes(b ^ KEY[i % len(KEY)] for i, b in enumerate(raw))).decode()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selected", type=Path, default=ROOT / "out/selected.jsonl")
    ap.add_argument("--seeds", type=Path, default=ROOT / "seeds/people.txt")
    ap.add_argument("--html", type=Path, default=ROOT / "game/index.html")
    args = ap.parse_args()

    cats = categories(args.seeds)
    puzzles = []
    for line in args.selected.read_text().splitlines():
        d = json.loads(line)
        puzzles.append(
            {
                "name": d["seed"],
                "cat": cats.get(d["seed"], "famous person"),
                "clues": [{"p": s["phrase"], "why": s["why"]} for s in d["selected"]],
            }
        )

    html = args.html.read_text()
    html, n = re.subn(
        r"^const PUZZLES = .*;$",
        lambda _: f'const PUZZLES = decode("{encode(puzzles)}");',
        html,
        count=1,
        flags=re.M,
    )
    if n != 1:
        raise SystemExit(f"no 'const PUZZLES = ...;' line in {args.html}")
    args.html.write_text(html)
    print(f"embedded {len(puzzles)} puzzles into {args.html}")


if __name__ == "__main__":
    main()
