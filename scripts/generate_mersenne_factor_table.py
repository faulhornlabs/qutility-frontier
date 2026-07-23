"""
Offline generator for the Mersenne-number factor table used by the Shor
period-finding benchmark's primitive-polynomial search
(frontier/shorbenchmark/_mersenne_factors.py).

For each degree n in [1, MAX_DEGREE], computes the complete set of distinct
prime factors of 2**n - 1 using sympy, and writes the result to
frontier/shorbenchmark/_mersenne_factors.json.

This script requires sympy, which is a dev-only tool dependency: the
generated JSON file is checked into the repo, and the runtime code
(_mersenne_factors.py) only depends on the standard library.

Usage:
    python scripts/generate_mersenne_factor_table.py [--max-degree N] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sympy import factorint

DEFAULT_MAX_DEGREE = 250
DEFAULT_OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "frontier"
    / "shorbenchmark"
    / "_mersenne_factors.json"
)


def generate(max_degree: int) -> dict:
    table = {}
    for n in range(1, max_degree + 1):
        order = (1 << n) - 1
        started = time.monotonic()
        factors = sorted(factorint(order).keys())
        elapsed = time.monotonic() - started
        table[str(n)] = factors
        print(f"n={n:>4}  primes={len(factors):>3}  {elapsed:6.2f}s")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, default=DEFAULT_MAX_DEGREE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    table = generate(args.max_degree)

    payload = {
        "_comment": (
            "Distinct prime factors of 2**n - 1 for n (the polynomial degree) "
            "from 1 to max_degree. Computed offline with sympy.factorint via "
            "scripts/generate_mersenne_factor_table.py -- do not edit by hand."
        ),
        "max_degree": args.max_degree,
        "factors": table,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=0))
    print(f"Wrote {len(table)} entries to {args.out}")


if __name__ == "__main__":
    main()
