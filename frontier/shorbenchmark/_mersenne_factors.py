"""Lookup table of complete prime factorizations of 2**n - 1.

The primitive-polynomial search in shorbenchmark.py needs the full set of
distinct prime factors of ``order = 2**degree - 1`` for each candidate
polynomial's degree. Trial-dividing ``order`` at runtime is only feasible for
small degrees; for larger degrees it is astronomically slow. Instead, the
factorizations are precomputed offline (see
scripts/generate_mersenne_factor_table.py, which uses sympy) and stored in
_mersenne_factors.json, checked into the repo.

Only degrees for which 2**n - 1 is *completely* factored are present in the
table. A missing entry means the factorization is not known/available, not
that it should be treated as having no factors -- callers must not silently
substitute a partial result, since the primitivity test requires every prime
factor of ``order`` to be correct.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

_DATA_PATH = Path(__file__).with_name("_mersenne_factors.json")


@lru_cache(maxsize=1)
def _load_table() -> Dict[int, Tuple[int, ...]]:
    payload = json.loads(_DATA_PATH.read_text())
    return {int(n): tuple(factors) for n, factors in payload["factors"].items()}


def mersenne_prime_factors(degree: int) -> Tuple[int, ...]:
    """Return the distinct prime factors of ``2**degree - 1``.

    Raises KeyError if the factorization for ``degree`` is not present in the
    precomputed table.
    """
    table = _load_table()
    try:
        return table[degree]
    except KeyError as exc:
        raise KeyError(
            f"No precomputed factorization of 2**{degree} - 1 is available. "
            f"The table currently covers degrees up to {max(table)}. Extend "
            "it with scripts/generate_mersenne_factor_table.py."
        ) from exc
