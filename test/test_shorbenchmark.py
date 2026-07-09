from __future__ import annotations

import pytest

from frontier import ShorPeriodFindingBenchmark
from frontier.shorbenchmark import shorbenchmark as _shor
from frontier.shorbenchmark._mersenne_factors import mersenne_prime_factors


def _is_primitive_reference(poly: int) -> bool:
    """Slow, self-contained reference primitivity test via trial division.

    Independent of the production code path (in particular, does not use
    mersenne_prime_factors), used to cross-check _is_primitive_polynomial.
    """

    def prime_factors(value: int) -> list[int]:
        factors = []
        divisor = 2
        remaining = value
        while divisor**2 <= remaining:
            if remaining % divisor == 0:
                factors.append(divisor)
                while remaining % divisor == 0:
                    remaining //= divisor
            divisor += 1 if divisor == 2 else 2
        if remaining > 1:
            factors.append(remaining)
        return factors

    degree = _shor._poly_degree(poly)
    if degree <= 0:
        return False
    if (poly & 1) == 0 or ((poly >> degree) & 1) == 0:
        return False

    x_poly = 0b10
    for prime_factor in prime_factors(degree):
        exp = 1 << (degree // prime_factor)
        test_poly = _shor._poly_pow_mod(x_poly, exp, poly) ^ x_poly
        if _shor._poly_gcd(poly, test_poly) != 1:
            return False

    if _shor._poly_pow_mod(x_poly, 1 << degree, poly) != x_poly:
        return False

    order = (1 << degree) - 1
    for prime_factor in prime_factors(order):
        if _shor._poly_pow_mod(x_poly, order // prime_factor, poly) == 1:
            return False

    return True


@pytest.mark.parametrize("degree", range(1, 15))
def test_is_primitive_polynomial_matches_reference(degree: int) -> None:
    for middle_bits in range(1 << max(degree - 1, 0)):
        poly = (1 << degree) | 1 | (middle_bits << 1)
        assert _shor._is_primitive_polynomial(poly) == _is_primitive_reference(poly)


def test_mersenne_prime_factors_divide_order() -> None:
    for degree in [1, 2, 3, 8, 17, 64, 200]:
        order = (1 << degree) - 1
        for prime_factor in mersenne_prime_factors(degree):
            assert order % prime_factor == 0


def test_mersenne_prime_factors_out_of_range_raises() -> None:
    with pytest.raises(KeyError):
        mersenne_prime_factors(10_000)


@pytest.mark.parametrize("degree", [3, 8, 32, 64, 128, 200])
def test_primitive_polynomial_for_degree_is_primitive(degree: int) -> None:
    poly = _shor._primitive_polynomial_for_degree(degree, sample_id=0)
    assert _shor._is_primitive_polynomial(poly)
    assert _shor._poly_degree(poly) == degree


def test_primitive_polynomial_for_degree_is_deterministic() -> None:
    for degree, sample_id in [(20, 0), (20, 1), (64, 3)]:
        first = _shor._primitive_polynomial_for_degree(degree, sample_id)
        second = _shor._primitive_polynomial_for_degree(degree, sample_id)
        assert first == second


def test_primitive_polynomial_for_degree_varies_with_sample_id() -> None:
    polys = {
        sample_id: _shor._primitive_polynomial_for_degree(20, sample_id)
        for sample_id in range(10)
    }
    assert len(set(polys.values())) > 1


def test_primitive_polynomial_for_degree_rejects_invalid_degree() -> None:
    with pytest.raises(ValueError):
        _shor._primitive_polynomial_for_degree(0)


def test_primitive_polynomial_for_degree_beyond_table_raises() -> None:
    with pytest.raises(KeyError):
        _shor._primitive_polynomial_for_degree(10_000, sample_id=0)

