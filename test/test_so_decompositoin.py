# test_so_decomposition.py
import numpy as np
import pytest

from ScalableVolumetricBenchmark import (
    so_decomposition,
    reconstruct,
    check_decomposition,
    GivensRotation,
)


def test_so_decomposition_round_trip():
    """
    Ensure that so_decomposition + reconstruct reproduces the
    original SO(N) matrix to numerical tolerance.
    """
    rng = np.random.default_rng(123)

    # Make a random orthogonal matrix using QR
    A = rng.normal(size=(6, 6))
    Q, R = np.linalg.qr(A)

    # Force det = +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    assert np.isclose(np.linalg.det(Q), 1.0, atol=1e-10)

    # Decompose
    G, D = so_decomposition(Q, atol=1e-10)

    # Reconstruct
    Q_rec = reconstruct(G, D)

    # Check fidelity
    err = np.linalg.norm(Q - Q_rec)

    assert err < 1e-8, f"Round-trip decomposition error too large: {err}"


def test_check_decomposition_returns_small_error():
    """
    check_decomposition() should return a small Frobenius error
    for a valid SO(N) matrix.
    """
    rng = np.random.default_rng(42)

    # Random orthogonal matrix with det = +1
    A = rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(A)

    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    err = check_decomposition(Q, atol=1e-10, verbose=False)

    assert err < 1e-8, f"Expected small reconstruction error, got {err}"
