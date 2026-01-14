"""SO(N) decomposition into Givens rotations.

This module implements a decomposition of a real special-orthogonal matrix
:math:`U \\in SO(N)` into a sequence of real Givens rotations and a final
diagonal matrix with entries :math:`\\pm 1`.

The algorithm follows the layout of the Clements et al. interferometer
decomposition (Optica 3, 1460–1465, 2016), specialized here to real-valued
orthogonal matrices.

The public API is intentionally small:

* :class:`GivensRotation` – a single real Givens rotation acting on two modes.
* :func:`so_decomposition` – factor an :math:`N  \\times N` matrix in :math:`SO(N)`.
* :func:`reconstruct` – reconstruct :math:`U` from its factors.
* :func:`plot_decomposition` – visualize the layout of rotations.
* :func:`check_decomposition` – quick numerical sanity check.

All indices in this module are **zero-based**, as is idiomatic in Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GivensRotation:
    """Real Givens rotation acting on two modes.

    The rotation acts on a 2D subspace spanned by basis vectors
    ``e[n]`` and ``e[m]`` and is the identity on all other modes.

    In matrix form the non-trivial 2×2 block is

    .. math::

        R = \\begin{pmatrix}
                \\cos\\theta & -\\sin\\theta \\\\
                \\sin\\theta &  \\cos\\theta
            \\end{pmatrix},

    so that the full rotation is ``R`` embedded into rows/columns
    ``(n, m)`` of an ``N x N`` identity.

    Attributes:
        n: Zero-based index of the first mode (row/column).
        m: Zero-based index of the second mode (row/column), ``j > i``.
        theta: Rotation angle in radians.
    """

    n: int
    m: int
    theta: float

    def __repr__(self) -> str:
        return f"GivensRotation(i={self.n}, j={self.m}, theta={self.theta:.4f})"

    def matrix(self, size: int) -> np.ndarray:
        """Return the dense matrix representation of the rotation.

        Args:
            size: Dimension ``N`` of the ambient :math:`SO(N)` matrix.

        Returns:
            (N, N) real orthogonal matrix representing this rotation.
        """
        if not (0 <= self.n < size and 0 <= self.m < size):
            raise ValueError(
                f"Rotation indices (i={self.n}, j={self.m}) "
                f"are out of bounds for size={size}."
            )

        R = np.eye(size, dtype=float)
        c = float(np.cos(self.theta))
        s = float(np.sin(self.theta))
        n, m = self.n, self.m

        R[n, n] = c
        R[n, m] = -s
        R[m, n] = s
        R[m, m] = c

        return R


def so_decomposition(
    U: np.ndarray,
    atol: float = 1e-10,
) -> Tuple[List[GivensRotation], np.ndarray]:
    """Decompose a real special-orthogonal matrix into Givens rotations.

    This implements a Clements-style scheme specialized to real matrices:

    .. math::

        U = D \\prod_k G_k,

    where each :math:`G_k` is a real Givens rotation and :math:`D` is a
    real diagonal matrix with entries :math:`\\pm 1`.

    Args:
        U: (N, N) real-valued matrix that should lie in :math:`SO(N)`.
        atol: Absolute tolerance used to validate orthogonality.

    Returns:
        Tuple (G, D) where

        * ``G`` is a list of :class:`GivensRotation` objects ordered so
          that::

              reconstruct(G, D) ~= U

        * ``D`` is an (N, N) real diagonal NumPy array with entries
          :math:`\\pm 1`.

    Raises:
        ValueError: If ``U`` is not square, not (approximately) orthogonal,
            or its determinant is not (approximately) +1.
    """
    U = np.array(U, dtype=float, copy=True)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("Input matrix U must be square (N x N).")

    N = U.shape[0]

    # Basic SO(N) validation.
    if not np.allclose(U.T @ U, np.eye(N), atol=atol):
        raise ValueError(
            "Input matrix U must be orthogonal: U.T @ U ~= I within "
            f"tolerance atol={atol}."
        )

    det = np.linalg.det(U)
    if not np.isclose(det, 1.0, atol=atol):
        raise ValueError(
            f"Input matrix must have determinant +1 (SO(N)). Got det(U)={det:.6f}."
        )

    rotations: List[GivensRotation] = []
    reverse_layer_rotations: List[GivensRotation] = []

    eps = 1e-12  # numerical zero for pivot checks

    # Main Clements-style sweep
    for layer in range(N - 1):
        if layer % 2 == 0:
            # Forward layer: act on columns from the right.
            for j in range(layer + 1):
                col1 = layer - j
                col2 = layer + 1 - j
                row = N - 1 - j

                num = U[row, col1]
                den = U[row, col2]

                if abs(den) < eps:
                    theta = np.sign(num) * (np.pi / 2.0)
                else:
                    # Use atan2 for better numerical stability.
                    theta = float(np.arctan2(num, den))

                c = float(np.cos(theta))
                s = float(np.sin(theta))

                G_inv = np.eye(N, dtype=float)
                G_inv[col1, col1] = c
                G_inv[col1, col2] = s
                G_inv[col2, col1] = -s
                G_inv[col2, col2] = c

                # Right multiplication modifies columns col1, col2.
                U = U @ G_inv
                rotations.append(GivensRotation(col1, col2, theta))
        else:
            # Reverse layer: act on rows from the left.
            for j in range(layer + 1):
                row1 = N + j - layer - 2
                row2 = N + j - layer - 1
                col = j

                num = U[row2, col]
                den = U[row1, col]

                if abs(den) < eps:
                    theta = np.sign(num) * (np.pi / 2.0)
                else:
                    # Minus sign matches the original implementation.
                    theta = float(-np.arctan2(num, den))

                c = float(np.cos(theta))
                s = float(np.sin(theta))

                G = np.eye(N, dtype=float)
                G[row1, row1] = c
                G[row1, row2] = -s
                G[row2, row1] = s
                G[row2, row2] = c

                # Left multiplication modifies rows row1, row2.
                U = G @ U
                reverse_layer_rotations.append(GivensRotation(row1, row2, theta))

    # Final diagonalisation pass.
    for rot in reversed(reverse_layer_rotations):
        n = rot.n
        m = rot.m

        # Undo the previous reverse-layer rotation.
        c = float(np.cos(rot.theta))
        s = float(np.sin(rot.theta))
        G_inv = np.eye(N, dtype=float)
        G_inv[n, n] = c
        G_inv[n, m] = s
        G_inv[m, n] = -s
        G_inv[m, m] = c
        U = G_inv @ U

        # Now choose a new rotation to zero the off-diagonal element.
        num = U[m, n]
        den = U[m, m]

        if abs(den) < eps:
            theta = np.sign(num) * (np.pi / 2.0)
        else:
            theta = float(np.arctan2(num, den))

        c = float(np.cos(theta))
        s = float(np.sin(theta))
        G_inv[n, n] = c
        G_inv[n, m] = s
        G_inv[m, n] = -s
        G_inv[m, m] = c
        U = U @ G_inv

        rotations.append(GivensRotation(n, m, theta))

    # At this point U should be (numerically) diagonal with ±1 entries.
    diag = np.diag(U)
    # Map extremely small values to +1, otherwise use the sign.
    diag_sign = np.where(diag >= 0.0, 1.0, -1.0)
    D = np.diag(diag_sign.astype(float))

    return rotations, D


def reconstruct(G: List[GivensRotation], D: np.ndarray) -> np.ndarray:
    """Reconstruct a matrix from its Givens rotations and diagonal factor.

    Args:
        G: List of Givens rotations as returned by :func:`so_decomposition`.
        D: (N, N) real diagonal matrix with entries :math:`\\pm 1`.

    Returns:
        (N, N) NumPy array approximating the original matrix.
    """
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square diagonal matrix.")

    N = D.shape[0]
    U = np.eye(N, dtype=float)

    for rot in G:
        U = rot.matrix(N) @ U

    U = D @ U
    return U


def check_decomposition(
    U: np.ndarray,
    atol: float = 1e-10,
    verbose: bool = True,
) -> float:
    """Compute and verify an SO(N) decomposition.

    This is a small helper for experiments and debugging. It runs
    :func:`so_decomposition`, reconstructs :math:`U` and returns the Frobenius
    norm of the difference.

    Args:
        U: Input matrix expected to be in :math:`SO(N)`.
        atol: Tolerance passed down to :func:`so_decomposition`.
        verbose: If True, prints the reconstruction error.

    Returns:
        Frobenius norm of the reconstruction error
        ``||U - reconstruct(G, D)||_F``.
    """
    G, D = so_decomposition(U, atol=atol)
    U_rec = reconstruct(G, D)
    err = float(np.linalg.norm(U - U_rec))

    if verbose:
        print(f"Reconstruction error: {err:.3e}")

    return err


def plot_decomposition(G: List[GivensRotation], size: Optional[int] = None) -> None:
    """Visualize a sequence of Givens rotations.

    Each mode is drawn as a horizontal line (y-axis), and each rotation is
    drawn as a vertical segment connecting the two modes it acts on. The
    x-axis is an abstract "layer" index derived from the order of rotations.

    Args:
        G: Sequence of Givens rotations.
        size: Total number of modes ``N``. If None (default), the
            smallest value consistent with the rotations in ``G`` is used.
    """
    if not G:
        raise ValueError("Cannot plot an empty list of rotations.")

    if size is None:
        size = max(max(rot.n, rot.m) for rot in G) + 1

    depth = np.zeros(size, dtype=int)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    for rot in G:
        n, m = rot.n, rot.m
        x = max(depth[n], depth[m])
        y1 = n + 1  # 1-based for nicer plotting
        y2 = m + 1

        ax.plot(
            [x, x],
            [y1, y2],
            ls="-",
            zorder=1,
            marker="o",
            markersize=8,
            color="tab:red",
        )
        ax.text(
            x + 0.1,
            y2 - 0.4,
            s=f"$\\theta$ = {rot.theta:.2f}",
            fontsize=8,
        )

        depth[n] += 1
        depth[m] += 1

    for mode in range(size):
        ax.hlines(mode + 1, -0.5, depth.max(), ls="--", zorder=0, color="black")

    ax.set_ylim(0.5, size + 0.5)
    ax.set_xlim(-0.5, depth.max() + 0.5)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
