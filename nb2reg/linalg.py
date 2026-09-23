"""Tiny dense linear algebra helpers (pure standard library).

Only the handful of operations needed by the NB2 fitter are provided:
Cholesky factorisation / positive-definite solve and a modified
Gram--Schmidt rank report used to reject rank-deficient designs with
locatable column indices.
"""

from __future__ import annotations

import math


class NonPositiveDefiniteError(Exception):
    """Raised when a matrix expected to be positive definite is not."""

    def __init__(self, pivot: int = -1, minor: float = 0.0):
        super().__init__(
            f"matrix is not positive definite at pivot {pivot} (minor={minor!r})"
        )
        self.pivot = pivot
        self.minor = minor


def cholesky(A: list[list[float]]) -> list[list[float]]:
    """Return lower-triangular L with A = L L^T.

    Raises NonPositiveDefiniteError if A is not (numerically) positive
    definite.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            if i == j:
                if s <= 0.0 or not math.isfinite(s):
                    raise NonPositiveDefiniteError(i, s)
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def solve_pd(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b for symmetric positive-definite A."""
    L = cholesky(A)
    n = len(A)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= L[i][k] * y[k]
        y[i] = s / L[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= L[k][i] * x[k]
        x[i] = s / L[i][i]
    return x


def design_rank(
    X: list[list[float]],
) -> tuple[int, list[int]]:
    """Rank of the design matrix via modified Gram--Schmidt.

    Returns ``(rank, dependent_columns)``.  ``dependent_columns`` lists
    column indices that lie in the span of earlier pivoted columns
    (including all-zero columns).  Pivoting is by largest residual norm,
    so a reported index is a reproducible locator, not a claim about
    which column an analyst ought to delete.
    """
    n = len(X)
    p = len(X[0])
    cols = [[X[i][j] for i in range(n)] for j in range(p)]
    norms0 = [math.sqrt(sum(v * v for v in c)) for c in cols]
    scale = max(norms0) if norms0 else 0.0
    tol = 1e-12 * max(scale, 1.0) * math.sqrt(max(n, 1))

    remaining = set(range(p))
    dependent: list[int] = []
    rank = 0
    while remaining:
        # pivot: column with largest current norm
        j = max(remaining, key=lambda k: sum(v * v for v in cols[k]))
        nrm2 = sum(v * v for v in cols[j])
        if nrm2 <= tol * tol:
            dependent.extend(sorted(remaining))
            break
        q = [v / math.sqrt(nrm2) for v in cols[j]]
        remaining.remove(j)
        rank += 1
        for k in list(remaining):
            dot = sum(q[i] * cols[k][i] for i in range(n))
            cols[k] = [cols[k][i] - dot * q[i] for i in range(n)]
    return rank, sorted(dependent)
