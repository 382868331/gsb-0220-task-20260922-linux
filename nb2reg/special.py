"""Digamma and trigamma via recurrence plus asymptotic expansion.

``math.lgamma`` is used for log-gamma itself; the polygamma functions are
not in the standard library, so they are implemented here.  Both only
need to be accurate for positive arguments (the NB2 likelihood calls
them at ``1/alpha`` and ``mu_i + 1/alpha``).

Error target (validated in tests against finite-difference ``lgamma``):
relative error <= 1e-11 for arguments >= 1, absolute error <= 1e-11 for
small positive arguments.
"""

from __future__ import annotations

import math

# Shift arguments to at least this large before the asymptotic series.
_SHIFT = 12.0


def digamma(x: float) -> float:
    """Digamma function psi(x) = d/dx log Gamma(x), x > 0."""
    if x <= 0.0:
        raise ValueError(f"digamma requires a positive argument, got {x!r}")
    f = 0.0
    while x < _SHIFT:
        # psi(x) = psi(x+1) - 1/x
        f -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    t = inv * inv
    # psi(x) = log(x) - 1/(2x) - sum B_{2k}/(2k x^{2k})
    series = (
        1.0 / 12.0 * t
        - 1.0 / 120.0 * t**2
        + 1.0 / 252.0 * t**3
        - 1.0 / 240.0 * t**4
        + 5.0 / 660.0 * t**5
        - 691.0 / 32760.0 * t**6
        + 1.0 / 12.0 * t**7
    )
    return f + math.log(x) - 0.5 * inv - series


def trigamma(x: float) -> float:
    """Trigamma function psi'(x), x > 0."""
    if x <= 0.0:
        raise ValueError(f"trigamma requires a positive argument, got {x!r}")
    f = 0.0
    while x < _SHIFT:
        # psi'(x) = psi'(x+1) + 1/x^2
        f += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    t = inv * inv
    # psi'(x) = 1/x + 1/(2x^2) + sum B_{2k}/x^{2k+1}
    series = (
        1.0 / 6.0 * t
        - 1.0 / 30.0 * t * t
        + 1.0 / 42.0 * t**3
        - 1.0 / 30.0 * t**4
        + 5.0 / 66.0 * t**5
        - 691.0 / 2730.0 * t**6
        + 7.0 / 6.0 * t**7
    )
    return f + inv + 0.5 * t + inv * series


# ---------------------------------------------------------------------------
# Stable finite differences at x >> y
#
# Naively evaluating lgamma(x+y)-lgamma(x) (or the polygamma analogues)
# loses all precision once x is huge compared with y: at x=1e7 each
# lgamma is ~1.5e8 while the difference is O(y log x), so floating
# round-off (abs ~1e-7) completely buries the score signal.  For
# integer y these differences admit exact, cancellation-free forms:
#
#   lgamma(x+y) - lgamma(x) = y*log(x) + sum_{j=1}^{y-1} log1p(j/x)
#   psi(x+y) - psi(x)       = sum_{j=0}^{y-1} 1/(x+j)
#   psi'(x+y) - psi'(x)     = -sum_{j=0}^{y-1} 1/(x+j)**2
#
# Above an O(y) cost threshold, Stirling/asymptotic *difference*
# formulas (large arguments at both ends, no near-equal subtraction)
# are used instead.
# ---------------------------------------------------------------------------

_EXACT_SUM_LIMIT = 1_000
_STIRLING_FLOOR = 12.0

# Bernoulli numbers B_2 .. B_16
_B2 = 1.0 / 6.0
_B4 = -1.0 / 30.0
_B6 = 1.0 / 42.0
_B8 = -1.0 / 30.0
_B10 = 5.0 / 66.0
_B12 = -691.0 / 2730.0
_B14 = 7.0 / 6.0
_B16 = -3617.0 / 510.0
_BERNS = [_B2, _B4, _B6, _B8, _B10, _B12, _B14, _B16]


def _stirling_lgamma(z: float) -> float:
    t = (z - 0.5) * math.log(z) - z + 0.5 * math.log(2.0 * math.pi)
    iz = 1.0 / z
    u = iz  # z^{-(2k-1)}, k = 1, 2, ...
    for k, b in enumerate(_BERNS):
        t += b / ((2 * k + 2) * (2 * k + 1)) * u
        u *= iz * iz
    return t


def _stirling_digamma_tail(z: float) -> float:
    iz = 1.0 / z
    t = math.log(z) - 0.5 * iz
    u = 1.0  # z^{-2k}
    for k, b in enumerate(_BERNS):
        u *= iz * iz
        t -= b / (2 * k + 2) * u
    return t


def _stirling_trigamma_tail(z: float) -> float:
    iz = 1.0 / z
    t = iz + 0.5 * iz * iz
    u = iz  # z^{-(2k+1)}
    for k, b in enumerate(_BERNS):
        u *= iz * iz
        t += b * u
    return t


def gamma_ratio_log(x: float, y: int) -> float:
    """Stable log Gamma(x+y) - log Gamma(x) for integer y >= 0, x > 0.

    Small integer y uses the cancellation-free product sum.  For large y
    both endpoints are expanded analytically so the O(x log x) bulk is
    never subtracted numerically; a small lower endpoint falls back to
    ``math.lgamma`` there.
    """
    yi = int(y)
    if yi == 0:
        return 0.0
    if yi <= _EXACT_SUM_LIMIT:
        s = yi * math.log(x)
        for j in range(1, yi):
            s += math.log1p(j / x)
        return s
    x2 = x + yi
    if x >= _STIRLING_FLOOR:
        # y log x + (x+y-.5) log(1+y/x) - y + series diff
        w = yi / x
        lw = math.log1p(w)
        s = yi * math.log(x) + (x2 - 0.5) * lw - yi
        ix = 1.0 / x
        ix2 = 1.0 / x2
        u1 = ix
        u2 = ix2
        for k, b in enumerate(_BERNS):
            s += b / ((2 * k + 2) * (2 * k + 1)) * (u2 - u1)
            u1 *= ix * ix
            u2 *= ix2 * ix2
        return s
    # Endpoints far apart in magnitude: safe direct subtraction.
    return _stirling_lgamma(x2) - math.lgamma(x)


def digamma_diff(x: float, y: int) -> float:
    """Stable psi(x+y) - psi(x) for integer y >= 0, x > 0."""
    yi = int(y)
    if yi == 0:
        return 0.0
    if yi <= _EXACT_SUM_LIMIT:
        return sum(1.0 / (x + j) for j in range(yi))
    x2 = x + yi
    if x >= _STIRLING_FLOOR:
        s = math.log1p(yi / x) + 0.5 * (1.0 / x - 1.0 / x2)
        ix = 1.0 / x
        ix2 = 1.0 / x2
        u1 = 1.0
        u2 = 1.0
        for k, b in enumerate(_BERNS):
            u1 *= ix * ix
            u2 *= ix2 * ix2
            s -= b / (2 * k + 2) * (u2 - u1)
        return s
    return _stirling_digamma_tail(x2) - digamma(x)


def trigamma_diff(x: float, y: int) -> float:
    """Stable psi'(x+y) - psi'(x) (<= 0) for integer y >= 0, x > 0."""
    yi = int(y)
    if yi == 0:
        return 0.0
    if yi <= _EXACT_SUM_LIMIT:
        return -sum(1.0 / ((x + j) * (x + j)) for j in range(yi))
    x2 = x + yi
    if x >= _STIRLING_FLOOR:
        s = (1.0 / x2 - 1.0 / x) + 0.5 * (1.0 / (x2 * x2) - 1.0 / (x * x))
        ix = 1.0 / x
        ix2 = 1.0 / x2
        u1 = ix
        u2 = ix2
        for k, b in enumerate(_BERNS):
            u1 *= ix * ix
            u2 *= ix2 * ix2
            s += b * (u2 - u1)
        return s
    return _stirling_trigamma_tail(x2) - trigamma(x)
