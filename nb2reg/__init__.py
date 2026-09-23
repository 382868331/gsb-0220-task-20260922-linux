"""nb2reg: offline NB2 negative-binomial regression, standard library only.

Public surface
--------------
- fit_nb2              joint MLE over (beta, log theta), or fixed alpha
- nb2_loglik           log-likelihood at a finite parameter point
- nb2_gradient         analytic gradient (beta..., or beta..., log theta)
- log_pmf / pmf        NB2 probability mass
- range_probability    sum of masses over a (possibly unbounded) range
- conditional_pmf      truncated (zero-truncated style) probability
- NB2Result            fitted parameters, loglik, gradient, diagnostics
- NB2InputError, RankDeficientDesignError
- NoFiniteMLEError     proof that no finite MLE exists
- ConvergenceBudgetError / StepFailureError
- LIMITS               documented numerical and interface bounds
"""

from .core import (
    LIMITS,
    NB2InputError,
    NB2Result,
    ConvergenceBudgetError,
    NoFiniteMLEError,
    RankDeficientDesignError,
    StepFailureError,
    conditional_pmf,
    fit_nb2,
    log_pmf,
    nb2_gradient,
    nb2_loglik,
    pmf,
    range_probability,
)
from .linalg import design_rank
from .special import digamma, trigamma

__version__ = "1.0.0"

__all__ = [
    "LIMITS",
    "NB2InputError",
    "NB2Result",
    "ConvergenceBudgetError",
    "NoFiniteMLEError",
    "RankDeficientDesignError",
    "StepFailureError",
    "conditional_pmf",
    "fit_nb2",
    "log_pmf",
    "nb2_gradient",
    "nb2_loglik",
    "pmf",
    "range_probability",
    "design_rank",
    "digamma",
    "trigamma",
    "__version__",
]
