"""NB2 negative-binomial regression for count measurements (stdlib only)."""

from .nb2 import (
    NB2Result,
    NBFitError,
    fit_nb2,
    fit_nb2_censored,
    nb2_loglik,
    nb2_objective,
    nb2_censored_loglik,
    nb2_censored_objective,
    nb2_interval_logprob,
    nb2_pmf,
    LOG_ALPHA_MIN,
    LOG_ALPHA_MAX,
    MAX_SAMPLES,
    MAX_COLUMNS,
)

__all__ = [
    "NB2Result",
    "NBFitError",
    "fit_nb2",
    "fit_nb2_censored",
    "nb2_loglik",
    "nb2_objective",
    "nb2_censored_loglik",
    "nb2_censored_objective",
    "nb2_interval_logprob",
    "nb2_pmf",
    "LOG_ALPHA_MIN",
    "LOG_ALPHA_MAX",
    "MAX_SAMPLES",
    "MAX_COLUMNS",
]
