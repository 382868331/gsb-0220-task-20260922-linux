"""NB2 negative-binomial regression for count measurements (stdlib only)."""

from .nb2 import (
    NB2Result,
    NBFitError,
    fit_nb2,
    nb2_loglik,
    nb2_objective,
    nb2_pmf,
    LOG_ALPHA_MIN,
    LOG_ALPHA_MAX,
    MAX_SAMPLES,
    MAX_COLUMNS,
)
from .zinb import (
    ZINBResult,
    ZINBFitError,
    fit_zinb,
    zinb_loglik,
    zinb_objective,
    zinb_pmf,
    LOGIT_PI_MIN,
    LOGIT_PI_MAX,
)

__all__ = [
    "NB2Result",
    "NBFitError",
    "fit_nb2",
    "nb2_loglik",
    "nb2_objective",
    "nb2_pmf",
    "LOG_ALPHA_MIN",
    "LOG_ALPHA_MAX",
    "MAX_SAMPLES",
    "MAX_COLUMNS",
    "ZINBResult",
    "ZINBFitError",
    "fit_zinb",
    "zinb_loglik",
    "zinb_objective",
    "zinb_pmf",
    "LOGIT_PI_MIN",
    "LOGIT_PI_MAX",
]
