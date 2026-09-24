"""NB2 and zero-inflated NB2 count regression (stdlib only)."""

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
    fit_zinb,
    zinb_loglik,
    zinb_objective,
    zinb_pmf,
    zinb_structural_probability,
    LOGIT_PI_MIN,
    LOGIT_PI_MAX,
)

__all__ = [
    "NB2Result",
    "ZINBResult",
    "NBFitError",
    "fit_nb2",
    "nb2_loglik",
    "nb2_objective",
    "nb2_pmf",
    "fit_zinb",
    "zinb_loglik",
    "zinb_objective",
    "zinb_pmf",
    "zinb_structural_probability",
    "LOG_ALPHA_MIN",
    "LOG_ALPHA_MAX",
    "LOGIT_PI_MIN",
    "LOGIT_PI_MAX",
    "MAX_SAMPLES",
    "MAX_COLUMNS",
]
