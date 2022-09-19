"""
Provides the marginal log-likelihood (MLL or Type-II-MAP) optimization of the prior hyperparameters.
"""
from .mll_optim import marginal_likelihood_hyperparams_optim
from .preconditioner import (
        BasePreconditioner, LowRankObservationCovPreconditioner, get_preconditioner)
from .weights_linearization import weights_linearization
from .utils import get_ordered_nn_params_vec
