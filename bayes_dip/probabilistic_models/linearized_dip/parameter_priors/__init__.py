"""
Provides priors that can be placed over parameters of network modules.

The parameters of a prior itself are called prior hyperparameters.
"""

from .priors import (
        BaseGaussPrior, NormalPrior, GPprior, IsotropicPrior, RadialBasisFuncCov,
        get_GPprior_RadialBasisFuncCov)
