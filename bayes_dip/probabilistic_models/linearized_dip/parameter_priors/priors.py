from typing import Callable, Sequence, Tuple, Dict, List
from abc import ABC, abstractmethod
from functools import partial
import numpy as np
from opt_einsum import contract
import torch
from torch import nn, linalg, Tensor
from torch.linalg import cholesky
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

class BaseGaussPrior(nn.Module, ABC):

    def __init__(self, init_hyperparams, modules, device):
        super().__init__()
        self.device = device
        self.modules = modules
        self._setup(self.modules)
        self._init_parameters(init_hyperparams)

    def get_params_under_prior(self,
            ):
        return [module.weight for module in self.modules]

    def _setup(self,
        modules: List[nn.Conv2d]):

        self.kernel_size = modules[0].kernel_size[0]
        for layer in modules:
            assert isinstance(layer, nn.Conv2d)
            assert layer.kernel_size[0] == layer.kernel_size[1]
            assert layer.kernel_size[0] == self.kernel_size

        self.kernel_size = modules[0].kernel_size[0]
        self.num_total_filters = sum(
                layer.in_channels * layer.out_channels for layer in modules)

    @abstractmethod
    def _init_parameters(self,
            init_hyperparams: Dict,
            ):

        raise NotImplementedError

    @abstractmethod
    def sample(self,
            shape: Tuple,
            ) -> Tensor:

        raise NotImplementedError

    @abstractmethod
    def log_prob(self,
            x: Tensor
            ) -> Tensor:

        raise NotImplementedError

    @abstractmethod
    def cov_mat(self,
            ) -> Tensor :

        raise NotImplementedError

    @abstractmethod
    def cov_log_det(self,
            ) -> Tensor:

        raise NotImplementedError

    @classmethod
    def _fast_prior_cov_mul(cls,
            v: Tensor,
            cov: Tensor
        ) -> Tensor:

        N = v.shape[0]
        v = v.view(-1, cov.shape[0], cov.shape[-1])
        v = v.permute(1, 0, 2)
        v_cov_mul = contract('nxk,nkc->ncx', v, cov)
        v_cov_mul = v_cov_mul.reshape([cov.shape[0] * cov.shape[-1], N]).T
        return v_cov_mul

    @classmethod
    def batched_cov_mul(cls,
            priors: Sequence,
            v: Tensor,
            use_cholesky: bool = False,
            use_inverse: bool = False,
            eps: float = 1e-6,
        ) -> Tensor:

        assert not (use_cholesky and use_inverse)

        cov = []
        for prior in priors:
            cov_mat = prior.cov_mat(return_cholesky=use_cholesky, eps=eps)
            if use_inverse:
                cov_mat = torch.inverse(cov_mat)
            cov.append(cov_mat.expand(
                    prior.num_total_filters, prior.kernel_size**2, prior.kernel_size**2))
        cov = torch.cat(cov)
        return cls._fast_prior_cov_mul(v, cov)

class RadialBasisFuncCov(nn.Module):

    # forward is not used, but we still use the nn.Module base to contain the nn.Parameters
    # pylint: disable=abstract-method

    def __init__(
        self,
        kernel_size: int,
        dist_func: Callable[[Tensor], float],
        device
        ):

        super().__init__()

        self.device = device
        self.log_lengthscale = nn.Parameter(torch.ones(1, device=self.device))
        self.log_variance = nn.Parameter(torch.ones(1, device=self.device))

        self.kernel_size = kernel_size
        self.dist_mat = self._compute_dist_matrix(dist_func)

    def init_parameters(self,
            lengthscale_init: float,
            variance_init: float
            ):

        nn.init.constant_(self.log_lengthscale,
                          np.log(lengthscale_init))
        nn.init.constant_(self.log_variance, np.log(variance_init))

    def _compute_dist_matrix(self,
            dist_func: Callable[[Tensor], float]
            ) -> Tensor:

        coords = [torch.as_tensor([i, j], dtype=torch.float32) for i in
                  range(self.kernel_size) for j in
                  range(self.kernel_size)]
        combs = [[el_1, el_2] for el_1 in coords for el_2 in coords]
        dist_mat = torch.as_tensor([dist_func(el1 - el2) for (el1,
                                   el2) in combs], dtype=torch.float32, device=self.device)
        return dist_mat.view(self.kernel_size ** 2, self.kernel_size ** 2)

    def unscaled_cov_mat(self,
            eps=1e-6
            ) -> Tensor:

        lengthscale = torch.exp(self.log_lengthscale)
        assert not torch.isnan(lengthscale)
        cov_mat = torch.exp(-self.dist_mat / lengthscale) + eps * torch.eye(
                *self.dist_mat.shape, device=self.device)
        return cov_mat

    def cov_mat(self,
            return_cholesky=False,
            eps=1e-6
            ) -> Tensor:

        variance = torch.exp(self.log_variance)
        assert not torch.isnan(variance)
        cov_mat = self.unscaled_cov_mat(eps=eps)
        cov_mat = variance * cov_mat
        return (cholesky(cov_mat) if return_cholesky else cov_mat)

    def log_det(self) -> Tensor:
        return 2 * self.cov_mat(return_cholesky=True).diag().log().sum()

    def log_lengthscale_cov_mat_grad(self) -> Tensor:
        # we multiply by the lengthscale value (chain rule)
        return self.dist_mat * self.cov_mat(return_cholesky=False) / torch.exp(self.log_lengthscale)

    def log_variance_cov_mat_grad(self) -> Tensor:
        # we multiply by the variance value (chain rule)
        return self.cov_mat(return_cholesky=False, eps=1e-6)

class GPprior(BaseGaussPrior):

    def __init__(self,
            init_hyperparams: Dict,
            modules: List[nn.Conv2d],
            covariance_constructor: Callable,
            device
            ):

        self.covariance_constructor = covariance_constructor

        super().__init__(init_hyperparams, modules, device)

    def _init_parameters(self,
            init_hyperparams: Dict
            ):

        self.cov.init_parameters(lengthscale_init=init_hyperparams['lengthscale'],
                    variance_init=init_hyperparams['variance']
                )

    def _setup(self, modules):

        super()._setup(modules=modules)
        self.cov = self.covariance_constructor(kernel_size=self.kernel_size,
                device=self.device)

    def sample(self,
            shape: Tuple,
            ) -> Tensor:
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.cov.kernel_size ** 2).to(self.device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        params_shape = (*shape, self.cov.kernel_size,
                                self.cov.kernel_size)
        return m.rsample(sample_shape=shape).view(params_shape)

    def log_prob(self,
            x: Tensor
            ) -> Tensor:
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.cov.kernel_size ** 2).to(self.device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        return m.log_prob(x)

    def cov_mat(self,  # pylint: disable=arguments-differ
            return_cholesky: bool = False,
            eps: float = 1e-6
            ) -> Tensor:
        return self.cov.cov_mat(return_cholesky=return_cholesky, eps=eps)

    def cov_log_det(self, ) -> Tensor:
        return self.cov.log_det()

    def cov_log_lengthscale_grad(self) -> Tensor:
        # we multiply by the lengthscale value (chain rule)
        return self.cov.log_lengthscale_cov_mat_grad()

    def cov_log_variance_grad(self) -> Tensor:
        # we multiply by the variance value (chain rule)
        return self.cov.log_variance_cov_mat_grad()

def get_GPprior_RadialBasisFuncCov(init_hyperparams, modules, device, dist_func=None):
    dist_func = dist_func or ( lambda x: linalg.norm(x, ord=2) )
    covariance_constructor = partial(RadialBasisFuncCov, dist_func=dist_func)
    return GPprior(
            init_hyperparams=init_hyperparams,
            modules=modules,
            covariance_constructor=covariance_constructor,
            device=device
        )

class NormalPrior(BaseGaussPrior):

    def _setup(self, modules):

        super()._setup(modules=modules)
        self.log_variance = nn.Parameter(
                torch.ones(1, device=self.device)
            )

    def _init_parameters(self,
            init_hyperparams: Dict
            ):
        nn.init.constant_(self.log_variance, np.log(init_hyperparams['variance']))

    def sample(self,
            shape: Tuple
            ) -> Tensor:
        mean = torch.zeros(self.kernel_size, device=self.device)
        m = Normal(loc=mean, scale=torch.exp(self.log_variance)**.5)
        return m.rsample(sample_shape=shape)

    def log_prob(self,
            x: Tensor
            ) -> Tensor:
        mean = torch.zeros(self.kernel_size, device=self.device)
        m = Normal(loc=mean, scale=torch.exp(self.log_variance)**.5)
        return m.log_prob(x)

    def cov_mat(self,  # pylint: disable=arguments-differ
            return_cholesky: bool = False,
            eps: float = 1e-6,
            ) -> Tensor:
        eye = torch.eye(self.kernel_size).to(self.device)
        fct = (
                torch.exp(0.5 * self.log_variance) if return_cholesky else
                torch.exp(self.log_variance))
        cov_mat = fct * (eye + eps)
        return cov_mat

    def cov_log_det(self) -> Tensor:
        return self.log_variance * self.kernel_size
