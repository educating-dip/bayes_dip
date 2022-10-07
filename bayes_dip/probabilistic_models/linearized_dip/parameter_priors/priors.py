"""
Provides priors that can be placed over parameters of a neural network.
"""
from typing import Any, Callable, Sequence, Tuple, Dict, List, Union
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
    """
    Base Gaussian prior class for :class:`nn.Conv2d` modules.

    The prior is placed over all filters of ``modules`` (passed to :meth:`__init__`),
    i.e. all input-channel-output-channel combinations of all modules share the prior.

    Implements efficient batched multiplication with the covariance matrices of multiple priors of
    the same type via the class method :meth:`batched_cov_mul`.
    """

    def __init__(self,
            init_hyperparams: Dict[str, Any],
            modules: Sequence[nn.Conv2d],
            device: Union[str, torch.device]):
        """
        Parameters
        ----------
        init_hyperparams : dict
            Initial values for the prior hyperparameters.
            As a convention, if the actual parameter stores a log value, the initial values are
            still passed as the non-log value, e.g. for :attr:`log_variance`: ``'variance': 1.``.
        modules : sequence of :class:`nn.Conv2d`
            Modules to place this prior over. The prior applies to all filters in all ``weight``
            parameters of these convolutional layers.
        device : str or torch.device
            Device.
        """
        super().__init__()
        self.device = device
        self.modules = modules
        self._setup(self.modules)
        self._init_parameters(init_hyperparams)

    @classmethod
    def get_params_under_prior_from_modules(cls,
            modules: Sequence[nn.Conv2d]) -> List[nn.Parameter]:
        """
        Return the list of all parameters under a prior of this type given the ``modules`` it would
        be placed over.

        Parameters
        ----------
        modules : sequence of :class:`nn.Conv2d`
            Modules, like the argument to :meth:`__init__`.
        """
        return [module.weight for module in modules]

    def get_params_under_prior(self) -> List[nn.Parameter]:
        """Return all parameters under this prior."""
        return self.get_params_under_prior_from_modules(modules = self.modules)

    def _setup(self, modules: List[nn.Conv2d]) -> None:
        """
        Setup callback, called in :meth:`__init__` (before :meth:`_init_parameters`).
        May be used to create parameter attributes.
        """

        self.kernel_size = modules[0].kernel_size[0]
        for layer in modules:
            assert isinstance(layer, nn.Conv2d)
            assert layer.kernel_size[0] == layer.kernel_size[1]
            assert layer.kernel_size[0] == self.kernel_size

        self.num_total_filters = sum(
                layer.in_channels * layer.out_channels for layer in modules)

    @abstractmethod
    def _init_parameters(self, init_hyperparams: Dict) -> None:
        """
        Initialization callback, called in :meth:`__init__` (after :meth:`_setup`).

        Should initialize the hyperparameters of this prior using the ``init_hyperparams`` argument
        to :meth:`__init__`.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def log_variance(self) -> nn.Parameter:
        """Log variance hyperparameter (an :class:`nn.Parameter` instance)"""
        raise NotImplementedError

    @abstractmethod
    def sample(self, shape: Tuple[int]) -> Tensor:
        """
        Draw samples from the prior in a differentiable way (e.g. via :meth:`Distribution.rsample`).

        Parameters
        ----------
        shape : tuple of int
            Sample shape (e.g. ``(n,)`` to draw ``n`` samples).

        Returns
        -------
        samples : Tensor
            Samples. Shape: ``(*shape, self.kernel_size ** 2)``.
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, x: Tensor) -> Tensor:
        """
        Return the log probability of ``x`` under this prior.

        Parameters
        ----------
        x : Tensor
            Value.

        Returns
        -------
        log_prob : Tensor
            Log probability.
        """
        raise NotImplementedError

    @abstractmethod
    def cov_mat(self) -> Tensor:
        """
        Return the covariance matrix.

        Returns
        -------
        cov_mat : Tensor
            Covariance matrix. Shape: ``(self.kernel_size ** 2, self.kernel_size ** 2)``.
        """
        raise NotImplementedError

    @abstractmethod
    def cov_log_det(self) -> Tensor:
        """
        Return the log determinant of the covariance matrix.
        """
        raise NotImplementedError

    @classmethod
    def _fast_prior_cov_mul(cls,
            v: Tensor,
            cov: Tensor
        ) -> Tensor:
        """
        Multiply concatenated weight vectors with a stack of covariance matrices.

        Parameters
        ----------
        v : Tensor
            Batch of flattened and concatenated ``weight`` parameters of :class:`Conv2d` modules.
            Shape: ``(batch_size, n * kernel_size ** 2)``, where ``n`` is the sum of the numbers of
            filters of all modules.
        cov : Tensor
            Covariance matrices.
            Shape: ``(n, kernel_size ** 2, kernel_size ** 2)``, where ``n`` is the sum of the
            numbers of filters of all modules.

        Returns
        -------
        v_cov_mul : Tensor
            Products. Has the same shape as ``v``.
        """
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
            **cov_mat_kwargs,
        ) -> Tensor:
        """
        Multiply weight vectors with a stack of covariance matrices.

        Parameters
        ----------
        priors : sequence of :class:`BaseGaussPrior`
            Sequence of priors of the same type.
            The covariance matrix of each prior is repeated ``prior.num_total_filters`` times,
            and the matrices of all priors are then concatenated.
        v : Tensor
            Batch of flattened and concatenated ``weight`` parameters of :class:`Conv2d` modules.
            Shape: ``(batch_size, n * kernel_size ** 2)``, where ``n`` is the sum of the numbers of
            filters of all modules of all ``priors``.
        use_cholesky : bool, optional
            If ``True``, multiply with the Cholesky factor instead. Cannot be combined with
            ``use_inverse``.
            The default is ``False``.
        use_inverse : bool, optional
            If ``True``, multiply with the inverse instead. Cannot be combined with
            ``use_cholesky``.
            The default is ``False``.
        cov_mat_kwargs : dict, optional
            Keyword arguments passed to :meth:`cov_mat` of each prior (e.g. an ``eps`` value).

        Returns
        -------
        v_cov_mul : Tensor
            Products. Has the same shape as ``v``.
        """
        assert not (use_cholesky and use_inverse)

        cov = []
        for prior in priors:
            cov_mat = prior.cov_mat(return_cholesky=use_cholesky, **cov_mat_kwargs)
            if use_inverse:
                cov_mat = torch.inverse(cov_mat)
            cov.append(cov_mat.expand(
                    prior.num_total_filters, prior.kernel_size**2, prior.kernel_size**2))
        cov = torch.cat(cov)
        return cls._fast_prior_cov_mul(v, cov)

class RadialBasisFuncCov(nn.Module):
    """
    Covariance function based on a distance function between kernel pixel coordinates:

    ``cov(coords0, coords1) = variance * exp(-dist_func(coords0, coords1) / lengthscale)``.
    """

    # forward is not used, but we still use the nn.Module base to contain the ``nn.Parameter``s
    # pylint: disable=abstract-method

    def __init__(
        self,
        kernel_size: int,
        dist_func: Callable[[Tensor], float],
        device
        ):
        """
        Parameters
        ----------
        kernel_size : int
            Convolution kernel size.
        dist_func : callable
            Distance function receiving coordinates
            ``torch.as_tensor([i, j], dtype=torch.float32)`` and returning a scalar.
        device : str or torch.device
            Device.
        """

        super().__init__()

        self.device = device
        self.log_lengthscale = nn.Parameter(torch.ones(1, device=self.device))
        self.log_variance = nn.Parameter(torch.ones(1, device=self.device))

        self.kernel_size = kernel_size
        self.dist_mat = self._compute_dist_matrix(dist_func)

    def init_parameters(self,
            lengthscale_init: float,
            variance_init: float
            ) -> None:
        """
        Initialize the parameters.

        Parameters
        ----------
        lengthscale_init : float
            Initial lengthscale value.
        variance_init : float
            Initial variance value.
        """
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
            eps: float = 1e-6
            ) -> Tensor:
        """
        Return the unscaled covariance matrix (excluding the variance scaling factor).

        Parameters
        ----------
        eps : float, optional
            Stabilizing value that is added to the diagonal. The default is ``1e-6``.

        Returns
        -------
        unscaled_cov_mat : Tensor
            Unscaled covariance matrix. Shape: ``(self.kernel_size ** 2, self.kernel_size ** 2)``.
        """
        lengthscale = torch.exp(self.log_lengthscale)
        assert not torch.isnan(lengthscale)
        cov_mat = torch.exp(-self.dist_mat / lengthscale) + eps * torch.eye(
                *self.dist_mat.shape, device=self.device)
        return cov_mat

    def cov_mat(self,
            return_cholesky: bool = False,
            eps: float = 1e-6
            ) -> Tensor:
        """
        Return the covariance matrix.

        Parameters
        ----------
        return_cholesky : bool, optional
            If ``True``, return the cholesky factor instead of the matrix itself.
        eps : float, optional
            Stabilizing value that is added to the diagonal before scaling with the variance.
            The default is ``1e-6``.

        Returns
        -------
        cov_mat_or_chol : Tensor
            Covariance matrix, or its cholesky factor if ``return_cholesky``.
            Shape: ``(self.kernel_size ** 2, self.kernel_size ** 2)``.
        """
        variance = torch.exp(self.log_variance)
        assert not torch.isnan(variance)
        cov_mat = self.unscaled_cov_mat(eps=eps)
        cov_mat = variance * cov_mat
        return (cholesky(cov_mat) if return_cholesky else cov_mat)

    def log_det(self) -> Tensor:
        """
        Return the log determinant of the covariance matrix.
        """
        return 2 * self.cov_mat(return_cholesky=True).diag().log().sum()

    def log_lengthscale_cov_mat_grad(self) -> Tensor:
        """
        Return the derivative of the covariance matrix w.r.t. :attr:`log_lengthscale`.
        """
        # we multiply by the lengthscale value (chain rule)
        return self.dist_mat * self.cov_mat(return_cholesky=False) / torch.exp(self.log_lengthscale)

    def log_variance_cov_mat_grad(self) -> Tensor:
        """
        Return the derivative of the covariance matrix w.r.t. :attr:`log_variance`.
        """
        # we multiply by the variance value (chain rule)
        return self.cov_mat(return_cholesky=False, eps=1e-6)

class GPprior(BaseGaussPrior):
    """Gaussian prior."""

    def __init__(self,
            init_hyperparams: Dict,
            modules: List[nn.Conv2d],
            covariance_constructor: Callable,
            device
            ):
        """
        Parameters
        ----------
        init_hyperparams : dict
            Initial prior hyperparameter values. Keys are ``'lengthscale'`` and ``'variance'``.
        modules : sequence of :class:`nn.Conv2d`
            Modules to place this prior over. The prior applies to all filters in all ``weight``
            parameters of these convolutional layers.
        covariance_constructor : callable
            Callable with arguments ``kernel_size`` and ``device`` returning a covariance object
            with parameters ``log_variance`` and ``log_lengthscale`` and methods
            ``init_parameters()``, ``cov_mat()``, ``log_det()``, ``log_lengthscale_cov_mat_grad()``
            and ``log_variance_cov_mat_grad()`` (see :class:`RadialBasisFuncCov` for an example
            implementation).
        device : str or torch.device
            Device.
        """

        self.covariance_constructor = covariance_constructor

        super().__init__(init_hyperparams, modules, device)

    def _init_parameters(self, init_hyperparams: Dict) -> None:

        self.cov.init_parameters(
                lengthscale_init=init_hyperparams['lengthscale'],
                variance_init=init_hyperparams['variance'])

    def _setup(self, modules):

        super()._setup(modules=modules)
        self.cov = self.covariance_constructor(kernel_size=self.kernel_size,
                device=self.device)

    @property
    def log_variance(self):
        """Log variance hyperparameter."""
        return self.cov.log_variance

    @property
    def log_lengthscale(self):
        """Log lengthscale hyperparameter."""
        return self.cov.log_lengthscale

    def sample(self, shape: Tuple[int]) -> Tensor:
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.kernel_size ** 2).to(self.device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        params_shape = (*shape, self.kernel_size,
                                self.kernel_size)
        return m.rsample(sample_shape=shape).view(params_shape)

    def log_prob(self, x: Tensor) -> Tensor:
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.kernel_size ** 2).to(self.device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        return m.log_prob(x)

    def cov_mat(self,  # pylint: disable=arguments-differ
            return_cholesky: bool = False,
            eps: float = 1e-6
            ) -> Tensor:
        return self.cov.cov_mat(return_cholesky=return_cholesky, eps=eps)

    def cov_log_det(self) -> Tensor:
        return self.cov.log_det()

    def cov_log_lengthscale_grad(self) -> Tensor:
        """
        Return the derivative of the covariance matrix w.r.t. :attr:`log_lengthscale`.
        """
        return self.cov.log_lengthscale_cov_mat_grad()

    def cov_log_variance_grad(self) -> Tensor:
        """
        Return the derivative of the covariance matrix w.r.t. :attr:`log_variance`.
        """
        return self.cov.log_variance_cov_mat_grad()

def get_GPprior_RadialBasisFuncCov(init_hyperparams, modules, device, dist_func=None):
    """
    Return a :class:`GPprior` instance with a :class:`RadialBasisFuncCov` covariance with
    ``dist_func`` defaulting to the Euclidean distance.

    Parameters
    ----------
    init_hyperparams : dict
        Initial prior hyperparameter values. Keys are ``'lengthscale'`` and ``'variance'``.
    modules : sequence of :class:`nn.Conv2d`
        Modules to place this prior over. The prior applies to all filters in all ``weight``
        parameters of these convolutional layers.
    device : str or torch.device
        Device.
    dist_func : callable, optional
        Distance function receiving coordinates
        ``torch.as_tensor([i, j], dtype=torch.float32)`` and returning a scalar.
        The default is ``lambda x: linalg.norm(x, ord=2)``.
    """
    dist_func = dist_func or ( lambda x: linalg.norm(x, ord=2) )
    covariance_constructor = partial(RadialBasisFuncCov, dist_func=dist_func)
    return GPprior(
            init_hyperparams=init_hyperparams,
            modules=modules,
            covariance_constructor=covariance_constructor,
            device=device
        )

class NormalPrior(BaseGaussPrior):
    """Normal prior."""

    def _setup(self, modules):

        super()._setup(modules=modules)
        self._log_variance = nn.Parameter(
                torch.ones(1, device=self.device)
            )

    @property
    def log_variance(self):
        """Log variance hyperparameter."""
        return self._log_variance

    def _init_parameters(self, init_hyperparams: Dict) -> None:
        nn.init.constant_(self.log_variance, np.log(init_hyperparams['variance']))

    def sample(self, shape: Tuple[int]) -> Tensor:
        mean = torch.zeros(self.kernel_size ** 2, device=self.device)
        m = Normal(loc=mean, scale=torch.exp(self.log_variance)**.5)
        return m.rsample(sample_shape=shape)

    def log_prob(self, x: Tensor) -> Tensor:
        mean = torch.zeros(self.kernel_size ** 2, device=self.device)
        m = Normal(loc=mean, scale=torch.exp(self.log_variance)**.5)
        return m.log_prob(x)

    def cov_mat(self,  # pylint: disable=arguments-differ
            return_cholesky: bool = False,
            eps: float = 1e-6
            ) -> Tensor:
        eye = torch.eye(self.kernel_size ** 2).to(self.device)
        fct = (
                torch.exp(0.5 * self.log_variance) if return_cholesky else
                torch.exp(self.log_variance))
        cov_mat = fct * (eye + eps)
        return cov_mat

    def cov_log_det(self) -> Tensor:
        return self.log_variance * self.kernel_size ** 2

class IsotropicPrior(BaseGaussPrior):
    """Isotropic prior (g-prior)."""

    def _setup(self, modules):
        self._log_variance = nn.Parameter(
                torch.ones(1, device=self.device)
        )

    @property
    def log_variance(self):
        """Log variance hyperparameter."""
        return self._log_variance

    @log_variance.setter
    def log_variance(self, value):
        self._log_variance.data[:] = value

    def _init_parameters(self, init_hyperparams: Dict) -> None:
        nn.init.constant_(self.log_variance, np.log(init_hyperparams['variance']))

    def sample(self, shape: Tuple[int]) -> Tensor:
        raise NotImplementedError

    def log_prob(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def cov_mat(self, use_cholesky: bool = False) -> Tensor:
        raise NotImplementedError

    def cov_log_det(self) -> Tensor:
        raise NotImplementedError

    @classmethod
    def batched_cov_mul(cls,  # pylint: disable=arguments-differ
            priors: Sequence,
            v: Tensor,
            use_cholesky: bool = False,
            use_inverse: bool = False,
            eps: float = 1e-6,
        ) -> Tensor:
        """
        Scale weight vectors with the variance.

        Parameters
        ----------
        priors : 1-sequence of :class:`IsotropicPrior`
            A single :class:`IsotropicPrior`, wrapped in a sequence of length ``1``.
            This function is not implemented for more than one prior.
        v : Tensor
            Batch of flattened and concatenated ``weight`` parameters of :class:`Conv2d` modules.
            Shape: ``(batch_size, n * kernel_size ** 2)``, where ``n`` is the sum of the numbers of
            filters of all modules of ``priors[0]``.
        use_cholesky : bool, optional
            If ``True``, multiply with the Cholesky factor instead. Cannot be combined with
            ``use_inverse``.
            The default is ``False``.
        use_inverse : bool, optional
            If ``True``, multiply with the inverse instead. Cannot be combined with
            ``use_cholesky``.
            The default is ``False``.
        eps : float, optional
            Stabilizing value used with ``use_inverse``, added to the scale before taking the
            inverse.
            The default is ``1e-6``.

        Returns
        -------
        v_cov_mul : Tensor
            Products. Has the same shape as ``v``.
        """

        assert not (use_cholesky and use_inverse)
        if len(priors) != 1:
            raise NotImplementedError

        prior = priors[0]

        scale = prior.log_variance.exp() if not use_cholesky else (.5*prior.log_variance).exp()
        if use_inverse:
            scale = (scale + eps).pow(-1)
        return scale * v
