"""
Provides preconditioners for applying CG to the linear system of the observation covariance.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict
from functools import partial
import torch
from torch import Tensor
from ..probabilistic_models import BaseObservationCov, LowRankObservationCov
from .preconditioner_utils import pivoted_cholesky

class BasePreconditioner(ABC):
    """
    Base preconditioner class.

    The preconditioner commonly is an approximation of the system matrix, and its inverse is
    applied to the residuals to solve an equivalent system with improved condition with conjugate
    gradients.

    The closure returned by :meth:`get_closure` multiplies with the inverse (like needed for
    preconditioning), i.e. it performs a :meth:`matmul` with ``use_inverse=True``.
    """

    def __init__(self,
            **update_kwargs,
            ) -> None:
        """
        Call :meth:`update`.

        Parameters
        ----------
        **update_kwargs : dict
            Keyword arguments passed to :meth:`update`.
        """
        self.update(**update_kwargs)

    @abstractmethod
    def sample(self,
            num_samples: int
            ) -> Tensor:
        """
        Sample from a Gaussian with the covariance given by ``self`` (e.g., matrix-vector products
        of the Cholesky factor of ``self`` with normal distributed samples) and mean zero.

        Parameters
        ----------
        num_samples : int
            Number of samples.

        Returns
        -------
        Tensor
            Samples. Shape: ``(num_samples, size)``.
        """
        raise NotImplementedError

    @abstractmethod
    def matmul(self,
            v: Tensor,
            use_inverse: bool,
            ) -> Tensor:
        """
        Multiply with self or its inverse.

        Evaluates ``mat @ v`` where ``mat`` is a matrix representation of ``self`` or its inverse.

        Parameters
        ----------
        v : Tensor
            Vectors. Shape: ``(size, batch_size)``.
        use_inverse : bool
            If ``True``, multiply with the inverse.

        Returns
        -------
        Tensor
            Products. Shape: same as ``v``.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self,
            **kwargs
            ) -> None:
        """
        Update ``self``, e.g. adapting the approximation to an updated matrix.
        """
        raise NotImplementedError

    def get_closure(self) -> Callable[[Tensor], Tensor]:
        """
        Return a preconditioning closure that multiplies with the inverse of ``self``.

        Returns
        -------
        closure : callable
            Closure multiplying with the inverse of ``self``. The closure receives and returns
            tensors of shape ``(size, batch_size)``.
        """
        return partial(self.matmul, use_inverse=True)

class LowRankObservationCovPreconditioner(BasePreconditioner):
    """
    Preconditioner employing a :class:`LowRankObservationCov` instance to approximate an observation
    covariance.
    """

    def __init__(self,
            low_rank_observation_cov: LowRankObservationCov,
            default_update_kwargs: Optional[dict] = None,
            ):  # pylint: disable=super-init-not-called
        """
        Parameters
        ----------
        low_rank_observation_cov : class:`bayes_dip.probabilistic_models.LowRankObservationCov`
            Low-rank approximation of an observation covariance. It is expected to be updated
            already, i.e. this constructor does *not* call :meth:`update`.
        default_update_kwargs : dict
            Default keyword arguments passed to :meth:`LowRankObservationCov.update` when calling
            :meth:`update` (may be enhanced and/or overridden by keyword arguments passed to
            :meth:`update`).
        """

        self.low_rank_observation_cov = low_rank_observation_cov
        self.default_update_kwargs = default_update_kwargs or {}
        # do not call super().__init__(), low_rank_observation_cov should be updated already

    def sample(self,
            num_samples: int,
            ) -> Tensor:
        return self.low_rank_observation_cov.sample(num_samples=num_samples, flat=True)

    def matmul(self,
            v: Tensor,
            use_inverse: bool = False,
            ) -> Tensor:
        with torch.no_grad():
            v = self.low_rank_observation_cov.matmul(v, use_inverse=use_inverse)
        return v

    def update(self, **kwargs) -> None:
        """
        Call :meth:`LowRankObservationCov.update`.

        The keyword arguments :attr:`default_update_kwargs` are passed, optionally updated with the
        passed keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments, enhancing and/or overriding :attr:`default_update_kwargs`.
        """
        merged_kwargs = self.default_update_kwargs.copy()
        merged_kwargs.update(kwargs)
        self.low_rank_observation_cov.update(**merged_kwargs)

class JacobiPreconditioner(BasePreconditioner):
    """
    Jacobi (diagonal) preconditioner.
    """

    def __init__(self,
            vector: Tensor,
            ):
        """
        Parameters
        ----------
        vector : Tensor
            Jacobi (diagonal) vector. Shape: ``(size,)``.
        """

        super().__init__(vector=vector)

    def sample(self,
            num_samples: int
            ) -> Tensor:
        normal_std = torch.randn(
            self.vector.shape[0], num_samples,
            device=self.vector.device
            )
        return self.vector[:, None].pow(0.5) * normal_std

    def matmul(self,
            v: Tensor,
            use_inverse: bool,
            ) -> Tensor:
        return (self.vector.pow(-1) if use_inverse else self.vector)[:, None] * v

    def update(self,
            **kwargs
            ) -> None:
        """
        Update the Jacobi (diagonal) vector.

        Parameters
        ----------
        vector : Tensor
            New diagonal vector. Shape: ``(size,)``.
            Must be passed as a keyword argument.
        """
        self.vector = kwargs['vector']

    def get_closure(self) -> Callable[[Tensor], Tensor]:
        return partial(self.matmul, use_inverse=True)

class IncompleteCholeskyPreconditioner(BasePreconditioner):
    """
    Preconditioner using the incomplete cholesky factorization for a matrix approximated as
    ``ichol @ ichol.T + noise * torch.eye(ichol.shape[0])``, like implemented in
    https://github.com/cornellius-gp/linear_operator/blob/987df55260afea79eb0590c7e546b221cfec3fe5/linear_operator/operators/added_diag_linear_operator.py#L84
    """

    def __init__(self,
            incomplete_cholesky: Tensor,
            log_noise_variance: Tensor,
            ):
        """
        Parameters
        ----------
        incomplete_cholesky : Tensor
            Incomplete Cholesky factor. Shape: ``(size, low_rank_rank_dim)``.
        log_noise_variance : Tensor
            Logarithm of the noise variance (cf. :attr:`ObservationCov.log_noise_variance`).
        """

        super().__init__(
                incomplete_cholesky=incomplete_cholesky,
                log_noise_variance=log_noise_variance)

    def sample(self,
            num_samples: int
            ) -> Tensor:
        """Not implemented."""
        raise NotImplementedError

    def matmul(self,
            v: Tensor,
            use_inverse: bool,
            ) -> Tensor:
        if not use_inverse:
            # matmul with covariance
            v = (self.incomplete_cholesky @ (self.incomplete_cholesky.T @ v)
                    + self.log_noise_variance.exp() * v)
        else:
            # matmul with inverse
            qqt_v = self._q_cache.matmul(self._q_cache.mT.matmul(v))
            v = self.log_noise_variance.exp().pow(-1) * (v - qqt_v)
        return v

    def update(self,
            **kwargs
            ) -> None:
        """
        Update the incomplete Cholesky factor and the logarithm of the noise variance.

        Parameters
        ----------
        incomplete_cholesky : Tensor
            Incomplete Cholesky factor. Shape: ``(size, low_rank_rank_dim)``.
            Must be passed as a keyword argument.
        log_noise_variance : Tensor
            Logarithm of the noise variance (cf. :attr:`ObservationCov.log_noise_variance`).
            Must be passed as a keyword argument.
        """
        self.incomplete_cholesky = kwargs['incomplete_cholesky']
        self.log_noise_variance = kwargs['log_noise_variance']

        n, k = self.incomplete_cholesky.shape

        _q_cache, _r_cache = torch.linalg.qr(
                torch.cat((self.incomplete_cholesky,
                        torch.exp(0.5 * self.log_noise_variance) * torch.eye(k,
                                dtype=self.incomplete_cholesky.dtype,
                                device=self.incomplete_cholesky.device)), dim=-2))
        self._q_cache = _q_cache[:n, :]

    def get_closure(self) -> Callable[[Tensor], Tensor]:
        return partial(self.matmul, use_inverse=True)


def get_preconditioner(observation_cov: BaseObservationCov, kwargs: Dict) -> BasePreconditioner:
    """
    Return a preconditioner for observation covariance.

    Parameters
    ----------
    kwargs : dict
        Keyword arguments.
        ``kwargs['name']`` is required, options are:

        * ``kwargs['name'] == 'low_rank_eig'``
            Return a :class:`LowRankObservationCovPreconditioner` instance.
            Further (required) keyword arguments are ``'batch_size'``, ``'low_rank_rank_dim'`` and
            ``'oversampling_param'`` (see :class:`LowRankObservationCovPreconditioner`).
        * ``kwargs['name'] == 'incomplete_chol'``
            Return an :class:`IncompleteCholeskyPreconditioner` instance.
            Further (required) keyword arguments are ``'batch_size'``, ``'low_rank_rank_dim'`` and
            ``'approx_diag_num_samples'`` (see :class:`IncompleteCholeskyPreconditioner`).

    Returns
    -------
    preconditioner : :class:`BasePreconditioner`
        Preconditioner instance.
    """
    if kwargs['name'] == 'low_rank_eig':
        update_kwargs = {'batch_size': kwargs['batch_size']}
        low_rank_observation_cov = LowRankObservationCov(
                trafo=observation_cov.trafo,
                image_cov=observation_cov.image_cov,
                low_rank_rank_dim=kwargs['low_rank_rank_dim'],
                oversampling_param=kwargs['oversampling_param'],
                requires_grad=False,
                device=observation_cov.device,
                **update_kwargs,
        )
        preconditioner = LowRankObservationCovPreconditioner(
                low_rank_observation_cov=low_rank_observation_cov,
                default_update_kwargs=update_kwargs,
        )
    elif kwargs['name'] == 'incomplete_chol':
        ichol, _ = pivoted_cholesky(
                closure=observation_cov.get_closure(),
                size=observation_cov.shape[0],
                max_iter=kwargs['low_rank_rank_dim'],
                approx_diag_num_samples=kwargs['approx_diag_num_samples'],
                batch_size=kwargs['batch_size'],
                error_tol=0.,  # always use kwargs['low_rank_rank_dim'] dimensions
                recompute_max_diag_values=True,
                device=observation_cov.device,
                )
        preconditioner = IncompleteCholeskyPreconditioner(
                incomplete_cholesky=ichol, log_noise_variance=observation_cov.log_noise_variance)
    else:
        raise ValueError(f'Unknown preconditioner name "{kwargs["name"]}".')

    return preconditioner
