"""Provides :class:`ObservationCov` and :class:`MatmulObservationCov`"""
from typing import Optional
from functools import lru_cache
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from bayes_dip.data.trafo import MatmulRayTrafo
from bayes_dip.utils import bisect_left  # for python >= 3.10: from bisect import bisect_left
from .base_image_cov import BaseImageCov
from .base_observation_cov import BaseObservationCov
from ..utils import make_choleskable

class ObservationCov(BaseObservationCov):
    """
    Covariance in observation space.
    """

    def forward(self,
                v: Tensor,
                use_noise_variance: bool = True,
                use_cholesky: bool = False,
                **kwargs
            ) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, 1, *self.trafo.obs_shape)``
        use_noise_variance : bool, optional
            Whether to include the noise variance diagonal term.
            The default is ``True``.
        use_cholesky : bool, optional
            Whether to multiply with one cholesky factor instead of the full matrix.
            If ``True``, :meth:`self.image_cov.forward` must support and implement the argument
            ``use_cholesky=True`` analogously.
            The default is ``False``.
        kwargs : dict, optional
            Keyword arguments passed to :meth:`self.image_cov.forward`.

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, 1, *self.trafo.obs_shape)``
        """
        assert not (use_noise_variance and use_cholesky)

        v_ = self.trafo.trafo_adjoint(v)
        if use_cholesky:
            v_ = self.image_cov(v_, use_cholesky=True, **kwargs)
        else:
            v_ = self.image_cov(v_, **kwargs)
            v_ = self.trafo(v_)

        v = v_ + v * self.log_noise_variance.exp() if use_noise_variance else v_

        return v

    def assemble_observation_cov(self,
        batch_size: int = 1,
        use_noise_variance: bool = True,
        sub_slice_batches: Optional[slice] = None,
        ) -> Tensor:
        """
        Parameters
        ----------
        batch_size : int, optional
            Batch size. The default is ``1``.
        use_noise_variance : bool, optional
            Whether to include the noise variance diagonal term. The default is ``True``.
        sub_slice_batches : slice, optional
            If specified, only assemble the specified subset (slice) of batches.
            Note that the slicing indices apply to the *batches* of rows (not the rows themselves).
            Useful to parallelize the assembly.

        Returns
        -------
        Tensor
            Assembled matrix. Shape: ``(np.prod(self.trafo.obs_shape),) * 2``.
        """

        obs_numel = np.prod(self.trafo.obs_shape)
        if sub_slice_batches is None:
            sub_slice_batches = slice(None)
        rows = []
        v = torch.empty((batch_size, 1, *self.trafo.obs_shape), device=self.device)

        with torch.no_grad():
            for i in tqdm(np.array(range(0, obs_numel, batch_size))[sub_slice_batches],
                        desc='get_prior_cov_obs_mat', miniters=obs_numel//batch_size//100
                    ):

                v[:] = 0.
                # set v.view(batch_size, -1) to be a subset of rows of torch.eye(obs_numel);
                # in last batch, it may contain some additional (zero) rows
                v.view(batch_size, -1)[:, i:i+batch_size].fill_diagonal_(1.)

                rows_batch = self.forward(
                    v,
                    use_noise_variance=use_noise_variance)

                rows_batch = rows_batch.view(batch_size, -1)

                if i+batch_size > obs_numel:  # last batch
                    rows_batch = rows_batch[:obs_numel%batch_size]
                rows_batch = rows_batch.cpu()  # collect on CPU
                rows.append(rows_batch)

        observation_cov_mat = torch.cat(rows, dim=0)

        return observation_cov_mat.to(self.device)

    @classmethod
    def get_stabilizing_eps(cls,
        observation_cov_mat: Tensor,
        eps_mode: str,
        eps: float,
        eps_min_for_auto: float = 1e-6,
        include_zero_for_auto: bool = True
        ) -> float:
        """
        Return a stabilizing epsilon value to add to the diagonal of the covariance matrix.

        Parameters
        ----------
        observation_cov_mat : Tensor
            Assembled observation covariance matrix
            (e.g., returned from :meth:`assemble_observation_cov`).
        eps_mode : str
            Mode for computing the stabilizing ``eps``. Options are:

                * ``'abs'``: ``eps``
                * ``'rel_mean_diag'``: ``eps * observation_cov_mat.diag().mean()``
                * ``'auto_abs'``: as much as needed,
                        up to ``eps``,
                        at least ``eps_min_for_auto`` (or ``0.`` if ``include_zero_for_auto``)
                * ``'auto_rel_mean_diag'``: as much as needed,
                        up to ``eps * observation_cov_mat.diag().mean()``,
                        at least ``eps_min_for_auto * observation_cov_mat.diag().mean()`` (or ``0.``
                        if ``include_zero_for_auto``)

        eps : float
            Absolute or relative epsilon or maximum epsilon, see ``eps_mode``.
        eps_min_for_auto : float, optional
            Minimum absolute or relative epsilon for automatic mode, see ``eps_mode``.
            Must be a positive number (defines the starting point of the log grid to test);
            note that ``include_zero_for_auto`` controls whether zero is also included in the grid.
            The default is ``1e-6``.
        include_zero_for_auto : bool, optional
            Whether to include zero in the grid for automatic mode. The default is ``True``.

        Returns
        -------
        float
            Value that should be added to the diagonal of the covariance matrix.
        """

        observation_cov_mat_diag_mean = observation_cov_mat.diag().mean().item()

        if eps_mode == 'abs':
            observation_cov_mat_eps = eps or 0.
        elif eps_mode == 'rel_mean_diag':
            observation_cov_mat_eps = (eps or 0.) * observation_cov_mat.diag().mean().item()
        elif eps_mode in ('auto_abs', 'auto_rel_mean_diag'):
            @lru_cache(maxsize=None)
            def observation_cov_mat_cholesky_decomposable(eps_value):
                try:
                    _ = torch.linalg.cholesky(
                            observation_cov_mat + eps_value * torch.eye(
                                observation_cov_mat.shape[0],
                                device=observation_cov_mat.device
                        )
                        )
                except RuntimeError:
                    return False
                return True
            fct = 1. if eps_mode == 'auto_abs' else observation_cov_mat_diag_mean
            assert eps >= eps_min_for_auto, "eps_min_for_auto must be lower than eps"
            # both eps and eps_min_for_auto are relative to fct
            eps_to_search = list(np.logspace(np.log10(eps_min_for_auto / eps), 0, 1000) * eps * fct)
            if include_zero_for_auto:
                eps_to_search = [0.] + eps_to_search
            i_eps = bisect_left(eps_to_search, True, key=observation_cov_mat_cholesky_decomposable)

            assert i_eps < len(eps_to_search), ('failed to make Kyy cholesky decomposable,'
                f' max eps is {eps_to_search[-1]} == {eps_to_search[-1] / fct} * Kyy.diag().mean()')

            observation_cov_mat_eps = eps_to_search[i_eps]
        elif eps_mode is None or eps_mode.lower() == 'none':
            observation_cov_mat_eps = 0.
        else:
            raise NotImplementedError

        return observation_cov_mat_eps


class MatmulObservationCov(BaseObservationCov):
    """
    Covariance in observation space computed with assembled Jacobian matrix.

    Use :class:`bayes_dip.probabilistic_models.MatmulNeuralBasisExpansion`
    for the matmul implementation of Jacobian vector products (:meth:`jvp`) and
    vector Jacobian products (:meth:`vjp`).
    """

    def __init__(self,
        trafo: MatmulRayTrafo,
        image_cov: BaseImageCov,
        **kwargs
        ) -> None:

        super().__init__(
                trafo=trafo,
                image_cov=image_cov,
                **kwargs)

        trafo_mat = self.trafo.matrix
        jac_mat = self.image_cov.neural_basis_expansion.matrix
        self.trafo_jac_mat = trafo_mat @ jac_mat
        self.jac_t_trafo_t_mat = self.trafo_jac_mat.T

    def forward(self, v: Tensor, matrix: Tensor = None, apply_make_choleskable=False) -> Tensor:

        if matrix is None:
            matrix = self.get_matrix(apply_make_choleskable=apply_make_choleskable)

        batch_size = v.shape[0]
        v = v.view(
            batch_size, -1, self.shape[0]
        )
        return (v @ matrix).view(
            batch_size, -1, *self.trafo.obs_shape
            )

    def get_matrix(self, apply_make_choleskable=False) -> Tensor:
        """
        Covariance in observation space computed via one explicit matmul.

        This usually leads to more stable numerics than the repeated closure evaluation performed by
        :meth:`ObservationCov.assemble_observation_cov`.

        Returns
        -------
        Tensor
            Observation covariance matrix.
            Shape: ``(np.prod(self.trafo.obs_shape),) * 2``.
        """

        matrix = (
                self.image_cov.inner_cov(self.trafo_jac_mat) @ self.jac_t_trafo_t_mat +
                self.log_noise_variance.exp() * torch.eye(self.shape[0], device=self.device)
            )
        if apply_make_choleskable:
            make_choleskable(matrix)
        return matrix
