"""Provides :class:`LowRankObservationCov`"""
from typing import Optional, Union
from math import ceil
import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm

from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from ..base_image_cov import BaseImageCov
from ..observation_cov import BaseObservationCov


class LowRankObservationCov(BaseObservationCov):

    """
    Covariance in observation space using low-rank matrix approximation.
    The algorithm is described in [1]_.

    .. [1] N. Halko, P.G. Martinsson, and J.A. Tropp, 2011, "Finding Structure with Randomness:
           probabilistic algorithms for constructing approximate matrix decompositions".
           SIAM Review. https://doi.org/10.1137/090771806.
    """

    def __init__(self,
            trafo: BaseRayTrafo,
            image_cov: BaseImageCov,
            init_noise_variance: float = 1.,
            low_rank_rank_dim: int = 100,
            oversampling_param: int = 5,
            load_state_dict: Optional[Union[str, dict]] = None,
            load_approx_basis: Optional[Union[str, dict]] = None,
            requires_grad=True,
            device=None,
            **update_kwargs,
            ) -> None:
        # pylint: disable=too-many-arguments

        assert not (load_approx_basis is not None and load_state_dict is None), (
                '`load_state_dict` is required when using `load_approx_basis`')

        super().__init__(
                trafo=trafo,
                image_cov=image_cov,
                init_noise_variance=init_noise_variance,
                device=device
        )

        if load_state_dict is not None:
            if not isinstance(load_state_dict, dict):
                load_state_dict = torch.load(load_state_dict, map_location=self.device)
            self.load_state_dict(load_state_dict)

        self.requires_grad = requires_grad
        self.log_noise_variance.requires_grad_(self.requires_grad)
        # can't call self.image_cov.requires_grad_() because it might be shared, will disable grads
        # locally when using self.image_cov

        self.low_rank_rank_dim = low_rank_rank_dim
        self.oversampling_param = oversampling_param
        self.random_matrix = self._assemble_random_matrix()

        if load_approx_basis is None:
            self.update(**update_kwargs)
        else:
            self.load_approx_basis(load_approx_basis)

    def load_approx_basis(self, filepath_or_dict: Union[str, dict]):
        d = filepath_or_dict
        if not isinstance(d, dict):
            d = torch.load(d, map_location=self.device)
        self.U, self.L, self.noise_variance_obs_and_eps, self.sysmat = (
                d['U'], d['L'], d['noise_variance_obs_and_eps'], d['sysmat'])

    def save_approx_basis(self, filepath: str):
        torch.save({
                'U': self.U.detach().cpu(), 'L': self.L.detach().cpu(),
                'noise_variance_obs_and_eps': self.noise_variance_obs_and_eps.detach().cpu(),
                'sysmat': self.sysmat.detach().cpu()},
                filepath)

    def _assemble_random_matrix(self, ) -> Tensor:
        low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param
        random_matrix = torch.randn(
                (low_rank_rank_dim, np.prod(self.trafo.obs_shape)),
                device=self.device)
        return random_matrix

    def get_low_rank_observation_cov_basis(self,
        use_cpu: bool = False,
        eps: float = 1e-3,
        verbose: bool = True,
        batch_size: int = 1,
        ):

        """
        Eigenvalue Decomposition in One Pass.

        This method implements Algo. 5.6 from Halko et al. and computes an
        approximate eigenvalue decomposition of the low-rank term of
        covariance in observation space.

        Parameters
        ----------
        use_cpu : bool, optional
            Whether to compute QR on CPU.
            The default is `False`.
        eps : float, optional
            Minumum value eigenvalues.
            The default is 1e-3.

        Returns
        -------
        U : Tensor
            Output. Shape: ``(np.prod(self.trafo.obs_shape), self.low_rank_rank_dim)``
        L : Tensor
            Output. Shape: ``(self.low_rank_rank_dim)``
        """

        num_batches = ceil((self.low_rank_rank_dim + self.oversampling_param) / batch_size)
        v_cov_obs_mat = []
        # image_cov might require grads (shared module), so disable grads if not self.requires_grad
        with torch.set_grad_enabled(self.requires_grad):
            for i in tqdm(range(num_batches), miniters=num_batches//100,
                    desc='get_cov_obs_low_rank'):
                rnd_vect = self.random_matrix[i * batch_size:(i+1) * batch_size, :].unsqueeze(dim=1)
                eff_batch_size = rnd_vect.shape[0]
                if eff_batch_size < batch_size:
                    rnd_vect = torch.cat(
                        [rnd_vect, torch.zeros(
                                (batch_size-eff_batch_size, *rnd_vect.shape[1:]),
                                    dtype=rnd_vect.dtype,
                                    device=rnd_vect.device)
                            ]
                        )
                # apply observation cov without noise variance term
                v = self.trafo.trafo_adjoint(rnd_vect)
                v = self.image_cov(v)
                v = self.trafo(v)
                if eff_batch_size < batch_size:
                    v = v[:eff_batch_size]
                v_cov_obs_mat.append(v)
        v_cov_obs_mat = torch.cat(v_cov_obs_mat)
        v_cov_obs_mat = v_cov_obs_mat.view(*v_cov_obs_mat.shape[:1], -1).T
        Q, _ = torch.linalg.qr(v_cov_obs_mat.cpu() if use_cpu else v_cov_obs_mat)
        Q = Q if not use_cpu else Q.to(self.device)
        B = torch.linalg.solve(self.random_matrix @ Q, v_cov_obs_mat.T @ Q)
        L, V = torch.linalg.eig(B)
        U = Q @ V.real
        if verbose:
            print(
                    f'L.min: {L.real[:self.low_rank_rank_dim].min()}, '
                    f'L.max: {L.real[:self.low_rank_rank_dim].max()}, '
                    f'L.num_vals_below_{eps}: {(L.real[:self.low_rank_rank_dim] < eps).sum()}\n')
        return U[:, :self.low_rank_rank_dim], L.real[:self.low_rank_rank_dim].clamp(min=eps)

    def forward(self,
            v: Tensor,
            use_inverse: bool = False,
        ) -> Tensor:
        batch_size = v.shape[0]
        v = v.view(batch_size, -1).T
        v = self.matmul(v, use_inverse=use_inverse)
        v = v.T.view(batch_size, 1, *self.trafo.obs_shape)
        return v

    def matmul(self,
            v: Tensor,
            use_inverse: bool = False,
        ) -> Tensor:

        if not use_inverse:
            # matmul with covariance
            v = (self.U @ (self.L[:, None] * (self.U.T @ v))
                    + self.noise_variance_obs_and_eps * v)
        else:
            # matmul with inverse
            v = (v / self.noise_variance_obs_and_eps) - (self.U @ torch.linalg.solve(
                    self.sysmat, self.U.T @ v / (self.noise_variance_obs_and_eps ** 2)))
        return v

    def update(self,
        use_cpu: bool = False,
        eps: float = 1e-3,
        full_diag_eps: float = 1e-6,
        batch_size: int = 1,
        ) -> None:

        self.U, self.L = self.get_low_rank_observation_cov_basis(
                eps=eps, use_cpu=use_cpu, batch_size=batch_size)
        self.noise_variance_obs_and_eps = self.log_noise_variance.exp() + full_diag_eps
        self.sysmat = torch.diag(1 / self.L) + self.U.T @ self.U / self.noise_variance_obs_and_eps

    def sample(self,
        num_samples: int = 10,
        ) -> Tensor:

        normal_std = torch.randn(
            self.shape[0], num_samples,
            device=self.device
            )
        normal_low_rank = torch.randn(
            self.low_rank_rank_dim, num_samples,
            device=self.device
            )
        samples = (normal_std * self.log_noise_variance.exp().pow(0.5)
                + (self.U * self.L.pow(0.5) ) @ normal_low_rank)
        return samples
