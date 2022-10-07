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
            oversampling_param: int = 10,
            load_state_dict: Optional[Union[str, dict]] = None,
            load_approx_basis: Optional[Union[str, dict]] = None,
            requires_grad=True,
            device=None,
            **update_kwargs,
            ) -> None:
        """
        Parameters
        ----------
        trafo : :class:`bayes_dip.data.BaseRayTrafo`
            Ray transform.
        image_cov : :class:`bayes_dip.probabilistic_models.BaseImageCov`
            Image space covariance module.
        init_noise_variance : float, optional
            Initial value for noise variance parameter. The default is ``1.``.
        low_rank_rank_dim : int, optional
            Number of dimensions of the low-rank approximation. The default is ``100``.
        oversampling_param : int, optional
            Number of oversampling dimensions for the low-rank approximation. The default is ``10``.
        load_state_dict : str or dict, optional
            State dict (or path to it) to load.
        load_approx_basis : str or dict, optional
            Approximate basis (or path to it) to load.
            Only supported if ``requires_grad=False``. If using this option, ``load_state_dict`` is
            required, and the user is responsible for consistency.
        requires_grad : bool, optional
            Whether gradient computation should be enabled. Note that the parameters are only used
            in :meth:`update`, so computations will use the values from the last :meth:`update`
            call, and back-propagating from the output of any method (e.g. :meth:`forward`)
            repeatedly, without calling :meth:`update` in between, requires ``retain_graph=True``.
            The default is ``True``.
        device : str or torch.device, optional
            Device. If ``None`` (the default), ``'cuda:0'`` is chosen if available or ``'cpu'``
            otherwise.
        update_kwargs : dict, optional
            Keyword arguments passed to :meth:`update`.
        """
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

    def load_approx_basis(self, filepath_or_dict: Union[str, dict]) -> None:
        """
        Load approximate basis. Only supported if ``not self.requires_grad``.

        Parameters
        ----------
        filepath_or_dict : str or dict
            Approximate basis (or path to it) to load.
        """
        assert not self.requires_grad
        d = filepath_or_dict
        if not isinstance(d, dict):
            d = torch.load(d, map_location=self.device)
        self.U, self.L, self.noise_stddev, self.noise_variance_obs_and_eps, self.sysmat = (
                d['U'], d['L'], d['noise_stddev'], d['noise_variance_obs_and_eps'], d['sysmat'])

    def save_approx_basis(self, filepath: str) -> None:
        """
        Save approximate basis to file.

        Parameters
        ----------
        filepath : str
            Path for saving.
        """
        torch.save({
                'U': self.U.detach().cpu(), 'L': self.L.detach().cpu(),
                'noise_stddev': self.noise_stddev,
                'noise_variance_obs_and_eps': self.noise_variance_obs_and_eps.detach().cpu(),
                'sysmat': self.sysmat.detach().cpu()},
                filepath)

    def _assemble_random_matrix(self) -> Tensor:
        low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param
        random_matrix = torch.randn(
                (low_rank_rank_dim, np.prod(self.trafo.obs_shape)),
                device=self.device)
        return random_matrix

    def get_low_rank_observation_cov_basis(self,
        use_cpu: bool = False,
        eps: float = 1e-1,
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
            The default is ``False``.
        eps : float, optional
            Minimum value eigenvalues.
            The default is ``1e-3``.
        verbose : bool, optional
            If ``True``, print eigenvalue information. The default is ``True``.
        batch_size : int, optional
            Batch size for multiplying with the observation covariance.
            The default is ``1``.

        Returns
        -------
        U : Tensor
            Eigenvectors. Shape: ``(np.prod(self.trafo.obs_shape), self.low_rank_rank_dim)``
        L : Tensor
            Eigenvalues. Shape: ``(self.low_rank_rank_dim)``
        """

        num_batches = ceil((self.low_rank_rank_dim + self.oversampling_param) / batch_size)
        # step 2 in Algorithm 4.1 of Halko et al.
        v_cov_obs_mat = []
        # image_cov might require grads (shared module), so disable grads if not self.requires_grad
        with torch.set_grad_enabled(self.requires_grad):
            for i in tqdm(range(num_batches), miniters=num_batches//100,
                    desc='get_low_rank_observation_cov_basis'):
                rnd_vect = self.random_matrix[i * batch_size:(i+1) * batch_size, None, None, :]
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
        v_cov_obs_mat = v_cov_obs_mat.view(*v_cov_obs_mat.shape[:1], -1).T  # new shape: (dy, L)
        # step 3 in Algorithm 4.1 of Halko et al.
        Q, _ = torch.linalg.qr(
                v_cov_obs_mat.cpu() if use_cpu else v_cov_obs_mat)  # shape: (dy, L), assuming L<=dy
        Q = Q if not use_cpu else Q.to(self.device)
        Q = Q[:, :self.low_rank_rank_dim]
        # step 1 in Algorithm 5.6 of Halko et al.
        B = torch.linalg.lstsq(self.random_matrix @ Q, v_cov_obs_mat.T @ Q).solution
        # step 2 in Algorithm 5.6 of Halko et al.
        L, V = torch.linalg.eig(B)
        # step 3 in Algorithm 5.6 of Halko et al.
        U = Q @ V.real
        if verbose:
            print(
                    f'L.min: {L.real.min()}, '
                    f'L.max: {L.real.max()}, '
                    f'L.num_vals_below_{eps}: {(L.real < eps).sum()}\n')
        return U, L.real.clamp(min=eps)

    def forward(self,
            v: Tensor,
            use_inverse: bool = False,
        ) -> Tensor:
        """
        Multiply with the covariance "matrix" (or its inverse).

        This is a non-flat version of :meth:`forward`, i.e. the same as
        ``self.matmul(v.view(v.shape[0], -1).T, use_inverse=use_inverse).T.view(*v.shape)``.

        Parameters
        ----------
        v : Tensor
            Observations. Shape: ``(batch_size, 1, *self.trafo.obs_shape)``.
        use_inverse : bool, optional
            If ``True``, multiply with the inverse instead. The default is ``False``.

        Returns
        -------
        Tensor
            Products. Shape: same as ``v``.
        """
        batch_size = v.shape[0]
        v = v.view(batch_size, -1).T
        v = self.matmul(v, use_inverse=use_inverse)
        v = v.T.view(batch_size, 1, *self.trafo.obs_shape)
        return v

    def matmul(self,
            v: Tensor,
            use_inverse: bool = False,
        ) -> Tensor:
        """
        Multiply with the covariance "matrix" (or its inverse).

        Evaluates ``mat @ v`` where ``mat`` is a matrix representation of ``self`` (or its inverse).

        See also the non-flat version: :meth:`forward`.

        Parameters
        ----------
        v : Tensor
            Observations. Shape: ``(np.prod(self.trafo.obs_shape), batch_size)``.
        use_inverse : bool, optional
            If ``True``, multiply with the inverse instead. The default is ``False``.

        Returns
        -------
        Tensor
            Products. Shape: same as ``v``.
        """

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
        eps: float = 1e-1,
        full_diag_eps: float = 1e-6,
        batch_size: int = 1,
        ) -> None:
        """
        Update the low-rank approximation and other state variables to the current parameter values.

        This is the only function directly involving parameters.

        Parameters
        ----------
        use_cpu : bool, optional
            Whether to compute QR on CPU.
            The default is ``False``.
        eps : float, optional
            Minumum value eigenvalues.
            The default is ``1e-3``.
        full_diag_eps : float, optional
            Value to add to the noise variance in :meth:`matmul` (for stabilization).
            The default is ``1e-6``.
        batch_size : int, optional
            Batch size for multiplying with the observation covariance.
            The default is ``1``.
        """

        self.U, self.L = self.get_low_rank_observation_cov_basis(
                use_cpu=use_cpu, eps=eps, batch_size=batch_size)
        self.noise_stddev = self.log_noise_variance.exp().pow(0.5)  # for self.sample()
        self.noise_variance_obs_and_eps = self.log_noise_variance.exp() + full_diag_eps
        self.sysmat = torch.diag(1 / self.L) + self.U.T @ self.U / self.noise_variance_obs_and_eps

    def sample(self,
        num_samples: int,
        flat: bool = False,
        ) -> Tensor:
        """
        Sample from a Gaussian with this covariance and mean zero.

        Parameters
        ----------
        num_samples : int
            Number of samples.
        flat : bool, optional
            If ``True``, return a flattened tensor with ``num_samples`` as second dimension (see the
            return value).

        Returns
        -------
        Tensor
            Samples.
            Shape: Either ``(num_samples, 1, *self.trafo.obs_shape)`` if ``not flat`` (the default),
            or ``(np.prod(self.trafo.obs_shape), num_samples)`` if ``flat``.
        """

        normal_std = torch.randn(
            self.shape[0], num_samples,
            device=self.device
            )
        normal_low_rank = torch.randn(
            self.low_rank_rank_dim, num_samples,
            device=self.device
            )
        samples = normal_std * self.noise_stddev + (self.U * self.L.pow(0.5) ) @ normal_low_rank
        return samples if flat else samples.T.view(num_samples, 1, *self.trafo.obs_shape)
