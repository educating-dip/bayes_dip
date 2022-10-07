"""
Provides :class:`LowRankNeuralBasisExpansion`.
"""

from typing import Optional, Tuple
from math import ceil
import torch
from torch import Tensor
from tqdm import tqdm
from .base_neural_basis_expansion import BaseNeuralBasisExpansion
from .neural_basis_expansion import NeuralBasisExpansion

class LowRankNeuralBasisExpansion(BaseNeuralBasisExpansion):

    """
    Wrapper class for Jacobian vector products and vector Jacobian products using low-rank Jacobian
    matrix approximation.
    The algorithm is described in [1]_.

    .. [1] N. Halko, P.G. Martinsson, and J.A. Tropp, 2011, "Finding Structure with Randomness:
           probabilistic algorithms for constructing approximate matrix decompositions".
           SIAM Review. https://doi.org/10.1137/090771806.
    """

    def __init__(self,
            neural_basis_expansion: NeuralBasisExpansion,
            low_rank_rank_dim: int,
            oversampling_param: int = 10,
            load_from_file: Optional[str] = None,
            device=None,
            batch_size: int = 1,
            use_cpu: bool = False) -> None:
        """
        Parameters
        ----------
        neural_basis_expansion : :class:`bayes_dip.probabilistic_models.NeuralBasisExpansion`
            Neural basis expansion instance to be approximated.
        low_rank_rank_dim : int
            Number of dimensions of the low-rank approximation.
        oversampling_param : int, optional
            Number of oversampling dimensions for the low-rank approximation. The default is ``10``.
        load_from_file : str, optional
            File path to load the approximation from (skipping computation).
        device : str or torch.device, optional
            Device. If ``None`` (the default), ``'cuda:0'`` is chosen if available or ``'cpu'``
            otherwise.
        batch_size : int, optional
            Batch size. The default is ``1``.
        use_cpu : bool, optional
            Whether to perform SVD on CPU. The default is ``False``.
        """

        super().__init__(
                nn_model=neural_basis_expansion.nn_model,
                nn_input=neural_basis_expansion.nn_input,
                ordered_nn_params=neural_basis_expansion.ordered_nn_params,
                nn_out_shape=neural_basis_expansion.nn_out_shape,
            )

        self.neural_basis_expansion = neural_basis_expansion
        self.oversampling_param = oversampling_param
        self.low_rank_rank_dim = low_rank_rank_dim
        self.device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

        if load_from_file is None:
            self.jac_U, self.jac_S, self.jac_Vh = self.get_low_rank_jac_basis(
                    use_cpu=use_cpu, batch_size=batch_size)
        else:
            self.load(load_from_file)

    def load(self, filepath: str) -> None:
        """
        Load the approximation from file.

        Parameters
        ----------
        filepath : str, optional
            File path.
        """
        jac_dict = torch.load(filepath, map_location=self.device)
        self.jac_U, self.jac_S, self.jac_Vh = (
                jac_dict['jac_U'], jac_dict['jac_S'], jac_dict['jac_Vh'])

    def save(self, filepath: str) -> None:
        """
        Save the approximation to file.

        Parameters
        ----------
        filepath : str, optional
            File path.
        """
        torch.save(
                {'jac_U': self.jac_U.cpu(), 'jac_S': self.jac_S.cpu(), 'jac_Vh': self.jac_Vh.cpu()},
                filepath)

    def _assemble_random_matrix(self, ) -> Tensor:
        total_low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param
        random_matrix = torch.randn(
            (self.num_params, total_low_rank_rank_dim),
            device=self.device
            ) # constructing Gaussian random matrix Omega
        return random_matrix

    def get_low_rank_jac_basis(self,
            use_cpu: bool = False, batch_size: int = 1) -> Tuple[Tensor, Tensor, Tensor]:

        """
        Direct SVD.

        This method implements Algo. 5.1 from Halko et al. and computes an
        approximate singular value decomposition of the Jacobian.

        Parameters
        ----------
        use_cpu : bool, optional
            Whether to compute QR on CPU.
            The default is ``False``.
        batch_size : int, optional
            Batch size for multiplying with the Jacobian.
            The default is ``1``.

        Returns
        -------
        U : Tensor
            Orthonormal matrix. Shape: ``(np.prod(self.nn_out_shape), self.low_rank_rank_dim)``
        S : Tensor
            Approximate singular values. Shape: ``(self.low_rank_rank_dim)``
        Vh : Tensor
            Orthonormal matrix. Shape: ``(self.low_rank_rank_dim, self.num_params)``
        """

        random_matrix = self._assemble_random_matrix() # draw a Gaussian random matrix Omega

        total_low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param

        assert total_low_rank_rank_dim <= self.jac_shape[0], (
                'low rank dim must not be larger than network output dimension')

        # Stage 1: Randomized Range Finder
        # (see Algo. 4.1; https://epubs.siam.org/doi/epdf/10.1137/090771806)

        # Identifying a subspace that captures most of the action of the Jacobian. Constructing a
        # matrix Q whose columns form an orthonormal basis for the range of Y = Jac @ Omega.
        num_batches_forward = ceil(total_low_rank_rank_dim / batch_size)
        low_rank_jac_v_mat = []
        for i in tqdm(range(num_batches_forward), miniters=num_batches_forward//100,
                desc='get_low_rank_jac_basis forward'):
            rnd_vect = random_matrix[
                    :, i * batch_size:(i * batch_size) + batch_size]
            low_rank_jac_v_mat_row = self.neural_basis_expansion.jvp(rnd_vect.T)
            low_rank_jac_v_mat.append(
                    low_rank_jac_v_mat_row.cpu() if use_cpu else low_rank_jac_v_mat_row)
        low_rank_jac_v_mat = torch.cat(low_rank_jac_v_mat)
        # Y = Jac @ Omega, (np.prod(self.nn_out_shape[2:]), total_low_rank_rank_dim)
        low_rank_jac_v_mat = low_rank_jac_v_mat.view(low_rank_jac_v_mat.shape[0], -1).T
        Q, _ = torch.linalg.qr(low_rank_jac_v_mat)
        Q = Q.to(self.device)
        Q = Q[:, :self.low_rank_rank_dim]

        # Stage 2: Direct SVD
        # (see Algo. 5.1.; https://epubs.siam.org/doi/epdf/10.1137/090771806)

        # Construction of a standard factorization using the information contained in the basis Q.
        num_batches_backward = ceil(self.low_rank_rank_dim / batch_size)
        qT_low_rank_jac_mat = []
        for i in tqdm(range(num_batches_backward), miniters=num_batches_backward//100,
                desc='get_low_rank_jac_basis backward'):
            qT_i = Q[:, i * batch_size:(i * batch_size) + batch_size].T
            qT_low_rank_jac_mat_row = self.neural_basis_expansion.vjp(
                    qT_i.view(qT_i.shape[0], *self.nn_out_shape).to(self.device))
            qT_low_rank_jac_mat_row.detach()
            qT_low_rank_jac_mat.append(
                    qT_low_rank_jac_mat_row.cpu() if use_cpu else qT_low_rank_jac_mat_row)
        B = torch.cat(qT_low_rank_jac_mat)  # (self.low_rank_rank_dim, self.num_params)

        U, S, Vh = torch.linalg.svd(B, full_matrices=False)

        return (Q @ U.to(self.device), S.to(self.device), Vh.to(self.device))

    def jvp(self, v) -> Tensor:

        return (self.jac_U @ (self.jac_S[:, None] * (self.jac_Vh @ v.T))).T.view(
                        -1, *self.nn_out_shape)

    def vjp(self, v) -> Tensor:

        return ((v.view(v.shape[0], -1) @ self.jac_U) * self.jac_S[None, :]) @ self.jac_Vh
