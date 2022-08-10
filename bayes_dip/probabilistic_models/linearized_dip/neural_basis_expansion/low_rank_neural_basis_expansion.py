from typing import Optional, Tuple
from math import ceil
import torch
from torch import Tensor
from tqdm import tqdm
from .base_neural_basis_expansion import BaseNeuralBasisExpansion
from .neural_basis_expansion import NeuralBasisExpansion

class LowRankNeuralBasisExpansion(BaseNeuralBasisExpansion):

    """
    Wrapper class for Jacobian vector products and vector Jacobian products using
    low-rank Jacobian matrix approximation. This class extracts approximate
    Jacobian bases based on Halko et al. ``Finding Structure with Randomness:
    probabilistic algorithms for constructing approximate matrix decompositions``
    (https://epubs.siam.org/doi/epdf/10.1137/090771806).
    This class (``:meth:get_batched_low_rank_jac``) uses randomization
    (``:meth:_assemble_random_matrix``) to perform low-rank matrix approximation.
    """

    def __init__(self,
            neural_basis_expansion: NeuralBasisExpansion,
            oversampling_param: int,
            low_rank_rank_dim: int,
            load_approx_basis_from: Optional[str] = None,
            device=None,
            vec_batch_size: int = 1,
            use_cpu: bool = False) -> None:


        """
        Parameters are the same as for
        :class:`bayes_dip.probabilistic_models.BaseNeuralBasisExpansion`.

        Except for,

        neural_basis_expansion :class:`bayes_dip.probabilistic_models.NeuralBasisExpansion`
            Wrapper class for Jacobian vector products and vector Jacobian products.
        oversampling_param : int
            Oversampling parameter.
        low_rank_rank_dim : int
            Low rank dimension. Extracting leading singular vectors.
        vec_batch_size : int
            Batch size. The default is `1`.
        use_cpu : bool, optional
            Whether to perform SVD on CPU. The default is False.
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

        if not load_approx_basis_from:
            self.jac_U, self.jac_S, self.jac_Vh = self.get_batched_low_rank_jac(
                    vec_batch_size=vec_batch_size, use_cpu=use_cpu)
        else:
            #TODO: load U, S and Vh
            self.load_approx_basis(load_approx_basis_from)
            raise NotImplementedError

    def load_approx_basis(self, path):
        pass

    def save_approx_basis(self, ):
        #TODO: save basis save_approx_basis
        pass

    def _assemble_random_matrix(self, ) -> Tensor:
        total_low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param
        random_matrix = torch.randn(
            (self.num_params, total_low_rank_rank_dim),
            device=self.device
            ) # constructing Gaussian random matrix Omega
        return random_matrix

    def get_batched_low_rank_jac(self,
            vec_batch_size: int = 1, use_cpu: bool = False) -> Tuple[Tensor, Tensor, Tensor]:

        random_matrix = self._assemble_random_matrix() # draw a Gaussian random matrix Omega

        total_low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param

        # Stage 1: Randomized Range Finder
        # (see Algo. 4.1; https://epubs.siam.org/doi/epdf/10.1137/090771806)

        # Identifying a subspace that captures most of the action of the Jacobian. Constructing a
        # matrix Q whose columns form an orthonormal basis for the range of Y = Jac @ Omega.
        num_batches = ceil(total_low_rank_rank_dim / vec_batch_size)
        low_rank_jac_v_mat = []
        for i in tqdm(range(num_batches), miniters=num_batches//100,
                desc='get_batched_jac_low_rank forward'):
            rnd_vect = random_matrix[
                    :, i * vec_batch_size:(i * vec_batch_size) + vec_batch_size]
            low_rank_jac_v_mat_row = self.neural_basis_expansion.jvp(rnd_vect.T)
            low_rank_jac_v_mat.append(
                    low_rank_jac_v_mat_row.cpu() if use_cpu else low_rank_jac_v_mat_row)
        low_rank_jac_v_mat = torch.cat(low_rank_jac_v_mat)
        # Y = Jac @ Omega, ( np.prod(self.nn_out_shape[2:]) , total_low_rank_rank_dim )
        low_rank_jac_v_mat = low_rank_jac_v_mat.view(low_rank_jac_v_mat.shape[0], -1).T
        Q, _ = torch.linalg.qr(low_rank_jac_v_mat)
        Q = Q.to(self.device)

        assert total_low_rank_rank_dim <= low_rank_jac_v_mat.shape[0], (
                'low rank dim must not be larger than network output dimension')

        # Stage 2: Direct SVD: (see Algo. 5.1.; https://epubs.siam.org/doi/epdf/10.1137/090771806)

        # Construction of a standard factorization using the information contained in the basis Q.

        qT_low_rank_jac_mat = []
        for i in tqdm(range(num_batches), miniters=num_batches//100,
                desc='get_batched_jac_low_rank backward'):
            qT_i = Q[:, i * vec_batch_size:(i * vec_batch_size) + vec_batch_size].T
            qT_low_rank_jac_mat_row = self.neural_basis_expansion.vjp(
                    qT_i.view(qT_i.shape[0], *self.nn_out_shape).to(self.device))
            qT_low_rank_jac_mat_row.detach()
            qT_low_rank_jac_mat.append(
                    qT_low_rank_jac_mat_row.cpu() if use_cpu else qT_low_rank_jac_mat_row)
        B = torch.cat(qT_low_rank_jac_mat) # ( (total_low_rank_rank_dim), self.num_params)

        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = U.to(self.device)
        S = S.to(self.device)
        Vh = Vh.to(self.device)

        return (
                Q[:, :self.low_rank_rank_dim] @ U[:self.low_rank_rank_dim, :self.low_rank_rank_dim],
                S[:self.low_rank_rank_dim],
                Vh[:self.low_rank_rank_dim, :])

    def jvp(self, v) -> Tensor:

        """
        v : Tensor
            Input. Shape: ``(batch_size, self.num_params)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, *self.nn_out_shape)``
        """
        return (self.jac_U @ (self.jac_S[:, None] * (self.jac_Vh @ v.T))).T.view(
                        -1, *self.nn_out_shape)

    def vjp(self, v) -> Tensor:
        """
        Parameters
        ----------
        v : Tensor
            Input. Shape: ``(batch_size, *self.nn_out_shape)``

        Returns
        -------
        Tensor
            Output. Shape: ``(batch_size, self.num_params)``
        """
        return ((v.view(v.shape[0], -1) @ self.jac_U) * self.jac_S[None, :]) @ self.jac_Vh
