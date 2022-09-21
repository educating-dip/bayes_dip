"""
Provides neural basis expansion with a scaling in weight space, used for the isotropic g-prior.
"""
from typing import Dict, Optional
from abc import ABC, abstractmethod
from warnings import warn
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm

from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from bayes_dip.data.trafo.matmul_ray_trafo  import MatmulRayTrafo
from .neural_basis_expansion import BaseNeuralBasisExpansion
from .base_neural_basis_expansion import BaseMatmulNeuralBasisExpansion

def compute_scale(
        neural_basis_expansion: BaseNeuralBasisExpansion,
        trafo: BaseRayTrafo,
        reduction: str = 'mean',
        eps: float = 1e-6,
        max_scale_thresh: float = 1e5,
        verbose: bool = True,
        batch_size: Optional[int] = 1,
        use_single_batch: Optional[bool] = None,
        device=None,
        ) -> Tensor:
    """
    Compute a scaling vector for the weight space, which can help to improve the condition of the
    (surrogate) observation covariance matrix.

    Parameters
    ----------
    neural_basis_expansion : :class:`bayes_dip.probabilistic_models.NeuralBasisExpansion`
        Neural basis expansion instance to be wrapped.
    trafo : :class:`bayes_dip.data.BaseRayTrafo`
        Ray transform.
    reduction : {``'mean'``, ``'sum'``}, optional
        Reduction kind for the tensors accumulated over observation space.
        If ``mean`` (the default), values are divided by ``np.prod(trafo.obs_shape)``.
    eps : float, optional
        Minimum value for clamping before taking the inverse. The default is ``1e-6``.
    max_scale_thresh : float, optional
        Maximum value, if exceeded, a warning is raised. The default is ``1e5``.
    verbose : bool, optional
        Whether to print minimum and maximum values before applying the square-root, the clamping
        and the inversion. The default is ``True``.
    batch_size : int, optional
        Batch size for trafo adjoint and vjp evaluations. This is not used if a single matmul is
        employed, see `use_single_batch`.
    use_single_batch : bool or `None`, optional
        Whether to perform a single matmul instead of evaluating in batches via a closure.
        If `None`, a single matmul is used iff both `neural_basis_expansion` and `trafo` are
        matmul implementations.
        If `True`, a single matmul is used iff `isinstance(trafo, MatmulRayTrafo)`.
        If `False`, batched closure evaluations are used.
    device : str or torch.device, optional
        Device. If `None` (the default), `'cuda:0'` is chosen if available or `'cpu'` otherwise.

    Returns
    -------
    scale_vec : Tensor
        Scale vector. Shape: ``(neural_basis_expansion.num_params,)``.
    """

    device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    vjp_no_scale = neural_basis_expansion.vjp
    if use_single_batch is None:
        use_single_batch = isinstance(neural_basis_expansion, BaseMatmulNeuralBasisExpansion)

    obs_numel = np.prod(trafo.obs_shape)

    with torch.no_grad():
        if use_single_batch and isinstance(trafo, MatmulRayTrafo):

            rows = vjp_no_scale(trafo.matrix.view(obs_numel, 1, 1, *trafo.im_shape)
                    ).pow(2).sum(dim=0)

        else:

            def closure(v):
                return vjp_no_scale(trafo.trafo_adjoint(v).unsqueeze(dim=1)).pow(2)

            v = torch.empty((batch_size, 1, *trafo.obs_shape), device=device)
            rows = torch.zeros((neural_basis_expansion.num_params), device=device)
            for i in tqdm(np.array(range(0, obs_numel, batch_size)),
                        desc='compute_scale', miniters=obs_numel//batch_size//100
                    ):
                v[:] = 0.
                # set v.view(batch_size, -1) to be a subset of rows of torch.eye(obs_numel);
                # in last batch, it may contain some additional (zero) rows
                v.view(batch_size, -1)[:, i:i+batch_size].fill_diagonal_(1.)
                rows_batch = closure(
                    v,
                )
                rows_batch = rows_batch.view(batch_size, -1)
                if i+batch_size > obs_numel:  # last batch
                    rows_batch = rows_batch[:obs_numel%batch_size]
                rows += rows_batch.sum(dim=0)

        if reduction == 'mean':
            rows /= obs_numel
        elif reduction == 'sum':
            pass
        else:
            raise ValueError(f'unknown reduction kind {reduction}')

        if verbose:
            print(f'scale.min: {rows.min()}, scale.max: {rows.max()}')

        if rows.max() > max_scale_thresh:
            warn('max scale values reached.')

        scale_vec = (rows).pow(0.5).clamp(min=eps).pow(-1)  # num_params

    return scale_vec

class MixinGpriorNeuralBasisExpansion(ABC):
    @property
    @abstractmethod
    def scale(self):
        raise NotImplementedError

    @abstractmethod
    def update_scale(self):
        raise NotImplementedError

    @abstractmethod
    def compute_scale(self):
        raise NotImplementedError

class GpriorNeuralBasisExpansion(BaseNeuralBasisExpansion, MixinGpriorNeuralBasisExpansion):
    def __init__(self,
            neural_basis_expansion: BaseNeuralBasisExpansion,
            trafo: BaseRayTrafo,
            scale_kwargs: Dict,
            device=None,
        ) -> None:

        super().__init__(
                nn_model=neural_basis_expansion.nn_model, nn_input=neural_basis_expansion.nn_input,
                ordered_nn_params=neural_basis_expansion.ordered_nn_params,
                nn_out_shape=neural_basis_expansion.nn_out_shape)

        self.device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.neural_basis_expansion = neural_basis_expansion
        self.trafo = trafo
        self.update_scale(**scale_kwargs)

    @property
    def scale(self):
        return self._scale

    def update_scale(self, **scale_kwargs):
        self._scale = self.compute_scale(**scale_kwargs)

    def compute_scale(self, **scale_kwargs) -> Tensor:
        scale_vec = compute_scale(
                neural_basis_expansion=self.neural_basis_expansion, trafo=self.trafo,
                **scale_kwargs)
        return scale_vec

    def jvp(self, v: Tensor) -> Tensor:
        return self.neural_basis_expansion.jvp(v * self.scale)

    def vjp(self, v: Tensor) -> Tensor:
        return self.neural_basis_expansion.vjp(v) * self.scale

class MatmulGpriorNeuralBasisExpansion(
        BaseMatmulNeuralBasisExpansion, MixinGpriorNeuralBasisExpansion):

    def __init__(self,
            neural_basis_expansion: BaseMatmulNeuralBasisExpansion,
            trafo: BaseRayTrafo,
            scale_kwargs: Dict,
        ) -> None:

        super().__init__(
                nn_model=neural_basis_expansion.nn_model, nn_input=neural_basis_expansion.nn_input,
                ordered_nn_params=neural_basis_expansion.ordered_nn_params,
                nn_out_shape=neural_basis_expansion.nn_out_shape)

        self.neural_basis_expansion = neural_basis_expansion
        self.trafo = trafo
        self._matrix, self._scale = None, None
        self.update_matrix(**scale_kwargs)

    @property
    def matrix(self):
        return self._matrix

    @property
    def scale(self):
        return self._scale

    def compute_scale(self, **scale_kwargs) -> Tensor:
        scale_vec = compute_scale(
                neural_basis_expansion=self.neural_basis_expansion, trafo=self.trafo,
                **scale_kwargs)
        return scale_vec

    def get_matrix(self, return_scale=False, **scale_kwargs) -> Tensor:
        matrix_no_scale = self.neural_basis_expansion.matrix
        scale = self.compute_scale(**scale_kwargs)
        matrix = matrix_no_scale * scale
        return (matrix, scale) if return_scale else matrix

    def update(self, **scale_kwargs) -> None:
        self._matrix, self._scale = self.get_matrix(**scale_kwargs, return_scale=True)

    # scale contributes to self.matrix, so need to update both if updating one
    update_matrix = update
    update_scale = update
