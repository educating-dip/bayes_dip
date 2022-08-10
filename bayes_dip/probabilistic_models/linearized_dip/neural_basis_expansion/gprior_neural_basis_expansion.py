from warnings import warn
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor

from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from .neural_basis_expansion import NeuralBasisExpansion


class GpriorNeuralBasisExpansion(NeuralBasisExpansion):
    def __init__(self,
            trafo: BaseRayTrafo,
            scale_kwargs,
            *args,
            **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)

        self.trafo = trafo
        self.scale = self.compute_scale(**scale_kwargs)

    def update_scale(self, **scale_kwargs):
        self.scale = self.compute_scale(**scale_kwargs)

    def compute_scale(self,
            reduction: str = 'mean',
            batch_size: int = 1,
            eps: float = 1e-6,
            max_scale_thresh: float = 1e5,
            verbose: bool = True,
            ) -> Tensor:

        vjp_no_scale = super().vjp

        def closure(v):
            return vjp_no_scale(self.trafo.trafo_adjoint(v).unsqueeze(dim=1)).pow(2)

        obs_numel = np.prod(self.trafo.obs_shape)
        v = torch.empty((batch_size, 1, *self.trafo.obs_shape), device=self.trafo.matrix.device)
        rows = torch.zeros((self.num_params), device=self.trafo.matrix.device)
        with torch.no_grad():
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
            if verbose:
                print(f'scale.min: {rows.min()}, scale.max: {rows.max()}, '
                    f'scale.num_vals_below_{eps}:{(rows < eps).sum()}\n'
                )
            if rows.max() > max_scale_thresh:
                warn('max scale values reached.')
            scale_vec = (rows.clamp_(min=eps) / obs_numel).pow(0.5) if reduction == 'mean' \
                    else rows.clamp_(min=eps).pow(0.5) # num_obs, num_params

        return scale_vec

    def jvp(self, v: Tensor) -> Tensor:
        return super().jvp(v * self.scale)

    def vjp(self, v: Tensor) -> Tensor:
        return super().vjp(v) * self.scale
