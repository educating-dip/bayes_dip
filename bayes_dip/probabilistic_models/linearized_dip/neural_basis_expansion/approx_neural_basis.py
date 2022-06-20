import torch
import torch.nn as nn
from tqdm import tqdm
from math import ceil
from typing import List, Tuple
from torch import Tensor
from .neural_basis_expansion import NeuralBasisExpansion

class ApproxNeuralBasisExpansion(NeuralBasisExpansion):

    def __init__(self,
            model: nn.Module,
            nn_input: Tensor,
            include_biases: bool,
            vec_batch_size: int,
            oversampling_param: int,
            low_rank_rank_dim: int,
            device: None,
            exclude_nn_layers: List = [],
            load_approx_basis_from: str = None,
            return_on_cpu: bool = False,
            use_cpu: bool = False) -> None:
            
        super().__init__(model=model,
                    nn_input=nn_input,
                    include_biases=include_biases,
                    exclude_nn_layers=exclude_nn_layers
            )

        self.vec_batch_size = vec_batch_size
        self.oversampling_param = oversampling_param
        self.low_rank_rank_dim = low_rank_rank_dim
        self.device = device 
        self.return_on_cpu = return_on_cpu
        self.use_cpu = use_cpu
        
        if not load_approx_basis_from:
            self.jac_U, self.jac_S, self.jac_Vh = self.get_batched_jac_low_rank()
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
        low_rank_rank_dim = self.low_rank_rank_dim + self.oversampling_param
        random_matrix = torch.randn(
            (self.num_params,
                    low_rank_rank_dim, 
                ),
            device=self.device
            )
        return random_matrix

    def get_batched_jac_low_rank(self, ) -> Tuple[Tensor, Tensor, Tensor]:

        random_matrix = self._assemble_random_matrix()
        num_batches = ceil((self.low_rank_rank_dim + self.oversampling_param) / self.vec_batch_size)
        low_rank_jac_v_mat = []
        for i in tqdm(range(num_batches), miniters=num_batches//100, desc='get_batched_jac_low_rank forward'):
            rnd_vect = random_matrix[:, i * self.vec_batch_size:(i * self.vec_batch_size) + self.vec_batch_size]
            _, low_rank_jac_v_mat_row = self.jvp(rnd_vect.T)
            low_rank_jac_v_mat_row.detach()
            low_rank_jac_v_mat.append( low_rank_jac_v_mat_row.cpu() if self.use_cpu else low_rank_jac_v_mat_row )
        low_rank_jac_v_mat = torch.cat(low_rank_jac_v_mat)
        low_rank_jac_v_mat = low_rank_jac_v_mat.view(*low_rank_jac_v_mat.shape[:1], -1).T
        Q, _ = torch.linalg.qr(low_rank_jac_v_mat)
        if not self.return_on_cpu:
            Q = Q.to(self.device)
        qT_low_rank_jac_mat = []
    
        assert self.low_rank_rank_dim + self.oversampling_param <= low_rank_jac_v_mat.shape[0], 'low rank dim must not be larger than network output dimension'

        for i in tqdm(range(num_batches), miniters=num_batches//100, desc='get_batched_jac_low_rank backward'):
            qT_i = Q[:, i * self.vec_batch_size:(i * self.vec_batch_size) + self.vec_batch_size].T
            qT_low_rank_jac_mat_row = self.vjp(qT_i.to(self.device))
            qT_low_rank_jac_mat_row.detach()
            qT_low_rank_jac_mat.append( qT_low_rank_jac_mat_row.cpu() if self.use_cpu else qT_low_rank_jac_mat_row )
        B = torch.cat(qT_low_rank_jac_mat)

        U, S, Vh = torch.linalg.svd(B, full_matrices=False)
        if not self.return_on_cpu:
            U = U.to(self.device)
            S = S.to(self.device)
            Vh = Vh.to(self.device)
        return  Q[:, :self.low_rank_rank_dim] @ U[:self.low_rank_rank_dim, :self.low_rank_rank_dim], S[:self.low_rank_rank_dim], Vh[:self.low_rank_rank_dim, :]
    
    def vjp_approx(self, v) -> Tensor:
        return (( v @ self.jac_U) * self.jac_S[None, :]) @ self.jac_Vh 

    def jvp_approx(self, v) -> Tensor:
        return (self.jac_U @ (self.jac_S[:, None] * (self.jac_Vh @ v.T))).T 