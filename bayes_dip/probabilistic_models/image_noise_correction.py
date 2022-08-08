import numpy as np
import scipy.sparse
import torch
import tensorly as tl
from bayes_dip.data.trafo.matmul_ray_trafo import MatmulRayTrafo, _convert_to_scipy_sparse_matrix
from bayes_dip.probabilistic_models.observation_cov import ObservationCov


def get_trafo_t_trafo_pseudo_inv_diag_mean(trafo: MatmulRayTrafo) -> float:

    trafo_mat = trafo.matrix
    if trafo_mat.is_sparse:
        # tl.truncated_svd does not support sparse tensors;
        # torch has a function svd_lowrank that is much faster,
        # but the result seems to differ from scipy.sparse.linalg.svds, so use scipy
        trafo_mat = _convert_to_scipy_sparse_matrix(trafo_mat)
        U_trafo, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo_mat, k=100)
        # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
        S_inv_Vh_trafo = 1./S_trafo[:, None] * Vh_trafo
        # trafo_T_trafo_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
        trafo_T_trafo_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
        diag_mean = np.mean(trafo_T_trafo_diag)
    else:
        # pseudo-inverse computation
        trafo_T_trafo = trafo_mat.T @ trafo_mat
        U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100)
        # diag_mean = ((U * (1./S)[None, :]) @ Vh).diag().mean().item()
        trafo_T_trafo_diag = torch.sum((U * (1./S)[None, :]) * Vh.T, axis=1)
        diag_mean = torch.mean(trafo_T_trafo_diag).item()
    return diag_mean


def get_image_noise_correction_term(observation_cov: ObservationCov) -> float:
    diag_mean = get_trafo_t_trafo_pseudo_inv_diag_mean(observation_cov.trafo)
    image_noise_correction_term = diag_mean * observation_cov.log_noise_variance.exp().item()
    return image_noise_correction_term
