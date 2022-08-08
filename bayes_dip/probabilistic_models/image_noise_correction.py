import torch
import tensorly as tl
from bayes_dip.data.trafo.matmul_ray_trafo import MatmulRayTrafo
from bayes_dip.probabilistic_models.observation_cov import ObservationCov


def get_trafo_t_trafo_pseudo_inv_diag_mean(trafo: MatmulRayTrafo) -> float:

    trafo_mat = trafo.matrix
    if trafo_mat.is_sparse:
        # pseudo-inverse computation
        U_trafo, S_trafo, V_trafo = torch.svd_lowrank(trafo_mat, q=100)
        # (V S U.T U S V.T)^-1 == (V S^2 V.T)^-1 == V S^-2 V.T
        V_S_inv_trafo = V_trafo * (1./S_trafo)[None, :]
        # trafo_T_trafo_diag = torch.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
        trafo_T_trafo_diag = torch.sum(V_S_inv_trafo**2, axis=1)
        diag_mean = torch.mean(trafo_T_trafo_diag).item()
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
