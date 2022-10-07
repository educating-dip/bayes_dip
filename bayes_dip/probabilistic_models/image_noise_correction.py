"""
Provides :func:`get_image_noise_correction_term`.
"""
import numpy as np
import scipy.sparse
import torch
from bayes_dip.data.trafo.matmul_ray_trafo import MatmulRayTrafo, _convert_to_scipy_sparse_matrix
from bayes_dip.probabilistic_models.observation_cov import ObservationCov


def get_trafo_t_trafo_pseudo_inv_diag_mean(
        trafo: MatmulRayTrafo, n_eigenvecs: int = 100) -> float:
    """
    Compute ``diag(mean(pinv(ray_trafo.T @ ray_trafo)))`` with the pseudo-inverse being approximated
    with a truncated SVD.

    Parameters
    ----------
    trafo : MatmulRayTrafo
        Ray transform.
    n_eigenvecs : int, optional
        Number of eigenvectors in the truncated SVD.

    Returns
    -------
    diag_mean : float
        Mean of the diagonal of the pseudo-inverse of ``ray_trafo.T @ ray_trafo``.
    """

    trafo_mat = trafo.matrix
    if trafo_mat.is_sparse:
        # tl.truncated_svd does not support sparse tensors;
        # torch has a function svd_lowrank that is much faster,
        # but the result seems to differ from scipy.sparse.linalg.svds, so use scipy
        trafo_mat = _convert_to_scipy_sparse_matrix(trafo_mat)
        _, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo_mat, k=n_eigenvecs)
        # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
        S_inv_Vh_trafo = 1./S_trafo[:, None] * Vh_trafo
        # trafo_T_trafo_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
        trafo_T_trafo_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
        diag_mean = np.mean(trafo_T_trafo_diag)
    else:
        # pseudo-inverse computation
        trafo_T_trafo = trafo_mat.T @ trafo_mat
        U, S, Vh = torch.linalg.svd(trafo_T_trafo)
        U, S, Vh = U[:, :n_eigenvecs], S[:n_eigenvecs], Vh[:n_eigenvecs, :]
        # diag_mean = ((U * (1./S)[None, :]) @ Vh).diag().mean().item()
        trafo_T_trafo_diag = torch.sum((U * (1./S)[None, :]) * Vh.T, axis=1)
        diag_mean = torch.mean(trafo_T_trafo_diag).item()
    return diag_mean


def get_image_noise_correction_term(observation_cov: ObservationCov) -> float:
    """
    Return an image noise correction term computed as
    ``diag(mean(pinv(ray_trafo.T @ ray_trafo))) * noise_variance``.

    This can be interpreted as a projection of the ``noise_variance`` in observation space to image
    space.

    Parameters
    ----------
    observation_cov : ObservationCov
        Observation covariance module.

    Returns
    -------
    image_noise_correction_term : float
        Image noise correction term.
    """
    diag_mean = get_trafo_t_trafo_pseudo_inv_diag_mean(observation_cov.trafo)
    image_noise_correction_term = diag_mean * observation_cov.log_noise_variance.exp().item()
    return image_noise_correction_term
