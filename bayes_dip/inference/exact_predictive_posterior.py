import torch
import numpy as np
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from bayes_dip.probabilistic_models import MatmulObservationCov
from .base_predictive_posterior import BasePredictivePosterior

def assert_positive_diag(mat):
    assert mat.diag().min() > 0

def make_choleskable(
        mat: Tensor, step: float = 1e-6, max_nsteps: int = 1000, verbose: bool = True) -> Tensor:

    succeed = False
    cnt = 0
    while not succeed:
        try:
            chol = torch.linalg.cholesky(mat)
            succeed = True
        except RuntimeError:
            mat[np.diag_indices(mat.shape[0])] += step
            cnt += 1
            assert cnt < max_nsteps
    if verbose:
        print(f'amount added to make choleskable: {cnt*step}')
    return chol

class ExactPredictivePosterior(BasePredictivePosterior):


    def __init__(self,
            observation_cov: MatmulObservationCov):
        super().__init__(observation_cov=observation_cov)

    def covariance(self,
        noise_x_correction_term: float = 1e-6,
        eps: float = 1e-6
        ) -> Tensor:

        obs_cov_mat = self.observation_cov.matrix  # obs_cov

        # jac, shape (dx, dparam)
        jac_mat = self.observation_cov.image_cov.neural_basis_expansion.matrix

        # jac @ param_cov, shape (dx, dparam), batched over dx
        jac_param_cov_mat = self.observation_cov.image_cov.inner_cov(jac_mat)

        # jac @ param_cov @ jac.T
        image_cov_mat = jac_param_cov_mat @ jac_mat.T
        image_cov_mat[np.diag_indices(image_cov_mat.shape[0])] += eps
        assert_positive_diag(image_cov_mat)

        # ray_trafo @ jac
        trafo_jac_mat = self.observation_cov.trafo.matrix @ jac_mat

        # jac @ param_cov @ jac.T @ ray_trafo.T
        cov_image_obs_mat = jac_param_cov_mat @ trafo_jac_mat.T
        # ray_trafo @ jac @ param_cov @ jac.T
        cov_obs_image_mat = cov_image_obs_mat.T

        # jac @ param_cov @ jac.T -
        #   jac @ param_cov @ jac.T @ ray_trafo.T @ obs_cov^-1 @ ray_trafo @ jac @ param_cov @ jac.T
        pred_cov_mat = image_cov_mat - cov_image_obs_mat @ torch.linalg.solve(
                obs_cov_mat, cov_obs_image_mat)
        pred_cov_mat[np.diag_indices(pred_cov_mat.shape[0])] += noise_x_correction_term
        assert_positive_diag(pred_cov_mat)

        return pred_cov_mat

    def multivariate_normal_distribution(self,
            mean: Tensor,
            **kwargs
            ) -> MultivariateNormal:

        pred_cov_mat = self.covariance(**kwargs)

        pred_cov_mat_chol = make_choleskable(pred_cov_mat)
        dist = MultivariateNormal(
                loc=mean.flatten(),
                scale_tril=pred_cov_mat_chol
            )

        return dist

    def log_prob(self,
            mean: Tensor,
            ground_truth: Tensor,
            **kwargs,
            ) -> float:
        # pylint: disable=arguments-differ

        assert ground_truth.shape == mean.shape
        assert ground_truth.shape[:2] == (1, 1)

        dist = self.multivariate_normal_distribution(mean=mean, **kwargs)

        log_prob_unscaled = dist.log_prob(ground_truth.flatten()).item()
        return log_prob_unscaled / np.prod(ground_truth.shape)

    def sample(self,
            num_samples: int,
            mean: Tensor,
            **kwargs
            ) -> Tensor:
        # pylint: disable=arguments-differ

        assert mean.shape[:2] == (1, 1)

        dist = self.multivariate_normal_distribution(mean=mean, **kwargs)

        return dist.rsample((num_samples,)).view(num_samples, *mean.shape[1:])
