import os
import torch
import numpy as np
from torch import Tensor
from bayes_dip.data import get_ray_trafo, get_kmnist_testset, SimulatedDataset
from bayes_dip.probabilistic_models import get_default_unet_gprior_dicts
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import (NeuralBasisExpansion, MatmulNeuralBasisExpansion, LowRankObservationCov, ParameterCov, ImageCov, ObservationCov,
    MatmulObservationCov, GpriorNeuralBasisExpansion, MatmulGpriorNeuralBasisExpansion)
from bayes_dip.marginal_likelihood_optim import LowRankObservationCovPreconditioner
from bayes_dip.inference import SampleBasedPredictivePosterior, ExactPredictivePosterior

def test_predictive_posterior_cov():
    dtype = torch.float32
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    kwargs = {'angular_sub_sampling': 1, 'im_shape': (28, 28), 'num_angles': 20}
    ray_trafo = get_ray_trafo('kmnist', kwargs=kwargs)
    ray_trafo.to(dtype=dtype, device=device)
    image_dataset = get_kmnist_testset()
    dataset = SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=0.05,
            use_fixed_seeds_starting_from=1,
            device=device)
    observation, ground_truth, filtbackproj = dataset[0]
    observation = observation[None].to(dtype=dtype, device=device)
    filtbackproj = filtbackproj[None].to(dtype=dtype, device=device)
    ground_truth = ground_truth[None].to(dtype=dtype, device=device)

    net_kwargs = {
                'scales': 3,
                'channels': [32, 32, 32],
                'skip_channels': [0, 4, 4],
                'use_norm': False,
                'use_sigmoid': True,
                'sigmoid_saturation_thresh': 15}
    reconstructor = DeepImagePriorReconstructor(
            ray_trafo, torch_manual_seed=1,
            device=device, net_kwargs=net_kwargs)
    optim_kwargs = {
                    'lr': 1e-4,
                    'iterations': 1000,
                    'loss_function': 'mse',
                    'gamma': 1e-4
                }
    _ = reconstructor.reconstruct(
                    observation,
                    filtbackproj=filtbackproj,
                    ground_truth=ground_truth,
                    recon_from_randn=False,
                    log_path=os.path.join('./', f'dip_optim_{0}'),
                    optim_kwargs=optim_kwargs)
    
    prior_assignment_dict, hyperparams_init_dict = get_default_unet_gprior_dicts(
            reconstructor.nn_model)
    parameter_cov = ParameterCov(
        reconstructor.nn_model, 
        prior_assignment_dict, 
        hyperparams_init_dict, 
        device=device
    )
    # parameter_cov.priors.gprior.log_variance.data = torch.tensor(np.log(0.5))
    scale_kwargs = {
        'reduction': 'sum',
        'batch_size': 10,
        'eps': 1e-6,
        'max_scale_thresh': 1e5,
        'verbose': True,
    }

    neural_basis_expansion = NeuralBasisExpansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
        )
    
    neural_basis_expansion = GpriorNeuralBasisExpansion(
                neural_basis_expansion=neural_basis_expansion,
                trafo=ray_trafo,
                scale_kwargs=scale_kwargs,
                device=device
        )
    image_cov = ImageCov(
                parameter_cov=parameter_cov,
                neural_basis_expansion=neural_basis_expansion
        )
    observation_cov = ObservationCov(
                trafo=ray_trafo,
                image_cov=image_cov,
                device=device
        )
    observation_cov.log_noise_variance.data = torch.tensor(np.log(0.05))

    predictive_posterior = SampleBasedPredictivePosterior(observation_cov)

    sample_kwargs= {
        'batch_size': 32,
        'use_conj_grad_inv': True,
        'cg_kwargs': {
            'rtol': 1e-3,
            'max_niter': 300,
            'ignore_numerical_warning': False
        }
    }
    
    low_rank_observation_cov = LowRankObservationCov(
                        trafo=ray_trafo,
                        image_cov=image_cov,
                        low_rank_rank_dim=400,
                        oversampling_param=5,
                        batch_size=32,
                        device=device
        )
    low_rank_preconditioner = LowRankObservationCovPreconditioner(
                        low_rank_observation_cov=low_rank_observation_cov
        )
    sample_kwargs['cg_kwargs']['precon_closure'] = (
                        lambda v: low_rank_preconditioner.matmul(v, use_inverse=True))
    samples, _ = predictive_posterior.sample_zero_mean(
                    num_samples=1000,
                    cov_obs_mat_chol=None,
                    return_residual_norm_list=True,
                    **sample_kwargs
                )
    samples = ray_trafo(samples)

    matmul_neural_basis_expansion = MatmulNeuralBasisExpansion(
                nn_model=reconstructor.nn_model,
                nn_input=filtbackproj,
                ordered_nn_params=parameter_cov.ordered_nn_params,
                nn_out_shape=filtbackproj.shape,
        )
    
    matmul_observation_cov = MatmulObservationCov(
                trafo=ray_trafo,
                image_cov= ImageCov(
                    parameter_cov=parameter_cov,
                    neural_basis_expansion =
                            MatmulGpriorNeuralBasisExpansion(
                        neural_basis_expansion=matmul_neural_basis_expansion,
                        trafo=ray_trafo,
                        scale_kwargs=scale_kwargs
                    )
                ),
                device=device
        )
    
    matmul_observation_cov.log_noise_variance.data = torch.tensor(np.log(0.05))
    exact_predictive_posterior = ExactPredictivePosterior(matmul_observation_cov)
    exact_samples = ray_trafo(
            exact_predictive_posterior.sample(
                num_samples=1000, mean=torch.zeros_like(ground_truth)
            )
        )

    def effective_dims(
        y_post_samples: Tensor
        ) -> float:

        return y_post_samples.pow(2).mean(dim=0).sum()

    print(effective_dims(exact_samples))
    print(effective_dims(samples))

if __name__ == '__main__':
    test_predictive_posterior_cov()
