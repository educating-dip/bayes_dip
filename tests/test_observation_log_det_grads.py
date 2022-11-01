import pytest
import torch
import functorch as ftch
from bayes_dip.data import get_ray_trafo, get_kmnist_testset, SimulatedDataset
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts, ParameterCov, NeuralBasisExpansion, ImageCov, ObservationCov, LowRankObservationCov
from bayes_dip.marginal_likelihood_optim.observation_cov_log_det_grad import approx_observation_cov_log_det_grads
from bayes_dip.probabilistic_models.linearized_dip.utils import get_inds_from_ordered_params
from bayes_dip.marginal_likelihood_optim import LowRankObservationCovPreconditioner

@pytest.fixture(scope='session')
def observation_cov():
    dtype = torch.float32
    device = 'cpu'
    kwargs = {
        'angular_sub_sampling': 1, 'im_shape': (28, 28), 'num_angles': 10, 'impl': 'astra_cpu'}
    ray_trafo = get_ray_trafo('kmnist', kwargs=kwargs)
    ray_trafo.to(dtype=dtype, device=device)
    image_dataset = get_kmnist_testset()
    dataset = SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=0.05,
            use_fixed_seeds_starting_from=1,
            device=device)
    observation, ground_truth, filtbackproj = dataset[0]
    filtbackproj = filtbackproj[None]  # add batch dim
    net_kwargs = {
            'scales': 3,
            'channels': [2, 2, 2],
            'skip_channels': [0, 0, 1],  # skip_channels[-2] seems to make estimation more
                                         # difficult (e.g. requiring more probes), so for shorter
                                         # test run-times we exclude it here
            'use_norm': False,
            'use_sigmoid': True,
            'sigmoid_saturation_thresh': 15
        }
    reconstructor = DeepImagePriorReconstructor(
            ray_trafo, torch_manual_seed=1,
            device=device, net_kwargs=net_kwargs)
    optim_kwargs = {
            'lr': 1e-2,
            'iterations': 100,
            'loss_function': 'mse',
            'gamma': 1e-4
        }
    _ = reconstructor.reconstruct(
        observation,
        filtbackproj=filtbackproj,
        ground_truth=ground_truth,
        recon_from_randn=False,
        log_path='./',
        optim_kwargs=optim_kwargs)
    prior_assignment_dict, hyperparams_init_dict = get_default_unet_gaussian_prior_dicts(
            reconstructor.nn_model)
    parameter_cov = ParameterCov(reconstructor.nn_model, prior_assignment_dict, hyperparams_init_dict, device=device)
    neural_basis_expansion = NeuralBasisExpansion(
            nn_model=reconstructor.nn_model,
            nn_input=filtbackproj,
            ordered_nn_params=parameter_cov.ordered_nn_params,
            nn_out_shape=filtbackproj.shape,
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
    return observation_cov

@pytest.fixture(scope='session')
def exact_grads(observation_cov):
    nn_model = observation_cov.image_cov.neural_basis_expansion.nn_model
    nn_input = observation_cov.image_cov.neural_basis_expansion.nn_input
    ordered_nn_params = observation_cov.image_cov.inner_cov.ordered_nn_params
    func, _ = ftch.make_functional(nn_model)
    inds_from_ordered_params = get_inds_from_ordered_params(nn_model, ordered_nn_params)
    def _func_model(*func_params_under_prior):
        func_params = [orig_param for orig_param in nn_model.parameters()]
        for i, func_param in zip(inds_from_ordered_params, func_params_under_prior):
            func_params[i] = func_param
        return func(func_params, nn_input)
    jac = torch.autograd.functional.jacobian(_func_model, tuple(ordered_nn_params))
    jac = torch.cat([jac_i.view(nn_input.numel(), -1) for jac_i in jac], dim=1)
    trafo_mat = observation_cov.trafo.matrix
    observation_cov_mat = (
            observation_cov.image_cov.inner_cov(trafo_mat @ jac) @ jac.T @ trafo_mat.T +
            observation_cov.log_noise_variance.exp() * torch.eye(observation_cov.shape[0]))
    sign, log_det = torch.linalg.slogdet(observation_cov_mat)
    assert sign > 0.
    exact_grads_tuple = torch.autograd.grad((0.5 * log_det,), observation_cov.parameters())
    return exact_grads_tuple

def test_approx_observation_log_det_grads(observation_cov, exact_grads):
    torch.manual_seed(1)
    grads, _ = approx_observation_cov_log_det_grads(
            observation_cov=observation_cov,
            precon=None,
            max_cg_iter=100,
            cg_rtol=1e-6,
            num_probes=200,
            )

    for (name, p), exact_grad in zip(observation_cov.named_parameters(), exact_grads):
        print(name, grads[p], exact_grad)
        assert torch.allclose(grads[p], exact_grad, rtol=1., atol=1e-2)

def test_approx_observation_log_det_grads_with_preconditioner(observation_cov, exact_grads):
    torch.manual_seed(1)
    low_rank_observation_cov = LowRankObservationCov(
            trafo=observation_cov.trafo,
            image_cov=observation_cov.image_cov,
            low_rank_rank_dim=100,
            oversampling_param=10,
            requires_grad=False,
            device=observation_cov.device
    )
    low_rank_preconditioner =  LowRankObservationCovPreconditioner(
            low_rank_observation_cov=low_rank_observation_cov
    )
    grads, _ = approx_observation_cov_log_det_grads(
            observation_cov=observation_cov,
            precon=low_rank_preconditioner,
            max_cg_iter=20,
            cg_rtol=1e-6,
            num_probes=100,
            )

    for (name, p), exact_grad in zip(observation_cov.named_parameters(), exact_grads):
        print(name, grads[p], exact_grad)
        assert torch.allclose(grads[p], exact_grad, rtol=1., atol=1e-2)
