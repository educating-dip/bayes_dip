import pytest
import torch
import functorch as ftch
from bayes_dip.data import get_ray_trafo, get_kmnist_testset, SimulatedDataset
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts, ParameterCov, NeuralBasisExpansion, ImageCov, ObservationCov
from bayes_dip.marginal_likelihood_optim.observation_cov_log_det_grad import approx_observation_cov_log_det_grads
from bayes_dip.probabilistic_models.linearized_dip.utils import get_inds_from_ordered_params

@pytest.fixture(scope='function')
def observation_cov():
    dtype = torch.float32
    device = 'cpu'
    kwargs = {'angular_sub_sampling': 1, 'im_shape': (28, 28), 'num_angles': 20}
    ray_trafo = get_ray_trafo('kmnist', kwargs=kwargs)
    ray_trafo.to(dtype=dtype, device=device)
    image_dataset = get_kmnist_testset()
    dataset = SimulatedDataset(
            image_dataset, ray_trafo,
            white_noise_rel_stddev=0.05,
            use_fixed_seeds_starting_from=1,
            device=device)
    _, _, filtbackproj = dataset[0]
    filtbackproj = filtbackproj[None]  # add batch dim
    net_kwargs = {
                'scales': 3,
                'channels': [8, 8, 8],
                'skip_channels': [0, 1, 1],
                'use_norm': False,
                'use_sigmoid': True,
                'sigmoid_saturation_thresh': 15}
    reconstructor = DeepImagePriorReconstructor(
            ray_trafo, torch_manual_seed=1,
            device=device, net_kwargs=net_kwargs)
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

def test_approx_observation_log_det_grads(observation_cov):
    torch.manual_seed(1)
    grads, _ = approx_observation_cov_log_det_grads(
            observation_cov=observation_cov,
            preconditioner=None,
            max_cg_iter=10,
            cg_rtol=1e-9,
            num_probes=1000,
            )
    # observation_cov_mat = observation_cov.assemble_observation_cov()
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
    exact_grads = torch.autograd.grad((0.5 * log_det,), observation_cov.parameters())
    for p, exact_grad in zip(observation_cov.parameters(), exact_grads):
        print(grads[p], exact_grad)
        assert torch.allclose(grads[p], exact_grad, rtol=1.)
