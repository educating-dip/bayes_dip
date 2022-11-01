import pytest
import torch
from bayes_dip.data import get_ray_trafo, get_kmnist_testset, SimulatedDataset
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts, ParameterCov

@pytest.fixture(scope='session')
def parameter_cov():
    dtype = torch.float32
    device = 'cpu'
    kwargs = {
        'angular_sub_sampling': 1, 'im_shape': (28, 28), 'num_angles': 20, 'impl': 'astra_cpu'}
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

    return parameter_cov

def test_parameter_cov_fw(parameter_cov):

    torch.manual_seed(1)
    identity = torch.eye(parameter_cov.shape[0])

    parameter_cov_mat_assembled = []
    for _, priors in parameter_cov.priors_per_prior_type.items():
        for prior in priors:
            for _ in range(prior.num_total_filters):
                parameter_cov_mat_assembled.append(prior.cov_mat())

    parameter_cov_mat_assembled = torch.block_diag(*parameter_cov_mat_assembled)
    parameter_cov_mat = parameter_cov(identity)
    assert torch.allclose(parameter_cov_mat, parameter_cov_mat_assembled)

def test_parameter_cov_fw_inv(parameter_cov):

    torch.manual_seed(1)
    identity = torch.eye(parameter_cov.shape[0])
    parameter_cov_mat = parameter_cov(identity)
    parameter_cov_mat_inv = torch.linalg.inv(parameter_cov_mat)
    parameter_cov_mat_inv_alt = parameter_cov(identity, use_inverse=True)
    assert torch.allclose(parameter_cov_mat_inv, parameter_cov_mat_inv_alt)