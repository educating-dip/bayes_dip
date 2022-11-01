import pytest
import torch
from bayes_dip.data import get_ray_trafo, get_kmnist_testset, SimulatedDataset
from bayes_dip.dip import DeepImagePriorReconstructor
from bayes_dip.probabilistic_models import get_default_unet_gaussian_prior_dicts, ParameterCov, NeuralBasisExpansion, MatmulNeuralBasisExpansion, ImageCov, MatmulObservationCov, ObservationCov

@pytest.fixture(scope='session')
def observation_cov_and_matmul_observation_cov():
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
    parameter_cov = ParameterCov(
            reconstructor.nn_model,
            prior_assignment_dict,
            hyperparams_init_dict,
            device=device
    )

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

    matmul_neural_basis_expansion = MatmulNeuralBasisExpansion(
            nn_model=reconstructor.nn_model,
            nn_input=filtbackproj,
            ordered_nn_params=parameter_cov.ordered_nn_params,
            nn_out_shape=filtbackproj.shape,
    )
    matmul_image_cov = ImageCov(
            parameter_cov=parameter_cov,
            neural_basis_expansion=matmul_neural_basis_expansion
    )
    matmul_observation_cov = MatmulObservationCov(
            trafo=ray_trafo,
            image_cov=matmul_image_cov,
            device=device
    )

    return observation_cov, matmul_observation_cov

def test_observation_cov_vs_matmul_observation_cov(observation_cov_and_matmul_observation_cov):

    observation_cov, matmul_observation_cov = observation_cov_and_matmul_observation_cov

    observation_cov_assembled = observation_cov.assemble_observation_cov()
    matmul_observation_cov_assembled = matmul_observation_cov.get_matrix(
            apply_make_choleskable=True)

    assert torch.allclose(observation_cov_assembled, matmul_observation_cov_assembled)
