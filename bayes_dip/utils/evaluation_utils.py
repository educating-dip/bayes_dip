"""
Utilities for evaluation.

A general note on hydra run paths: the methods in this module expect paths to single hydra output
folders (containing the ``.hydra/`` subfolder), which may either be a normal hydra output path or a
path to a sub-folder of a hydra multirun path (such sub-folders are called "multirun path" in this
module's documentation).
"""
import os
import re
from typing import Tuple, Dict, List, Union, Optional
import numpy as np
import torch
from torch import Tensor
from omegaconf import OmegaConf
from bayes_dip.utils.experiment_utils import get_standard_ray_trafo, get_predefined_patch_idx_list
from bayes_dip.probabilistic_models import (
        get_trafo_t_trafo_pseudo_inv_diag_mean, get_default_unet_gaussian_prior_dicts, ImageCov,
        NeuralBasisExpansion, MatmulNeuralBasisExpansion, ParameterCov)
from bayes_dip.inference import get_image_patch_mask_inds
from bayes_dip.inference.sample_based_predictive_posterior import (
        predictive_cov_image_patch_log_prob_unscaled_batched)
from bayes_dip.dip import UNet

DEFAULT_OUTPUTS_PATH = '../experiments/outputs'
DEFAULT_MULTIRUN_PATH = '../experiments/multirun'

def is_single_level_date_time_path(path: str):
    """
    Check whether a hydra output or multirun path has date and time encoded in a single folder name
    as specified in our custom configs ``hydra.run.dir=outputs/${now:%Y-%m-%dT%H:%M:%S.%fZ}`` and
    ``hydra.sweep.dir=multirun/${now:%Y-%m-%dT%H:%M:%S.%fZ}`` (as opposed to the default config that
    uses a date folder and a time folder inside).
    """
    path = path.rstrip('/\\')
    if re.match(
            r"([0-9]){4}-([0-9]){2}-([0-9]){2}T([0-9]){2}:([0-9]){2}:([0-9]){2}.([0-9]){6}Z",
            os.path.basename(path)):
        # our custom single-level format
        is_single_level = True
    elif (
            re.match(r"([0-9]){4}-([0-9]){2}-([0-9]){2}",
                    os.path.basename(os.path.dirname(path))) and
            re.match(r"([0-9]){2}-([0-9]){2}-([0-9]){2}",
                    os.path.basename(path))):
        # default format
        is_single_level = False
    else:
        raise ValueError('unknown path format')
    return is_single_level

def translate_output_path(path: str, outputs_path: Optional[str] = DEFAULT_OUTPUTS_PATH):
    """
    Translate a hydra output path (e.g. "arbitrary/path/to/outputs/???") to a new outputs root path
    (i.e. "{outputs_path}/???").
    """
    path = path.rstrip('/\\')
    if outputs_path is not None:
        if is_single_level_date_time_path(path):
            path = os.path.join(
                    outputs_path,
                    os.path.basename(path))
        else:
            path = os.path.join(
                    outputs_path,
                    os.path.basename(os.path.dirname(path)),
                    os.path.basename(path))
    return path

def translate_multirun_path(path: str, multirun_path: Optional[str] = DEFAULT_MULTIRUN_PATH):
    """
    Translate a hydra multirun path (e.g. "arbitrary/path/to/multirun/???/?") to a new multirun root
    path (i.e. "{multirun_path}/???/?").
    """
    path = path.rstrip('/\\')
    if multirun_path is not None:
        if is_single_level_date_time_path(os.path.dirname(path)):
            path = os.path.join(
                    multirun_path,
                    os.path.basename(os.path.dirname(path)),
                    os.path.basename(path))
        else:
            path = os.path.join(
                    multirun_path,
                    os.path.basename(os.path.dirname(os.path.dirname(path))),
                    os.path.basename(os.path.dirname(path)),
                    os.path.basename(path))
    return path

def translate_path(
        path: str,
        experiment_paths: Optional[Dict] = None):
    """
    Translate a hydra output or multirun path (e.g. "arbitrary/path/to/outputs/???" or
    "arbitrary/path/to/multirun/???/?") to a new outputs or multirun root path (i.e.
    "{experiment_paths['outputs_path']}/???" or "{experiment_paths['multirun_path']}/???/?").

    Parameters
    ----------
    path : str
        Path of the hydra run.
    experiment_paths : dict, optional
        New outputs and/or multirun root path. The respective keys are ``'outputs_path'`` and
        ``'multirun_path'``, with defaults ``DEFAULT_OUTPUTS_PATH`` and ``DEFAULT_MULTIRUN_PATH``.

    Returns
    -------
    path : str
        Translated path.
    """
    path = path.rstrip('/\\')
    experiment_paths = experiment_paths or {}
    is_multirun = all(c in '0123456789' for c in os.path.basename(path))
    if is_multirun:
        multirun_path = experiment_paths.get('multirun_path', None)
        multirun_path = multirun_path if multirun_path is not None else DEFAULT_MULTIRUN_PATH
        path = translate_multirun_path(path=path, multirun_path=multirun_path)
    else:
        outputs_path = experiment_paths.get('outputs_path', None)
        outputs_path = outputs_path if outputs_path is not None else DEFAULT_OUTPUTS_PATH
        path = translate_output_path(path=path, outputs_path=outputs_path)
    return path

def get_ground_truth(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None) -> Tensor:
    """
    Return the ground truth for sample ``sample_idx`` from ``run_path``.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    experiment_paths : dict, optional
        See :func:`translate_path`.

    Returns
    -------
    ground_truth : Tensor
        Ground truth. Shape: ``(im_size, im_size)``.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    ground_truth = torch.load(os.path.join(run_path, f'sample_{sample_idx}.pt'),
            map_location='cpu')['ground_truth'].detach()
    return ground_truth.squeeze(1).squeeze(0)

def get_observation(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None) -> Tensor:
    """
    Return the ground truth for sample ``sample_idx`` from ``run_path``.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    experiment_paths : dict, optional
        See :func:`translate_path`.

    Returns
    -------
    observation : Tensor
        Ground truth. Shape: ``(im_size, im_size)``.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    observation = torch.load(os.path.join(run_path, f'sample_{sample_idx}.pt'),
            map_location='cpu')['observation'].detach()
    return observation.squeeze(1).squeeze(0)

def get_recon(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None) -> Tensor:
    """
    Return the reconstruction for sample ``sample_idx`` from ``run_path``.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    experiment_paths : dict, optional
        See :func:`translate_path`.

    Returns
    -------
    recon : Tensor
        Reconstruction. Shape: ``(im_size, im_size)``.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    recon = torch.load(os.path.join(run_path, f'recon_{sample_idx}.pt'),
            map_location='cpu').detach()
    return recon.squeeze(1).squeeze(0)

def get_abs_diff(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None) -> Tensor:
    """
    Return the absolute difference between reconstruction and ground truth for sample ``sample_idx``
    from ``run_path``.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    experiment_paths : dict, optional
        See :func:`translate_path`.

    Returns
    -------
    abs_diff : Tensor
        Absolute difference. Shape: ``(im_size, im_size)``.
    """
    ground_truth = get_ground_truth(
            run_path=run_path, sample_idx=sample_idx, experiment_paths=experiment_paths)
    recon = get_recon(
            run_path=run_path, sample_idx=sample_idx, experiment_paths=experiment_paths)
    abs_diff = torch.abs(ground_truth - recon)
    return abs_diff

def get_density_data(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None) -> Tuple[Dict, bool]:
    """
    Return the density data for sample ``sample_idx`` from ``run_path``.

    Supports both ``experiments/exact_density.py`` and ``experiments/sample_based_density.py`` runs.

    Returns
    -------
    data : dict
        Density data from "exact_predictive_posterior_{sample_idx}.pt" or
        "sample_based_predictive_posterior_{sample_idx}.pt".
    is_exact : bool
        If ``True``, the run is a ``experiments/exact_density.py`` run; if ``False`` it is a
        ``experiments/sample_based_density.py`` run. If the run is of neither kind, a
        ``RuntimeError`` is raised.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    exact_filepath = os.path.join(
            run_path, f'exact_predictive_posterior_{sample_idx}.pt')
    sample_based_filepath = os.path.join(
            run_path, f'sample_based_predictive_posterior_{sample_idx}.pt')
    if os.path.isfile(exact_filepath):
        filepath, is_exact = exact_filepath, True
    elif os.path.isfile(sample_based_filepath):
        filepath, is_exact = sample_based_filepath, False
    else:
        raise RuntimeError('Cannot find density data.')
    data = torch.load(filepath, map_location='cpu')
    return data, is_exact

def _recompute_image_noise_correction_term(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None) -> float:
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    ray_trafo = get_standard_ray_trafo(cfg)
    observation_cov_filename = (
            f'observation_cov_{sample_idx}.pt' if cfg.inference.load_iter is None else
            f'observation_cov_{sample_idx}_iter_{cfg.inference.load_iter}.pt')
    log_noise_variance = torch.load(os.path.join(
            translate_path(cfg.inference.load_path, experiment_paths=experiment_paths),
            observation_cov_filename),
            map_location='cpu')['log_noise_variance']
    diag_mean = get_trafo_t_trafo_pseudo_inv_diag_mean(ray_trafo)
    image_noise_correction_term = diag_mean * log_noise_variance.exp().item()
    return image_noise_correction_term

def get_nn_model(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None,
        device=None,
        ) -> Tensor:
    """
    Return the U-Net, with the model parameters loaded from the specified run, and the filtered
    back-projection (which is the network input).

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    experiment_paths : dict, optional
        See :func:`translate_path`.
    device : str or torch.device, optional
        Device.

    Returns
    -------
    nn_model : :class:`bayes_dip.dip.network.unet.UNet`
        U-Net used for DIP, with loaded model parameters.
    filtbackproj : Tensor
        Filtered back-projection (network input). Shape: ``(1, 1, *im_shape)``.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    assert not cfg.dip.recon_from_randn  # would need to re-create random input
    net_kwargs = {
            'scales': cfg.dip.net.scales,
            'channels': cfg.dip.net.channels,
            'skip_channels': cfg.dip.net.skip_channels,
            'use_norm': cfg.dip.net.use_norm,
            'use_sigmoid': cfg.dip.net.use_sigmoid,
            'sigmoid_saturation_thresh': cfg.dip.net.sigmoid_saturation_thresh}
    filtbackproj = torch.load(
            os.path.join(run_path, f'sample_{sample_idx}.pt'), map_location=device)['filtbackproj']
    nn_model = UNet(
            in_ch=1,
            out_ch=1,
            channels=net_kwargs['channels'][:net_kwargs['scales']],
            skip_channels=net_kwargs['skip_channels'][:net_kwargs['scales']],
            use_sigmoid=net_kwargs['use_sigmoid'],
            use_norm=net_kwargs['use_norm'],
            sigmoid_saturation_thresh=net_kwargs['sigmoid_saturation_thresh']
            ).to(device)
    nn_model.load_state_dict(
            torch.load(os.path.join(run_path, f'dip_model_{sample_idx}.pt'), map_location=device))
    return nn_model, filtbackproj

def recompute_reconstruction(
        run_path: str, sample_idx: int,
        experiment_paths: Optional[Dict] = None,
        device=None,
        ) -> Tensor:
    """
    Recompute the reconstruction from saved network model parameters.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    experiment_paths : dict, optional
        See :func:`translate_path`.
    device : str or torch.device, optional
        Device.

    Returns
    -------
    reconstruction : Tensor
        DIP reconstruction. Shape: ``(1, 1, *im_shape)``.
    """
    nn_model, filtbackproj = get_nn_model(
            run_path=run_path, sample_idx=sample_idx, experiment_paths=experiment_paths,
            device=device)
    recon = nn_model(filtbackproj).detach()  # pylint: disable=not-callable
    return recon

def get_patch_idx_to_mask_inds_dict(
        patch_idx_list: Union[List[int], str, None], im_shape: Tuple[int, int], patch_size: int):
    """
    Return a dictionary with the selected patch indices as keys and the mask index arrays defining
    the patches as values.
    """
    all_patch_mask_inds = get_image_patch_mask_inds(im_shape, patch_size=patch_size)
    if patch_idx_list is None:
        patch_idx_list = list(range(len(all_patch_mask_inds)))
    elif isinstance(patch_idx_list, str):
        patch_idx_list = get_predefined_patch_idx_list(
                name=patch_idx_list, patch_size=patch_size)
    return {idx: all_patch_mask_inds[idx] for idx in patch_idx_list}

def get_sample_based_cov_diag(
        data: Dict,
        patch_idx_list: Union[List[int], str, None],
        patch_size: int,
        im_shape: Tuple[int, int]
        ) -> Tensor:
    """
    Return the diagonal of the posterior covariance for the specified ``patch_idx_list``.

    Parameters
    ----------
    data : dict
        Density data, like the first return value of :func:`get_density_data`.
    patch_idx_list : list of int, str or None
        Patch indices for which to populate the returned tensor. The pixels of patches that are not
        selected will be ``torch.nan`` in the returned tensor.
        Must be a subset of the ``patch_idx_list`` from the original run.
    patch_size : int
        Side length of the patches (patches are usually square). Must be the same as for the
        original run.
    im_shape : 2-tuple of int
        Image shape.

    Returns
    -------
    cov_diag : Tensor
        Diagonal of the posterior covariance. Shape: ``im_shape``.
    """

    # fill in all patches from data
    cov_diag = torch.full(im_shape, torch.nan, dtype=data['patch_cov_diags'][0].dtype)
    for mask_inds, diag in zip(data['patch_mask_inds'], data['patch_cov_diags']):
        cov_diag.view(-1)[mask_inds] = diag.detach()

    cov_diag_requested_mask = torch.zeros(im_shape, dtype=torch.bool)
    for mask_inds in get_patch_idx_to_mask_inds_dict(
            patch_idx_list=patch_idx_list, im_shape=im_shape,
            patch_size=patch_size).values():
        cov_diag_requested_mask.view(-1)[mask_inds] = True
    # assert that the parts inside the requested patch_idx_list are filled in
    assert not torch.any(torch.isnan(cov_diag[cov_diag_requested_mask]))
    # set parts outside the requested patch_idx_list to nan
    cov_diag[torch.bitwise_not(cov_diag_requested_mask)] = torch.nan

    return cov_diag

def restrict_sample_based_density_data_to_new_patch_idx_list(
        data: Dict,
        patch_idx_list: Union[List[int], str, None],
        orig_patch_idx_list: Union[List[int], str, None],
        patch_size: int,
        im_shape: Tuple[int, int]
        ) -> Dict:
    """
    Return a version of sample based density data that is restricted to a subset of patch indices.

    The log probability is also recomputed with the new patch selection.

    Parameters
    ----------
    data : dict
        The original density data, like the first return value of :func:`get_density_data`.
    patch_idx_list : list of int, str or None
        Patch indices to restrict to.
        Must be a subset of ``orig_patch_idx_list``.
        If a string,
        ``bayes_dip.utils.experiment_utils.get_predefined_patch_idx_list(patch_idx_list)`` is used.
        If ``None``, all patch indices are used.
    orig_patch_idx_list : list of int, str or None
        Value of ``cfg.inference.patch_idx_list`` from the original run.
    patch_size : int
        Side length of the patches (patches are usually square). Must be the same as for the
        original run.
    im_shape : 2-tuple of int
        Image shape.

    Returns
    -------
    data_restricted : dict
        Restricted density data.
    """
    orig_patch_idx_list = list(get_patch_idx_to_mask_inds_dict(  # original indices
            patch_idx_list=orig_patch_idx_list, im_shape=im_shape,
            patch_size=patch_size))
    patch_idx_to_mask_inds_dict = get_patch_idx_to_mask_inds_dict(  # news indices and mask inds
            patch_idx_list=patch_idx_list, im_shape=im_shape,
            patch_size=patch_size)
    assert all(a < b for a, b in zip(orig_patch_idx_list[:-1], orig_patch_idx_list[1:]))
    indices = np.searchsorted(orig_patch_idx_list, list(patch_idx_to_mask_inds_dict))
    data_restricted = {
        'patch_mask_inds': [data['patch_mask_inds'][i] for i in indices],
        'patch_log_probs_unscaled': [data['patch_log_probs_unscaled'][i] for i in indices],
        'log_prob': None,  # fill later
        'patch_cov_diags': [data['patch_cov_diags'][i] for i in indices],
        'image_noise_correction_term': data['image_noise_correction_term'],
    }
    assert all(np.array_equal(mask_inds, orig_mask_inds) for mask_inds, orig_mask_inds in zip(
            patch_idx_to_mask_inds_dict.values(), data_restricted['patch_mask_inds']))
    total_num_pixels_in_patches_restricted = sum(
            len(mask_inds) for mask_inds in data_restricted['patch_mask_inds'])
    data_restricted['log_prob'] = np.sum(
            data_restricted['patch_log_probs_unscaled']) / total_num_pixels_in_patches_restricted
    return data_restricted

def get_stddev(run_path: str, sample_idx: int,
        patch_idx_list: Optional[Union[List[int], str]] = None,
        subtract_image_noise_correction_if_any: bool = True,
        experiment_paths: Optional[Dict] = None) -> Tensor:
    """
    Return the standard deviation (i.e. the square root of the diagonal of the posterior covariance)
    for sample ``sample_idx`` from ``run_path``.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    patch_idx_list : list of int or str, optional
        Patch indices to restrict to. Only supported with sample based density runs.
        Must be a subset of ``cfg.inference.patch_idx_list`` of the original run.
        If a string,
        ``bayes_dip.utils.experiment_utils.get_predefined_patch_idx_list(patch_idx_list)`` is used.
        If ``None`` (the default), all patch indices are used.
    subtract_image_noise_correction_if_any : bool, optional
        Whether to subtract the image noise correction term from the covariance diagonal before
        taking the square root.
        The default is ``True``.
    experiment_paths : dict, optional
        See :func:`translate_path`.

    Returns
    -------
    stddev : Tensor
        Standard deviation. Shape: ``(im_size, im_size)``.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    data, is_exact = get_density_data(
            run_path=run_path, sample_idx=sample_idx, experiment_paths=experiment_paths)
    if is_exact:
        assert patch_idx_list is None, 'cannot use patch_idx_list with exact density'
        cov_diag = data['cov'].detach().diag().reshape((cfg.dataset.im_size,) * 2)
    else:
        cov_diag = get_sample_based_cov_diag(
                data=data,
                patch_idx_list=patch_idx_list,
                patch_size=cfg.inference.patch_size,
                im_shape=(cfg.dataset.im_size,) * 2)
    if (subtract_image_noise_correction_if_any and cfg.inference.get(
            'add_image_noise_correction_term',
            True,  # in old runs without this config, image_noise_correction_term was added
            )):
        image_noise_correction_term = data.get('image_noise_correction_term', None)
        if image_noise_correction_term is None:
            image_noise_correction_term = _recompute_image_noise_correction_term(
                    run_path=run_path, sample_idx=sample_idx, experiment_paths=experiment_paths)
        print(f'subtracting {image_noise_correction_term} (image noise correction) from cov_diag')
        cov_diag -= image_noise_correction_term
    stddev = torch.sqrt(cov_diag)
    return stddev

def compute_log_prob_for_patch_size_from_cov(
        cov: Tensor,
        patch_size: int,
        mean: Tensor,
        ground_truth: Tensor,
        ) -> float:
    """
    Compute a patch-based log probability from a given covariance matrix, e.g. computed by
    ``experiments/exact_density.py``.

    Parameters
    ----------
    cov : Tensor
        Covariance matrix. Shape: ``(np.prod(im_shape), np.prod(im_shape))``.
    patch_size : int
        Patch size.
    mean : Tensor
        Mean of the posterior image distribution. Shape: ``(1, 1, *im_shape)``.
    ground_truth : Tensor
        Ground truth. Shape: ``(1, 1, *im_shape)``.

    Returns
    -------
    log_prob : float
        Patch-based log probability.
    """
    assert len(cov.shape) == 2
    assert len(mean.shape) >= 2 and np.prod(mean.shape[:-2]) == 1
    im_shape = mean.shape[-2:]
    assert np.prod(im_shape) == cov.shape[0]
    cov = cov.detach()
    all_patch_mask_inds = get_image_patch_mask_inds(im_shape, patch_size=patch_size)
    # compute all log probs in a single batch (cov should be small since it is passed as a matrix)
    batch_len_mask_inds = [len(mask_inds) for mask_inds in all_patch_mask_inds]
    max_len_mask_inds = max(batch_len_mask_inds)
    batch_recon = torch.stack([
            torch.nn.functional.pad(mean.flatten()[all_patch_mask_inds[patch_idx]],
                    (0, max_len_mask_inds - len_mask_inds))
            for patch_idx, len_mask_inds in enumerate(batch_len_mask_inds)])
    batch_ground_truth = torch.stack([
            torch.nn.functional.pad(ground_truth.flatten()[all_patch_mask_inds[patch_idx]],
                    (0, max_len_mask_inds - len_mask_inds))
            for patch_idx, len_mask_inds in enumerate(batch_len_mask_inds)])
    # use identity for padding dims in predictive_cov_image_patch
    # (the determinant then is the same as for
    #  predictive_cov_image_patch[:len(mask_inds), :len(mask_inds)])
    batch_predictive_cov_image_patch = torch.stack([
            torch.block_diag(
                    cov[mask_inds][:, mask_inds],
                    torch.eye(max_len_mask_inds-len(mask_inds),
                            dtype=cov.dtype, device=cov.device))
            for mask_inds in all_patch_mask_inds])
    log_probs_unscaled = predictive_cov_image_patch_log_prob_unscaled_batched(
            recon_masked=batch_recon,
            ground_truth_masked=batch_ground_truth,
            predictive_cov_image_patch=batch_predictive_cov_image_patch)
    return (np.sum(log_probs_unscaled.cpu().numpy()) / cov.shape[0]).item()

def get_image_cov(run_path: str, sample_idx: int,
        iter: Optional[int] = None, use_matmul_neural_basis_expansion: bool = False,
        experiment_paths: Optional[Dict] = None, device=None) -> ImageCov:
    """
    Reinstantiate the image covariance object with the network parameters and prior hyperparameter
    state from a run of ``experiments/dip_mll_optim.py`` or ``experiments/exact_dip_mll_optim.py``.

    Does not support the isotropic g-prior, i.e. ``cfg.priors.use_gprior`` must be ``False``.

    Parameters
    ----------
    run_path : str
        Path of the hydra run.
    sample_idx : int
        Sample index.
    iter : int, optional
        Iteration of the MLL optimization from which to load the hyperparameters.
        If `None`, the final state is used.
    use_matmul_neural_basis_expansion : bool, optional
        Whether to use :class:`MatmulNeuralBasisExpansion` instead of :class:`NeuralBasisExpansion`.
    experiment_paths : dict, optional
        See :func:`translate_path`.
    device : str or torch.device, optional
        Device.

    Returns
    -------
    image_cov : :class:`bayes_dip.probabilistic_models.linearized_dip.image_cov.ImageCov`
        Image covariance object.
    """
    run_path = translate_path(run_path, experiment_paths=experiment_paths)
    device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))

    nn_model, filtbackproj = get_nn_model(
            run_path=run_path, sample_idx=sample_idx, experiment_paths=experiment_paths,
            device=device)

    prior_assignment_dict, hyperparams_init_dict = get_default_unet_gaussian_prior_dicts(
            nn_model)
    parameter_cov = ParameterCov(
            nn_model,
            prior_assignment_dict,
            hyperparams_init_dict,
            device=device
    )
    assert not cfg.priors.use_gprior
    neural_basis_expansion_kwargs = dict(
            nn_model=nn_model,
            nn_input=filtbackproj,
            ordered_nn_params=parameter_cov.ordered_nn_params,
            nn_out_shape=filtbackproj.shape)
    if use_matmul_neural_basis_expansion:
        neural_basis_expansion = MatmulNeuralBasisExpansion(**neural_basis_expansion_kwargs)
    else:
        neural_basis_expansion = NeuralBasisExpansion(**neural_basis_expansion_kwargs)
    image_cov = ImageCov(
            parameter_cov=parameter_cov,
            neural_basis_expansion=neural_basis_expansion
    )

    observation_cov_filename = (
            f'observation_cov_{sample_idx}.pt' if iter is None else
            f'observation_cov_{sample_idx}_iter_{iter}.pt')
    observation_cov_state_dict = torch.load(
            os.path.join(run_path, observation_cov_filename), map_location='cpu')

    image_cov_state_dict = {
            k[len('image_cov.'):]: v for k, v in observation_cov_state_dict.items()
            if k.startswith('image_cov.')}
    image_cov.load_state_dict(image_cov_state_dict)

    return image_cov

def find_single_log_file(log_dir: str) -> str:
    """
    Return path of a single log file found in a log directory by recursing its sub-directories.

    Parameters
    ----------
    log_dir : str
        Log directory that contains exactly one log file in any sub-directory (including itself).

    Returns
    -------
    log_file : str
        Full path of the single log file (the path includes ``log_dir`` as a prefix).
    """
    log_file = None
    for path, _, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                if log_file is None:
                    log_file = os.path.join(path, file)
                else:
                    raise RuntimeError(f'found multiple log files in {log_dir}')
    if log_file is None:
        raise RuntimeError(f'did not find log file in {log_dir}')
    return log_file

def extract_tensorboard_scalars(
        log_file: str, save_as_npz: str = '', tags: Optional[List[str]] = None) -> dict:
    """
    Extract scalars from a tensorboard log file.

    Parameters
    ----------
    log_file : str
        Tensorboard log filepath.
    save_as_npz : str, optional
        File path to save the extracted scalars as a npz file.
    tags : list of str, optional
        If specified, only extract these tags.
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError:
        raise RuntimeError('Tensorboard\'s event_accumulator could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

    ea = event_accumulator.EventAccumulator(
            log_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    tags = tags or ea.Tags()['scalars']

    scalars = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        scalars[tag + '_steps'] = np.asarray(steps)
        scalars[tag + '_scalars'] = np.asarray(values)

    if save_as_npz:
        np.savez(save_as_npz, **scalars)

    return scalars
