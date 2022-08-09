from typing import Optional, Dict, Tuple, Callable, List
from torch.nn import Conv2d
from bayes_dip.dip.network.unet import UNet
from bayes_dip.probabilistic_models.linearized_dip.parameter_priors import (
        get_GPprior_RadialBasisFuncCov, NormalPrior, IsotropicPrior)
from bayes_dip.utils import get_modules_by_names
from bayes_dip.data import BaseRayTrafo

def get_default_unet_gaussian_prior_dicts(
        nn_model: UNet,
        gaussian_prior_hyperparams_init: Optional[Dict] = None,
        normal_prior_hyperparams_init: Optional[Dict] = None,
        ) -> Tuple[Dict[str, Tuple[Callable, List[str]]], Dict[str, Dict]]:

    if gaussian_prior_hyperparams_init is None:
        gaussian_prior_hyperparams_init = {}
    gaussian_prior_hyperparams_init.setdefault('variance', 1.)
    gaussian_prior_hyperparams_init.setdefault('lengthscale', 0.1)

    if normal_prior_hyperparams_init is None:
        normal_prior_hyperparams_init = {}
    normal_prior_hyperparams_init.setdefault('variance', 1.)

    prior_assignment_dict = {}
    hyperparams_init_dict = {}

    prior_assignment_dict['inc'] = (get_GPprior_RadialBasisFuncCov, [
        f'inc.conv.{name}' for name, module in nn_model.inc.conv.named_modules()
        if isinstance(module, Conv2d)])
    hyperparams_init_dict['inc'] = gaussian_prior_hyperparams_init.copy()

    for i, down_module in enumerate(nn_model.down):
        prior_assignment_dict[f'down_{i}'] = (get_GPprior_RadialBasisFuncCov, [
            f'down.{i}.conv.{name}'
            for name, module in down_module.conv.named_modules()
            if isinstance(module, Conv2d)])
        hyperparams_init_dict[f'down_{i}'] = gaussian_prior_hyperparams_init.copy()

    for i, up_module in enumerate(nn_model.up):
        prior_assignment_dict[f'up_{i}'] = (get_GPprior_RadialBasisFuncCov, [
            f'up.{i}.conv.{name}'
            for name, module in up_module.conv.named_modules()
            if isinstance(module, Conv2d)])
        hyperparams_init_dict[f'up_{i}'] = gaussian_prior_hyperparams_init.copy()

    for i, up_module in enumerate(nn_model.up):
        if up_module.skip:  # uses skip_conv
            prior_assignment_dict[f'skip_{i}'] = (NormalPrior, [
                f'up.{i}.skip_conv.{name}'
                for name, module in up_module.skip_conv.named_modules()
                if isinstance(module, Conv2d)])
        hyperparams_init_dict[f'skip_{i}'] = normal_prior_hyperparams_init.copy()

    assert isinstance(nn_model.outc.conv, Conv2d)
    prior_assignment_dict['outc'] = (NormalPrior, ['outc.conv'])
    hyperparams_init_dict['outc'] = normal_prior_hyperparams_init.copy()

    # assert that prior_type functions and kernel size match
    for prior_type, convs in prior_assignment_dict.values():
        if prior_type is NormalPrior:
            assert all(
                    conv.kernel_size[0] * conv.kernel_size[1] == 1
                    for conv in get_modules_by_names(nn_model, convs))
        else:  # GPprior
            assert all(
                    conv.kernel_size[0] * conv.kernel_size[1] > 1
                    for conv in get_modules_by_names(nn_model, convs))

    return prior_assignment_dict, hyperparams_init_dict


def get_default_unet_gprior_dicts(
        nn_model: UNet,
        gprior_hyperparams_init: Optional[Dict] = None,
        ) -> Tuple[Dict[str, Tuple[Callable, List[str]]], Dict[str, Dict]]:

    if gprior_hyperparams_init is None:
        gprior_hyperparams_init = {}
    gprior_hyperparams_init.setdefault('variance', 1.)

    prior_assignment_dict = {}
    hyperparams_init_dict = {}

    prior_assignment_dict['gprior'] = (IsotropicPrior, [
        f'{name}' for name, module in nn_model.named_modules() 
        if isinstance(module, Conv2d)]) 

    hyperparams_init_dict['gprior'] = gprior_hyperparams_init.copy()
    return prior_assignment_dict, hyperparams_init_dict