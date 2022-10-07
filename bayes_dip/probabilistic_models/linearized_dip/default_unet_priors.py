"""
Provides getters for default prior assignment and hyperparameter initialization dictionaries.
"""

from typing import Optional, Dict, Tuple, Callable, List
from torch.nn import Conv2d
from bayes_dip.dip.network.unet import UNet
from bayes_dip.probabilistic_models.linearized_dip.parameter_priors import (
        get_GPprior_RadialBasisFuncCov, NormalPrior, IsotropicPrior)
from bayes_dip.utils import get_modules_by_names


def get_default_unet_gaussian_prior_dicts(
        nn_model: UNet,
        gaussian_prior_hyperparams_init: Optional[Dict] = None,
        normal_prior_hyperparams_init: Optional[Dict] = None,
        ) -> Tuple[Dict[str, Tuple[Callable, List[str]]], Dict[str, Dict]]:
    """
    Return default prior assignment and hyperparameter initialization dictionaries for the
    linearized DIP with GP/Normal priors for the U-Net.

    One prior is assigned per convolutional block (not layer).
    GP priors are placed over 3x3 convolutions and Normal priors are placed over 1x1 convolutions.

    Parameters
    ----------
    nn_model : :class:`bayes_dip.dip.network.unet.UNet`
        Network.
    gaussian_prior_hyperparams_init : dict, optional
        Custom initial values for the variances and lengthscales of GP priors.
        The default is ``{'variance': 1., 'lengthscale': 0.1}``.
    normal_prior_hyperparams_init : dict, optional
        Custom initial values for the variances of Normal priors.
        The default is ``{'variance': 1.}``.

    Returns
    -------
    prior_assignment_dict : dict
        Dictionary defining priors over the convolutional modules of the network.
        E.g., for a U-Net with 3 scales, no skip connections and without norm layers the prior
        assignment is:

        .. code-block:: python

            prior_assignment_dict = {
                'inc': (get_GPprior_RadialBasisFuncCov, ['inc.conv.0']),
                'down_0': (get_GPprior_RadialBasisFuncCov, ['down.0.conv.0', 'down.0.conv.2']),
                'down_1': (get_GPprior_RadialBasisFuncCov, ['down.1.conv.0', 'down.1.conv.2']),
                'up_0': (get_GPprior_RadialBasisFuncCov, ['up.0.conv.0', 'up.0.conv.2']),
                'up_1': (get_GPprior_RadialBasisFuncCov, ['up.1.conv.0', 'up.1.conv.2']),
                'outc': (NormalPrior, ['outc.conv']),
            }
    hyperparams_init_dict : dict
        Dictionary defining the initial hyperparameter values for the priors.
        It has the same keys as ``prior_assignment_dict``, containing
        ``gaussian_prior_hyperparams_init`` for the GP priors and ``normal_prior_hyperparams_init``
        for the Normal priors (filled with the default initial values if any of them is not passed).
    """

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
    """
    Return default prior assignment and hyperparameter initialization dictionaries for the isotropic
    g-prior for the U-Net.

    A single isotropic prior is assigned to all convolutional modules.

    Parameters
    ----------
    nn_model : :class:`bayes_dip.dip.network.unet.UNet`
        Network.
    gprior_hyperparams_init : dict, optional
        Custom initial value for the variance of the isotropic prior.
        The default is ``{'variance': 1.}``.

    Returns
    -------
    prior_assignment_dict : dict
        Dictionary defining the single isotropic g-prior over all modules.
        E.g., for a U-Net with 3 scales, no skip connections and without norm layers the prior
        assignment is:

        .. code-block:: python

            prior_assignment_dict = {
                'gprior': (IsotropicPrior, ['inc.conv.0', 'down.0.conv.0', 'down.0.conv.2',
                        'down.1.conv.0', 'down.1.conv.2', 'up.0.conv.0', 'up.0.conv.2',
                        'up.1.conv.0', 'up.1.conv.2', 'outc.conv']),
            }
    hyperparams_init_dict : dict
        Dictionary defining the initial variance value for the isotropic g-prior.
        It has the same single key as ``prior_assignment_dict``, containing
        ``gprior_hyperparams_init`` (filled with the default initial value if not passed).
    """

    if gprior_hyperparams_init is None:
        gprior_hyperparams_init = {}
    gprior_hyperparams_init.setdefault('variance', .01)

    prior_assignment_dict = {}
    hyperparams_init_dict = {}

    prior_assignment_dict['gprior'] = (IsotropicPrior, [
        f'{name}' for name, module in nn_model.named_modules()
        if isinstance(module, Conv2d)])

    hyperparams_init_dict['gprior'] = gprior_hyperparams_init.copy()
    return prior_assignment_dict, hyperparams_init_dict
