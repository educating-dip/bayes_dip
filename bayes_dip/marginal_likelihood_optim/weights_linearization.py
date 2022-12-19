"""
Provides an optimization routine for the network weights in the linearized model,
:func:`weights_linearization`.
"""
from typing import Tuple
import torch
from torch import nn, Tensor
from tqdm import tqdm
from bayes_dip.data.trafo.base_ray_trafo import BaseRayTrafo
from bayes_dip.probabilistic_models import BaseNeuralBasisExpansion
from ..utils import batch_tv_grad, PSNR, eval_mode  # pylint: disable=unused-import

def weights_linearization(
        trafo: BaseRayTrafo,
        neural_basis_expansion: BaseNeuralBasisExpansion,
        use_sigmoid: bool,
        map_weights: Tensor,
        observation: Tensor,
        ground_truth: Tensor,
        optim_kwargs: dict,
        ) -> Tuple[Tensor, Tensor]:
    # pylint: disable=too-many-locals
    """
    Optimize the network weights in the linearized model, with the same loss as for the
    TV-regularized DIP.

    Parameters
    ----------
    trafo : :class:`.BaseRayTrafo`
        Ray transform.
    neural_basis_expansion : :class:`.BaseNeuralBasisExpansion`
        Neural basis expansion of a :class:`~bayes_dip.dip.UNet` or
        :class:`bayes_dip.dip.UNetReturnPreSigmoid` model. See ``use_sigmoid`` for instructions
        on how to handle final sigmoid activations.
    use_sigmoid : bool
        Whether to apply sigmoid on the output of the linear model; useful to keep a final sigmoid
        activation out of the linearization, i.e., by passing a ``neural_basis_expansion`` for a
        version of the network that excludes the final sigmoid, and passing ``use_sigmoid=True``.
        If the network does not have a final sigmoid activation, or if the full model including
        sigmoid should be linearized, ``use_sigmoid=False`` should be passed (and
        ``neural_basis_expansion.nn_model`` should include the sigmoid if any).
    map_weights : Tensor
        MAP weights (DIP network model weights). Shape: ``(neural_basis_expansion.jac_shape[1],)``.
    observation : Tensor
        Observation. Shape: ``(1, 1, *trafo.obs_shape)``.
    ground_truth : Tensor
        Ground truth. Shape: ``(1, 1, *trafo.im_shape)``.
    optim_kwargs : dict
        Optimization keyword arguments (most are required). The arguments are:

        ``'iterations'`` : int
            Number of iterations.
        ``'lr'`` : float
            Learning rate.
        ``'simplified_eqn'`` : bool
            Whether to use the simplified model ``J @ lin_weights`` instead of the "standard" model
            ``J @ (lin_weights - map_weights) + pre_sigmoid_recon`` (each model is followed by
            sigmoid if ``neural_basis_expansion.nn_model.use_sigmoid``), where ``J`` is given by
            ``neural_basis_expansion`` and ``pre_sigmoid_recon`` is the output of
            ``neural_basis_expansion.nn_model`` with ``return_pre_sigmoid=True``.
        ``'noise_precision'`` : float
            Weighting factor for the data discrepancy term (should usually be ``1.``).
        ``'gamma'`` : float
            Weighting factor of the TV loss term (should usually be the same as for the DIP
            optimization).
        ``'wd'`` : float
            Weight decay rate.

    Returns
    -------
    lin_weights : Tensor
        Weights for the linearized model. Shape: ``(neural_basis_expansion.jac_shape[1],)``.
    lin_recon : Tensor
        Reconstruction. Shape: ``(1, 1, *trafo.im_shape)``.
    """

    nn_model = neural_basis_expansion.nn_model
    nn_input = neural_basis_expansion.nn_input

    with torch.no_grad():
        nn_model_recon = nn_model(nn_input, saturation_safety=True)

    lin_weights_fd = (
            nn.Parameter(torch.zeros_like(map_weights)) if optim_kwargs['simplified_eqn']
            else map_weights.clone())
    optimizer = torch.optim.Adam([lin_weights_fd], lr=optim_kwargs['lr'], weight_decay=0)
    
    # optimizer = torch.optim.SGD([lin_weights_fd], lr=optim_kwargs['lr'], nesterov=True, momentum=0.9, weight_decay=0.)

    precision = optim_kwargs['noise_precision']

    with tqdm(range(optim_kwargs['iterations']),
                miniters=optim_kwargs['iterations']//1) as pbar, \
            eval_mode(nn_model):
        for _ in pbar:

            if optim_kwargs['simplified_eqn']:
                fd_vector = lin_weights_fd
            else:
                fd_vector = lin_weights_fd - map_weights

            # J_tilde w
            lin_recon = neural_basis_expansion.jvp(fd_vector[None, :]).detach().squeeze(dim=1)

            if not optim_kwargs['simplified_eqn']:
                lin_recon = lin_recon + nn_model_recon

            if use_sigmoid:
                lin_recon = lin_recon.sigmoid()

            # A J_tilde w
            proj_lin_recon = trafo(lin_recon)
            
            observation = observation.view(*proj_lin_recon.shape)
            norm_grad = trafo.trafo_adjoint( observation - proj_lin_recon ) # A^T (y_tilde - A J_tilde w)
            tv_grad = batch_tv_grad(lin_recon)

            # loss = (torch.nn.functional.mse_loss(
            #                 proj_lin_recon, observation.view(*proj_lin_recon.shape))
            #         + optim_kwargs['gamma'] * tv_loss(lin_recon))
            # v = - 2 / observation.numel() * precision * norm_grad + optim_kwargs['gamma'] * tv_grad
            
            # - 2 * B^-1 * A^T (y_tilde - A J_tilde w)
            v = - 2 * precision * norm_grad

            if use_sigmoid:
                v = v * lin_recon * (1 - lin_recon)

            optimizer.zero_grad()
            
            # L(w) = (1/2) * ||y - A J_tilde w||^2_{B^-1} + 1/2 * ||w||^2_{Sigma_theta^-1}
            
            # - (2 * B^-1) * J_tilde^T A^T (y_tilde - A J_tilde w) + (2 * Sigma_theta^-1) * w
            
            # w_exact = (B^-1 * J_tilde^T A^T A J_tilde + Sigma_theta^-1)^-1 * (B^-1 * J_tilde^T A^T y_tilde)
            
            grads_vec = neural_basis_expansion.vjp(v.view(1, 1, 1, *trafo.im_shape)).squeeze(dim=0)
            lin_weights_fd.grad = grads_vec + optim_kwargs['wd'] * lin_weights_fd.detach()
            
            # print(lin_weights_fd.grad)
            # lin_weights_fd.grad = torch.clip(lin_weights_fd.grad, min=-1.0, max=1.0)
            optimizer.step()

            pbar.set_description(
                    f'l2_norm={lin_weights_fd.pow(2).sum():.3f}',
                    refresh=False)

    return lin_weights_fd.detach(), lin_recon.detach()
