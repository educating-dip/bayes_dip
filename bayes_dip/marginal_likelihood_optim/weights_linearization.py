import torch
from torch import nn
from tqdm import tqdm
from bayes_dip.dip import UNetReturnPreSigmoid
from bayes_dip.probabilistic_models import NeuralBasisExpansion, GpriorNeuralBasisExpansion
from ..utils import batch_tv_grad, PSNR, eval_mode  # pylint: disable=unused-import

def weights_linearization(
        trafo, neural_basis_expansion, map_weights, observation, ground_truth, optim_kwargs):
    # pylint: disable=too-many-locals

    nn_model = neural_basis_expansion.nn_model
    nn_input = neural_basis_expansion.nn_input

    if nn_model.use_sigmoid:
        nn_model_no_sigmoid = UNetReturnPreSigmoid(nn_model)
        neural_basis_expansion = NeuralBasisExpansion(
                nn_model=nn_model_no_sigmoid,
                nn_input=nn_input,
                ordered_nn_params=neural_basis_expansion.ordered_nn_params,
                nn_out_shape=nn_input.shape,
        )
        if optim_kwargs['use_gprior']: 
            neural_basis_expansion = GpriorNeuralBasisExpansion(
                neural_basis_expansion=neural_basis_expansion,
                trafo=trafo,
                scale_kwargs=optim_kwargs['gprior_scale_kwargs'],
                device=observation.device,
            )
    else:
        nn_model_no_sigmoid = nn_model

    with torch.no_grad():
        recon_no_activation = nn_model_no_sigmoid(nn_input, saturation_safety=True)

    lin_weights_fd = (
            nn.Parameter(torch.zeros_like(map_weights)) if optim_kwargs['simplified_eqn']
            else map_weights.clone())
    optimizer = torch.optim.Adam([lin_weights_fd], lr=optim_kwargs['lr'], weight_decay=0)

    precision = optim_kwargs['noise_precision']

    with tqdm(range(optim_kwargs['iterations']),
                miniters=optim_kwargs['iterations']//100) as pbar, \
            eval_mode(nn_model_no_sigmoid):
        for _ in pbar:

            if optim_kwargs['simplified_eqn']:
                fd_vector = lin_weights_fd
            else:
                fd_vector = lin_weights_fd - map_weights

            lin_recon = neural_basis_expansion.jvp(fd_vector[None, :]).detach().squeeze(dim=1)

            if not optim_kwargs['simplified_eqn']:
                lin_recon = lin_recon + recon_no_activation

            if nn_model.use_sigmoid:
                lin_recon = lin_recon.sigmoid()

            proj_lin_recon = trafo(lin_recon)

            observation = observation.view(*proj_lin_recon.shape)
            norm_grad = trafo.trafo_adjoint( observation - proj_lin_recon )
            tv_grad = batch_tv_grad(lin_recon)

            # loss = (torch.nn.functional.mse_loss(
            #                 proj_lin_recon, observation.view(*proj_lin_recon.shape))
            #         + optim_kwargs['gamma'] * tv_loss(lin_recon))
            v = - 2 / observation.numel() * precision * norm_grad + optim_kwargs['gamma'] * tv_grad

            if nn_model.use_sigmoid:
                v = v * lin_recon * (1 - lin_recon)

            optimizer.zero_grad()

            grads_vec = neural_basis_expansion.vjp(v.view(1, 1, 1, *trafo.im_shape)).squeeze(dim=0)
            lin_weights_fd.grad = grads_vec + optim_kwargs['wd'] * lin_weights_fd.detach()
            optimizer.step()

            pbar.set_description(
                    f'psnr={PSNR(lin_recon.detach().cpu().numpy(),ground_truth.cpu().numpy()):.1f}',
                    refresh=False)

    return lin_weights_fd.detach(), lin_recon.detach()
