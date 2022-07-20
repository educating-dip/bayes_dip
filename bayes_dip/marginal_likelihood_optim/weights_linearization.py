from calendar import c
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import functorch as ftch
from .utils import get_ordered_nn_params_vec
from ..probabilistic_models.linearized_dip.neural_basis_expansion import unflatten_nn_functorch
from ..utils import tv_loss, batch_tv_grad, PSNR 

def weights_linearization(observation_cov, observation, ground_truth, optim_kwargs):
    
    neural_basis_expansion = observation_cov.image_cov.neural_basis_expansion
    nn_input = neural_basis_expansion.nn_input
    forward_kwargs = {'return_pre_sigmoid': True}
    device = observation_cov.device

    with torch.no_grad(): 
        recon_no_activation = neural_basis_expansion.nn_model(
            nn_input, 
            **{'return_pre_sigmoid': True}
        )
        map_weights = torch.clone(
            get_ordered_nn_params_vec(observation_cov.image_cov.inner_cov)
        )
    
    lin_weights_fd = nn.Parameter(torch.zeros_like(map_weights, device=device))
    optimizer = torch.optim.Adam([lin_weights_fd], **{'lr': optim_kwargs['lr']}, weight_decay=0)
    
    precision = 1.
    neural_basis_expansion.nn_model.eval()

    neural_basis_expansion.func_forward_kwargs = forward_kwargs
    with tqdm(range(optim_kwargs['iterations']), miniters=optim_kwargs['iterations']//100) as pbar:
        for _ in pbar:

            if optim_kwargs['simplified_eqn']:
                fd_vector = lin_weights_fd
            else:
                fd_vector = lin_weights_fd - map_weights
            
            lin_recon = neural_basis_expansion.jvp(fd_vector[None, :]).detach()

            if not optim_kwargs['simplified_eqn']:
                lin_recon = lin_recon + recon_no_activation
            
            if optim_kwargs['use_sigmoid']:
                lin_recon = lin_recon.sigmoid()

            proj_lin_recon = observation_cov.trafo(lin_recon)

            observation = observation.view(*proj_lin_recon.shape)
            norm_grad = observation_cov.trafo.trafo_adjoint( observation - proj_lin_recon ).flatten()
            tv_grad = batch_tv_grad(lin_recon.squeeze(dim=0)).flatten() 

            loss = torch.nn.functional.mse_loss(proj_lin_recon, observation.view(*proj_lin_recon.shape)) + optim_kwargs['gamma'] * tv_loss(lin_recon)
            v = - 2 / observation.numel() * precision * norm_grad + optim_kwargs['gamma'] * tv_grad

            if optim_kwargs['use_sigmoid']:
                v = v * lin_recon.flatten() * (1 - lin_recon.flatten())
            
            optimizer.zero_grad()
            neural_basis_expansion.nn_model.zero_grad()
            to_grad = neural_basis_expansion.nn_model(nn_input, return_pre_sigmoid=True).flatten() * v
            to_grad.sum().backward()

            grads_vec = torch.cat([param.grad.flatten().detach() for param in neural_basis_expansion.ordered_nn_params])
            lin_weights_fd.grad = grads_vec + optim_kwargs['wd'] * lin_weights_fd.detach()
            optimizer.step()

            pbar.set_description(f'psnr={PSNR(lin_recon.detach().cpu().numpy(), ground_truth.cpu().numpy())}', refresh=False)
    
    neural_basis_expansion.func_forward_kwargs = {}
    return lin_weights_fd.detach(), lin_recon.detach()