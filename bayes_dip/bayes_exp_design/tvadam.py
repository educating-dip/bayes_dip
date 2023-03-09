import datetime
import socket
import os
import torch
import numpy as np
import tensorboardX

from warnings import warn
from torch.optim import Adam
from torch.nn import MSELoss
from tqdm import tqdm
from bayes_dip.utils import tv_loss, PSNR


def show_image(outputname, data, cmap, clim=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,5))
    im = ax.imshow(data, cmap=cmap)
    ax.axis('off')
    if clim is not None:
        im.set_clim(*clim)
    plt.savefig(outputname + '.png', bbox_inches='tight', pad_inches=0.0)

class TVAdamReconstructor:

    """
    Reconstructor minimizing a TV-functional with the Adam optimizer.
    """

    def __init__(self, ray_trafo_module):

        self.device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.forward_op_module = ray_trafo_module.to(self.device)

    def reconstruct(self, observation, filtbackproj, ground_truth=None, log=False, optim_kwargs={}, **kwargs):

        torch.random.manual_seed(10)
        self.output = filtbackproj.clone().detach().to(self.device)
        self.output.requires_grad = True
        self.model = torch.nn.Identity()
        self.optimizer = Adam([self.output], lr=optim_kwargs['lr'])
        y_delta = observation.clone().detach().to(self.device)

        if optim_kwargs['loss_function'] == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        if log:
            current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
            comment = 'TVAdam'
            logdir = os.path.join(
                optim_kwargs['log_path'],
                current_time + '_' + socket.gethostname() + comment)
            self.writer = tensorboardX.SummaryWriter(logdir=logdir)

        best_loss = np.infty
        best_output = self.model(self.output).clone().detach()
        with tqdm(range(optim_kwargs['iterations']), desc='TV', disable=not optim_kwargs['show_pbar']) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.model(self.output)
                loss = criterion(self.forward_op_module(output),
                                 y_delta) + optim_kwargs['gamma'] * tv_loss(output)
                loss.backward()
                self.optimizer.step()
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    pbar.set_postfix({'best_loss': best_loss})
                    best_output = torch.nn.functional.relu(output.detach()) if optim_kwargs['use_relu_out'] else output.detach()

                if log and ground_truth is not None:
                    best_output_psnr = PSNR(best_output.detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(
                        (torch.nn.functional.relu(output.detach()) if optim_kwargs['use_relu_out'] else output.detach()).cpu(), ground_truth.cpu())
                    pbar.set_postfix({'output_psnr': output_psnr})
                    self.writer.add_scalar('best_output_psnr', best_output_psnr, i)
                    self.writer.add_scalar('output_psnr', output_psnr, i)

        if log:
            self.writer.close()

        return best_output[0, 0, ...]
