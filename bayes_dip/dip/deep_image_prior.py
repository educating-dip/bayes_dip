import os
import socket
import datetime
from warnings import warn 
from copy import deepcopy
from contextlib import nullcontext
import torch
import numpy as np
import tensorboardX
from torch import Tensor
from torch.nn import MSELoss
from tqdm import tqdm
from bayes_dip.utils import get_original_cwd
from bayes_dip.utils import tv_loss, PSNR, normalize
from bayes_dip.data import BaseRayTrafo
from .network import UNet

class DeepImagePriorReconstructor():
    """
    CT reconstructor applying DIP with TV regularization (see [2]_).
    The DIP was introduced in [1].
    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018, "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           https://doi.org/10.1109/CVPR.2018.00984
    .. [2] D. Otero Baguer, J. Leuschner, M. Schmidt, 2020, "Computed
           Tomography Reconstruction Using Deep Image Prior and Learned
           Reconstruction Methods". Inverse Problems.
           https://doi.org/10.1088/1361-6420/aba415
    """

    def __init__(self, 
            ray_trafo: BaseRayTrafo,
            torch_manual_seed: int = 1, 
            device=None, 
            net_kwargs=None):

        self.device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.ray_trafo = ray_trafo.to(self.device)
        self.net_kwargs = net_kwargs
        self.init_model(torch_manual_seed)
        self.net_input = None
        self.optimizer = None

    def init_model(self,
            torch_manual_seed: int):

        with torch.random.fork_rng([self.device]) if torch_manual_seed else nullcontext():
            if torch_manual_seed:
                torch.random.manual_seed(torch_manual_seed)

            self.model = UNet(
                in_ch=1,
                out_ch=1,
                channels=self.net_kwargs['channels'][:self.net_kwargs['scales']],
                skip_channels=self.net_kwargs['skip_channels'][:self.net_kwargs['scales']],
                use_sigmoid=self.net_kwargs['use_sigmoid'],
                use_norm=self.net_kwargs['use_norm'],
                sigmoid_saturation_thresh= self.net_kwargs['sigmoid_saturation_thresh']
                ).to(self.device)

    def load_pretrain_model(self, 
            learned_params_path: str):

        path = os.path.join(
            get_original_cwd(),
            learned_params_path if learned_params_path.endswith('.pt') \
                else learned_params_path + '.pt')
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def reconstruct(self, 
            noisy_observation: Tensor, 
            fbp: Tensor = None, 
            ground_truth: Tensor = None, 
            recon_from_randn: bool = False,
            use_tv_loss: bool = True,
            log_path: str = '.',
            show_pbar: bool = True,
            optim_kwargs=None) -> Tensor:

        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        comment = 'DIP' if not use_tv_loss else 'DIP+TV'
        logdir = os.path.join(
            log_path,
            current_time + '_' + socket.gethostname() + comment)
        writer = tensorboardX.SummaryWriter(logdir=logdir)
        
        self.model.to(self.device)
        self.model.train()

        if recon_from_randn:
            self.net_input = 0.1 * \
                torch.randn(1, 1, *self.ray_trafo.im_shape, device=self.device)
        else:
            self.net_input = fbp.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optim_kwargs['lr'])        
        y_delta = noisy_observation.to(self.device)
        if optim_kwargs['loss_function'] == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        best_loss = np.inf
        best_output = self.model(self.net_input).detach()
        best_params_state_dict = deepcopy(self.model.state_dict())

        with tqdm(range(optim_kwargs['iterations']), desc='DIP', disable= not show_pbar, miniters=optim_kwargs['iterations']//100) as pbar:
            for i in pbar:
                self.optimizer.zero_grad()
                output = self.model(self.net_input)
                loss = criterion(self.ray_trafo(output), y_delta) 
                if use_tv_loss: 
                    loss = loss + optim_kwargs['gamma'] * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                if loss.item() < best_loss:
                    best_params_state_dict = deepcopy(self.model.state_dict())
                self.optimizer.step()

                for p in self.model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_output = output.detach()

                if ground_truth is not None:
                    best_output_psnr = PSNR(best_output.detach().cpu(), ground_truth.cpu())
                    output_psnr = PSNR(output.detach().cpu(), ground_truth.cpu())
                    pbar.set_description(f'DIP output_psnr={output_psnr:.1f}', refresh=False)
                    writer.add_scalar('best_output_psnr', best_output_psnr, i)
                    writer.add_scalar('output_psnr', output_psnr, i)

                writer.add_scalar('loss', loss.item(),  i)
                if i % 1000 == 0:
                    writer.add_image('reco', normalize(best_output[0, ...]).cpu().numpy(), i)

        self.model.load_state_dict(best_params_state_dict)
        writer.close()

        return best_output
