"""
Provides :class:`DeepImagePriorReconstructor`.
"""
import os
import socket
from typing import Optional, Union
import datetime
from warnings import warn
from copy import deepcopy
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
    The DIP was introduced in [1]_.

    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018, "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           https://doi.org/10.1109/CVPR.2018.00984
    .. [2] D. Otero Baguer, J. Leuschner, and M. Schmidt, 2020, "Computed
           Tomography Reconstruction Using Deep Image Prior and Learned
           Reconstruction Methods". Inverse Problems.
           https://doi.org/10.1088/1361-6420/aba415
    """

    def __init__(self,
            ray_trafo: BaseRayTrafo,
            torch_manual_seed: Union[int, None] = 1,
            device=None,
            net_kwargs=None,
            load_params_path: Optional[str] = None):
        """
        Parameters
        ----------
        ray_trafo : :class:`bayes_dip.data.BaseRayTrafo`
            Ray transform.
        torch_manual_seed : int or None, optional
            Random number generator seed, used for initializing the network.
            If ``None``, no seed is set and the global random generator is advanced;
            otherwise, the manual seed is set on a forked generator used for the initialization.
            The default is ``1``.
        device : str or torch.device, optional
            Device for the reconstruction.
            If ``None`` (the default), ``'cuda:0'`` is chosen if available or ``'cpu'`` otherwise.
        net_kwargs : dict, optional
            Network architecture keyword arguments.
        load_params_path : str, optional
            If specified, load the specified parameters instead of random initialization.
        """

        self.device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.ray_trafo = ray_trafo.to(self.device)
        self.net_kwargs = net_kwargs
        self.init_nn_model(torch_manual_seed)
        if load_params_path is not None:
            self.load_params(load_params_path)
        self.net_input = None
        self.optimizer = None

    def init_nn_model(self,
            torch_manual_seed: Union[int, None]):
        """
        Initialize the network :attr:`nn_model`.

        Parameters
        ----------
        torch_manual_seed : int or None
            Random number generator seed.
            If ``None``, no seed is set and the global random generator is advanced;
            otherwise, the manual seed is set on a forked generator used for the initialization.
        """
        fork_rng_kwargs = {'enabled': torch_manual_seed is not None}
        if self.device != 'cpu':
            fork_rng_kwargs['devices'] = [self.device]
        with torch.random.fork_rng(**fork_rng_kwargs):
            if torch_manual_seed is not None:
                torch.random.manual_seed(torch_manual_seed)

            self.nn_model = UNet(
                in_ch=1,
                out_ch=1,
                channels=self.net_kwargs['channels'][:self.net_kwargs['scales']],
                skip_channels=self.net_kwargs['skip_channels'][:self.net_kwargs['scales']],
                use_sigmoid=self.net_kwargs['use_sigmoid'],
                use_norm=self.net_kwargs['use_norm'],
                sigmoid_saturation_thresh=self.net_kwargs['sigmoid_saturation_thresh']
                ).to(self.device)

    def load_params(self,
            params_path: str):
        """
        Load model state dict from file.

        Parameters
        ----------
        params_path : str
            Path to the parameters, either absolute or relative to the original
            current working directory.
        """

        path = os.path.join(
            get_original_cwd(),
            params_path if params_path.endswith('.pt') \
                else params_path + '.pt')
        self.nn_model.load_state_dict(torch.load(path, map_location=self.device))

    def reconstruct(self,
            noisy_observation: Tensor,
            filtbackproj: Optional[Tensor] = None,
            ground_truth: Optional[Tensor] = None,
            recon_from_randn: bool = False,
            use_tv_loss: bool = True,
            log_path: str = '.',
            show_pbar: bool = True,
            optim_kwargs=None) -> Tensor:
        """
        Reconstruct (by "training" the DIP network).

        Parameters
        ----------
        noisy_observation : Tensor
            Noisy observation. Shape: ``(1, 1, *self.ray_trafo.obs_shape)``.
        filtbackproj : Tensor, optional
            Filtered back-projection. Used as the network input if ``not recon_from_randn``.
            Shape: ``(1, 1, *self.ray_trafo.im_shape)``
        ground_truth : Tensor, optional
            Ground truth. Used to print and log PSNR values.
            Shape: ``(1, 1, *self.ray_trafo.im_shape)``
        recon_from_randn : bool, optional
            If ``True``, normal distributed noise with std-dev 0.1 is used as the network input;
            if ``False`` (the default), ``filtbackproj`` is used as the network input.
        use_tv_loss : bool, optional
            Whether to include the TV loss term.
            The default is ``True``.
        log_path : str, optional
            Path for saving tensorboard logs. Each call to reconstruct creates a sub-folder
            in ``log_path``, starting with the time of the reconstruction call.
            The default is ``'.'``.
        show_pbar : bool, optional
            Whether to show a progress bar.
            The default is ``True``.
        optim_kwargs : dict, optional
            Keyword arguments for optimization.
            The arguments are:

            ``'gamma'`` : float
                Weighting factor of the TV loss term, the default is ``1e-4``.
            ``'lr'`` : float
                Learning rate, the default is ``1e-4``.
            ``'iterations'`` : int
                Number of iterations, the default is ``10000``.
            ``'loss_function'`` : str
                Discrepancy loss function, the default is ``'mse'``.

        Returns
        -------
        best_output : Tensor
            Model output with the minimum loss achieved during the training.
            Shape: ``(1, 1, *self.ray_trafo.im_shape)``.
        """

        writer = tensorboardX.SummaryWriter(
                logdir=os.path.join(log_path, '_'.join((
                        datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
                        socket.gethostname(),
                        'DIP' if not use_tv_loss else 'DIP+TV'))))

        optim_kwargs = optim_kwargs or {}
        optim_kwargs.setdefault('gamma', 1e-4)
        optim_kwargs.setdefault('lr', 1e-4)
        optim_kwargs.setdefault('iterations', 10000)
        optim_kwargs.setdefault('loss_function', 'mse')

        self.nn_model.train()

        self.net_input = (
            0.1 * torch.randn(1, 1, *self.ray_trafo.im_shape, device=self.device)
            if recon_from_randn else
            filtbackproj.to(self.device))

        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=optim_kwargs['lr'])
        noisy_observation = noisy_observation.to(self.device)
        if optim_kwargs['loss_function'] == 'mse':
            criterion = MSELoss()
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        min_loss_state = {
            'loss': np.inf,
            'output': self.nn_model(self.net_input).detach(),  # pylint: disable=not-callable
            'params_state_dict': deepcopy(self.nn_model.state_dict()),
        }

        with tqdm(range(optim_kwargs['iterations']), desc='DIP', disable=not show_pbar,
                miniters=optim_kwargs['iterations']//100) as pbar:

            for i in pbar:
                self.optimizer.zero_grad()
                output = self.nn_model(self.net_input)  # pylint: disable=not-callable
                loss = criterion(self.ray_trafo(output), noisy_observation)
                if use_tv_loss:
                    loss = loss + optim_kwargs['gamma'] * tv_loss(output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nn_model.parameters(), max_norm=1)

                if loss.item() < min_loss_state['loss']:
                    min_loss_state['loss'] = loss.item()
                    min_loss_state['output'] = output.detach()
                    min_loss_state['params_state_dict'] = deepcopy(self.nn_model.state_dict())

                self.optimizer.step()

                for p in self.nn_model.parameters():
                    p.data.clamp_(-1000, 1000) # MIN,MAX

                if ground_truth is not None:
                    min_loss_output_psnr = PSNR(
                            min_loss_state['output'].cpu(), ground_truth.cpu())
                    output_psnr = PSNR(
                            output.detach().cpu(), ground_truth.cpu())
                    pbar.set_description(f'DIP output_psnr={output_psnr:.1f}', refresh=False)
                    writer.add_scalar('min_loss_output_psnr', min_loss_output_psnr, i)
                    writer.add_scalar('output_psnr', output_psnr, i)

                writer.add_scalar('loss', loss.item(),  i)
                if i % 1000 == 0:
                    writer.add_image('reco', normalize(
                            min_loss_state['output'][0, ...]).cpu().numpy(), i)

        self.nn_model.load_state_dict(min_loss_state['params_state_dict'])
        writer.close()

        return min_loss_state['output']
