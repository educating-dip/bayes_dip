"""
Provides :class:`ParamFanBeam2DRayTrafo`.

The implementation is based on
https://github.com/jmaces/aapm-ct-challenge/blob/81cdf946438ea2227eb4007f9e3ac4e546d4b0c4/aapm-ct/operators.py#L211.
"""

from typing import Tuple
from math import ceil
import torch
from torch import Tensor
import numpy as np
from .base_ray_trafo import BaseRayTrafo
from .padded_ray_trafo import PaddedRayTrafo


def fft1(x):
    """ 1-dimensional centered Fast Fourier Transform. """
    x = ifftshift(x, dim=(-1,))
    x = torch.fft.fft(x, norm='ortho')
    x = fftshift(x, dim=(-1,))
    return x


def ifft1(x):
    """ 1-dimensional centered Inverse Fast Fourier Transform. """
    x = ifftshift(x, dim=(-1,))
    x = torch.fft.ifft(x, norm='ortho')
    x = fftshift(x, dim=(-1,))
    return x


def roll(x, shift, dim):
    """ np.roll for torch tensors. """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """ np.fft.fftshift for torch tensors. """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [xdim // 2 for xdim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """ np.fft.ifftshift for torch tensors. """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(xdim + 1) // 2 for xdim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


class ParamFanBeam2DRayTrafo(BaseRayTrafo):
    """
    Fan beam ray transform with parameters supporting autograd (e.g. grad w.r.t. angles).

    The implementation is based on
    https://github.com/jmaces/aapm-ct-challenge/blob/81cdf946438ea2227eb4007f9e3ac4e546d4b0c4/aapm-ct/operators.py#L211.
    """

    def __init__(self,
            im_shape: Tuple[int, int],
            obs_shape: Tuple[int, int],
            angles: Tensor,  # in rad
            scale: Tensor,
            d_source: Tensor,
            s_detect: float = 1.,  # we keep this parameter fixed
            flat: bool = True,
            filter_type: str = 'hamming',
            learn_inv_scale: bool = False,
            ):

        super().__init__(im_shape=im_shape, obs_shape=obs_shape)

        # alias names to stay close to the original implementation
        self.n = im_shape
        self.m = obs_shape
        self.n_detect = self.obs_shape[1]

        self._angles = torch.nn.Parameter(
            angles, requires_grad=angles.requires_grad
        )
        self.scale = torch.nn.Parameter(
            scale, requires_grad=scale.requires_grad
        )
        self.d_source = torch.nn.Parameter(
            d_source, requires_grad=d_source.requires_grad
        )
        self.s_detect = s_detect
        self.flat = flat
        self.filter_type = filter_type
        self.inv_scale = torch.nn.Parameter(
            torch.tensor(1.0), requires_grad=learn_inv_scale
        )

    @property
    def angles(self) -> np.ndarray:
        """:class:`np.ndarray` : The angles (in radian)."""
        return self._angles.detach().cpu().numpy()

    def _d_detect(self):
        if self.flat:
            return (
                abs(self.s_detect)
                * self.n_detect
                / self.n[0]
                * torch.sqrt(
                    self.d_source * self.d_source
                    - (self.n[0] / 2.0) * (self.n[0] / 2.0)
                )
                - self.d_source
            )
        else:
            return (
                abs(self.s_detect)
                * (self.n_detect / 2.0)
                / torch.asin((self.n[0] / 2.0) / self.d_source)
                - self.d_source
            )

    def trafo(self, x: Tensor) -> Tensor:
        return self.dot(x)

    def dot(self, x: Tensor) -> Tensor:
        # detector positions
        device = self._angles.device

        s_range = (
            torch.arange(self.n_detect, device=device).unsqueeze(
                0
            )
            - self.n_detect / 2.0
            + 0.5
        ) * self.s_detect
        if self.flat:
            p_detect_x = s_range
            p_detect_y = -self._d_detect()
        else:
            gamma = s_range / (self.d_source + self._d_detect())
            p_detect_x = (self.d_source + self._d_detect()) * torch.sin(gamma)
            p_detect_y = self.d_source - (
                self.d_source + self._d_detect()
            ) * torch.cos(gamma)

        # source position
        p_source_x = 0.0
        p_source_y = self.d_source

        # rotate rays from source to detector over all angles
        pi = torch.acos(torch.zeros(1)).item() * 2.0
        cs = torch.cos(self._angles).unsqueeze(1)
        sn = torch.sin(self._angles).unsqueeze(1)
        r_p_source_x = p_source_x * cs - p_source_y * sn
        r_p_source_y = p_source_x * sn + p_source_y * cs
        r_dir_x = p_detect_x * cs - p_detect_y * sn - r_p_source_x
        r_dir_y = p_detect_x * sn + p_detect_y * cs - r_p_source_y

        # find intersections of rays with circle for clipping
        if self.flat:
            max_gamma = torch.atan(
                (self.s_detect * (self.n_detect / 2.0))
                / (self.d_source + self._d_detect())
            )
        else:
            max_gamma = (self.s_detect * (self.n_detect / 2.0)) / (
                self.d_source + self._d_detect()
            )
        radius = self.d_source * torch.sin(max_gamma)
        a = r_dir_x * r_dir_x + r_dir_y * r_dir_y
        b = r_p_source_x * r_dir_x + r_p_source_y * r_dir_y
        c = (
            r_p_source_x * r_p_source_x
            + r_p_source_y * r_p_source_y
            - radius * radius
        )
        ray_length_threshold = 1.0
        discriminant_sqrt = torch.sqrt(
            torch.max(
                b * b - a * c,
                torch.tensor(ray_length_threshold, device=device),
            )
        )
        lambda_1 = (-b - discriminant_sqrt) / a
        lambda_2 = (-b + discriminant_sqrt) / a

        # clip ray accordingly
        r_p_source_x = r_p_source_x + lambda_1 * r_dir_x
        r_p_source_y = r_p_source_y + lambda_1 * r_dir_y
        r_dir_x = r_dir_x * (lambda_2 - lambda_1)
        r_dir_y = r_dir_y * (lambda_2 - lambda_1)

        # use batch and channel dimensions for vectorized interpolation
        original_dim = x.ndim
        while x.ndim < 4:
            x = x.unsqueeze(0)
        assert x.shape[-3] == 1  # we can handle only single channel data
        x = x.transpose(-4, -3)  # switch batch and channel dim

        # integrate over ray
        num_steps = torch.ceil(
            torch.sqrt(r_dir_x * r_dir_x + r_dir_y * r_dir_y)
        ).max()
        diff_x = r_dir_x / num_steps
        diff_y = r_dir_y / num_steps
        steps = (
            torch.arange(
                int(num_steps.detach().cpu().numpy()), device=device
            )
            .unsqueeze(1)
            .unsqueeze(1)
        )
        grid_x = r_p_source_x.unsqueeze(0) + steps * diff_x.unsqueeze(0)
        grid_y = r_p_source_y.unsqueeze(0) + steps * diff_y.unsqueeze(0)

        grid_x = grid_x / (
            self.n[0] / 2.0 - 0.5
        )  # rescale image positions to [-1, 1]
        grid_y = grid_y / (
            self.n[1] / 2.0 - 0.5
        )  # rescale image positions to [-1, 1]
        grid = torch.stack([grid_y, grid_x], dim=-1)
        inter = torch.nn.functional.grid_sample(
            x.expand((int(num_steps.detach().cpu().numpy()), -1, -1, -1)),
            grid,
            align_corners=True,
        )

        sino = inter.sum(dim=0, keepdim=True) * torch.sqrt(
            diff_x * diff_x + diff_y * diff_y
        ).unsqueeze(0).unsqueeze(0)

        # undo batch and channel manipulations
        sino = sino.transpose(-4, -3)  # unswitch batch and channel dim
        while sino.ndim > original_dim:
            sino = sino.squeeze(0)

        return sino * self.scale

    def trafo_adjoint(self, observation: Tensor) -> Tensor:
        # Unfiltered back projection (approx. adjoint).
        observation = self._reweight_sinogram(observation * self.scale)  # TODO check whether this should be included in the adjoint
        return self._adj(observation)  # * self.scale

    def _adj(self, sino):
        """ Basic back projection without filtering or pre weighting. """
        device = self._angles.device

        # image coordinate grid
        p_x = torch.linspace(
            -self.n[0] / 2.0 + 0.5,
            self.n[0] / 2.0 - 0.5,
            self.n[0],
            device=device,
        ).unsqueeze(1)
        p_y = torch.linspace(
            -self.n[1] / 2.0 + 0.5,
            self.n[1] / 2.0 - 0.5,
            self.n[1],
            device=device,
        ).unsqueeze(0)

        # check if coordinate is within circle
        if self.flat:
            max_gamma = torch.atan(
                (self.s_detect * (self.n_detect / 2.0))
                / (self.d_source + self._d_detect())
            )
        else:
            max_gamma = (self.s_detect * (self.n_detect / 2.0)) / (
                self.d_source + self._d_detect()
            )
        radius = self.d_source * torch.sin(max_gamma)
        p_r = torch.sqrt(p_x * p_x + p_y * p_y)
        mask = p_r <= radius

        # use batch and channel dimensions for vectorized interpolation
        original_dim = sino.ndim
        while sino.ndim < 4:
            sino = sino.unsqueeze(0)
        assert sino.shape[-3] == 1  # we can handle only single channel data
        sino = sino.transpose(-4, -3)  # switch batch and channel dim

        # rotated coordinate grid
        pi = torch.acos(torch.zeros(1)).item() * 2.0
        cs = torch.cos(self._angles).unsqueeze(1).unsqueeze(1)
        sn = torch.sin(self._angles).unsqueeze(1).unsqueeze(1)
        p_x_r = cs * p_x + sn * p_y
        p_y_r = -sn * p_x + cs * p_y

        # find angles and detector positions defining rays through coordinate
        if self.flat:
            grid_d = (
                (self.d_source + self._d_detect())
                * p_x_r
                / (self.d_source - p_y_r)
            )
        else:
            grid_d = (self.d_source + self._d_detect()) * torch.atan(
                p_x_r / (self.d_source - p_y_r)
            )
        grid_a = (
            torch.arange(self.m[0], device=device)
            .unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, self.n[0], self.n[1])
            - self.m[0] / 2.0
            + 0.5
        )

        grid_d = grid_d / (
            (self.n_detect / 2.0 - 0.5) * self.s_detect
        )  # rescale valid detector positions to [-1,1]
        grid_a = grid_a / (self.m[0] / 2.0 - 0.5)  # rescale angles to [-1,1]
        grid = torch.stack([grid_d, grid_a], dim=-1)
        inter = torch.nn.functional.grid_sample(
            sino.expand(self.m[0], -1, -1, -1), grid, align_corners=True
        )

        # compute integral reweighting factors and integrate
        if self.flat:
            weight = (self.d_source + self._d_detect()).pow(2) / (
                self.d_source - p_y_r
            ).pow(2)
        else:
            weight = (self.d_source + self._d_detect()).pow(2) / (
                (self.d_source - p_y_r).pow(2) + p_x_r.pow(2)
            )
        x = mask * (inter * (weight).unsqueeze(1)).sum(dim=0, keepdim=True)

        # undo batch and channel manipulations
        x = x.transpose(-4, -3)  # unswitch batch and channel dim
        while x.ndim > original_dim:
            x = x.squeeze(0)

        return x / self.s_detect

    def _reweight_sinogram(self, sino: Tensor) -> Tensor:
        """ Reweight sinogram contributions to back projections. """
        return sino * self._get_pre_weight()

    def _filter_sinogram(self, sino: Tensor) -> Tensor:
        """ Pad and filter sinogram. """
        device = self._angles.device

        # pad sinogram to reduce periodicity artefacts
        target_size = max(
            64, int(2 ** np.ceil(np.log2(2 * self.m[-1])))
        )
        pad = target_size - self.m[-1]
        sino_pad = torch.nn.functional.pad(sino, (pad // 2, pad - pad // 2))

        # fft along detector direction
        sino_fft = fft1(sino_pad)

        # apply frequency filter
        f = self._get_fourier_filter().to(device)
        filtered_sino_fft = sino_fft * f

        # ifft along detector direction
        filtered_sino_pad = ifft1(filtered_sino_fft).real

        # remove padding and rescale
        filtered_sino = filtered_sino_pad[..., pad // 2 : -(pad - pad // 2)]

        return filtered_sino

    def fbp(self, observation: Tensor) -> Tensor:
        # Filtered back projection (FBP).
        observation = self._reweight_sinogram(observation / self.scale)
        observation = self._filter_sinogram(observation)
        return self.inv_scale * self._adj(observation)  # / self.scale

    trafo_flat = BaseRayTrafo._trafo_flat_via_trafo
    trafo_adjoint_flat = BaseRayTrafo._trafo_adjoint_flat_via_trafo_adjoint

    def _get_fourier_filter(self):
        device = self._angles.device

        """ Ramp Fourier filter for the FBP. """
        size = max(64, int(2 ** np.ceil(np.log2(2 * self.m[-1]))))

        pi = torch.acos(torch.zeros(1)).item() * 2.0
        n = torch.cat(
            [
                torch.arange(1, size // 2 + 1, 2, device=device),
                torch.arange(size // 2 - 1, 0, -2, device=device),
            ]
        )
        f = torch.zeros(size, device=device)
        f[0] = 0.25
        if self.flat:
            f[1::2] = -1 / (pi * n).pow(2)
        else:
            f[1::2] = -self.s_detect.pow(2) / (
                pi
                * (self.d_source + self._d_detect())
                * torch.sin(
                    n
                    * self.s_detect
                    / (self.d_source + self._d_detect())
                )
            ).pow(2)
        f = fftshift(f, dim=(-1,))

        filt = fft1(f).real

        if self.filter_type == "hamming":
            # hamming filter
            fac = torch.tensor(
                np.hamming(size).astype(np.float32), device=device
            )
        elif self.filter_type == "hann":
            # hann filter
            fac = torch.tensor(
                np.hanning(size).astype(np.float32), device=device
            )
        elif self.filter_type == "cosine":
            # cosine filter
            fac = torch.sin(
                torch.linspace(0, pi, size + 1, device=device)[:-1]
            )
        else:
            # ramp / ram-lak filter
            fac = 1.0

        return fac * filt

    def _get_pre_weight(self):
        """ Pre filtering weighting for back projections. """
        device = self._angles.device

        s_range = (
            torch.arange(self.n_detect, device=device).unsqueeze(
                0
            )
            - self.n_detect / 2.0
            + 0.5
        ) * self.s_detect
        if self.flat:
            weight = self.d_source / torch.sqrt(
                (self.d_source + self._d_detect()).pow(2) + s_range.pow(2)
            )
        else:
            weight = (
                self.d_source
                / (self.d_source + self._d_detect())
                * torch.cos(s_range / (self.d_source + self._d_detect()))
            )
        return weight

def get_param_fan_beam_2d_ray_trafo(
        im_shape: Tuple[int, int],
        num_angles: int,
        num_det_pixels: int,
        src_radius: float,
        angular_sub_sampling: int = 1,
        filter_type: str = 'hamming'):

    angles = torch.from_numpy(np.linspace(
            0., 2.*np.pi, num=num_angles, endpoint=False, dtype=np.float32
            )[::angular_sub_sampling])
    ray_trafo = ParamFanBeam2DRayTrafo(
            im_shape=im_shape,
            obs_shape=(num_angles, num_det_pixels),
            angles=angles,
            scale=torch.tensor(1.),
            d_source=torch.tensor(src_radius),
            filter_type=filter_type)
    
    return ray_trafo

def get_rect_padded_param_fan_beam_2d_ray_trafo(
        im_shape: Tuple[int, int],
        num_angles: int,
        num_det_pixels: int,
        src_radius: float,
        angular_sub_sampling: int = 1,
        filter_type: str = 'hamming'):

    angles = torch.from_numpy(np.linspace(
            0., 2.*np.pi, num=num_angles, endpoint=False, dtype=np.float32
            )[::angular_sub_sampling])

    halfpad = (ceil(im_shape[0] / 2 * (np.sqrt(2.) - 1.)), ceil(im_shape[1] / 2 * (np.sqrt(2.) - 1.)))
    padded_im_shape = (im_shape[0] + 2 * halfpad[0], im_shape[1] + 2 * halfpad[1])

    ray_trafo = ParamFanBeam2DRayTrafo(
            im_shape=padded_im_shape,
            obs_shape=(num_angles, num_det_pixels),
            angles=angles,
            scale=torch.tensor(1.),
            d_source=torch.tensor(src_radius),
            filter_type=filter_type)
    
    padded_ray_trafo = PaddedRayTrafo(im_shape=im_shape, ray_trafo=ray_trafo)

    return padded_ray_trafo
