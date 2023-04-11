"""
Provides simulation by applying a ray transform and white noise.
"""

from typing import Iterable, Optional, Sequence, Union, Iterator, Any, Tuple, Callable
import numpy as np
import torch
from torch import Tensor

from .trafo import BaseRayTrafo


def simulate(x: Tensor, ray_trafo: BaseRayTrafo, white_noise_rel_stddev: float,
        rng: Optional[np.random.Generator] = None):
    """
    Compute ``observation = ray_trafo(x)`` and add white noise with standard
    deviation ``white_noise_rel_stddev * mean(abs(observation))``.

    Parameters
    ----------
    x : :class:`torch.Tensor`
        Image, passed to ``ray_trafo``.
    ray_trafo : callable
        Function computing the noise-free observation.
    white_noise_rel_stddev : float
        Relative standard deviation of the noise that is added.
    rng : :class:`np.random.Generator`, optional
        Random number generator. If ``None`` (the default),
        a new generator ``np.random.default_rng()`` is used.
    """

    observation = ray_trafo(x)

    if rng is None:
        rng = np.random.default_rng()
    noise = torch.from_numpy(rng.normal(
            scale=white_noise_rel_stddev * torch.mean(torch.abs(observation)).item(),
            size=observation.shape)).to(
                    dtype=observation.dtype, device=observation.device)

    noisy_observation = observation + noise

    return noisy_observation


class SimulatedDataset(torch.utils.data.Dataset):
    """
    CT dataset simulated from provided ground truth images.

    Each item of this dataset is a tuple ``noisy_observation, x, filtbackproj``, where

        * ``noisy_observation = ray_trafo(x) + noise``
          (shape: ``(1,) + obs_shape``)
        * ``x`` is the ground truth (label)
          (shape: ``(1,) + im_shape``)
        * ``filtbackproj = FBP(noisy_observation)``
          (shape: ``(1,) + im_shape``)
    """

    def __init__(self,
            image_dataset: Union[Sequence[Tensor], Iterable[Tensor]],
            ray_trafo: BaseRayTrafo,
            white_noise_rel_stddev: float,
            use_fixed_seeds_starting_from: Optional[int] = 1,
            ray_trafo_full: Optional[BaseRayTrafo] = None, 
            subsampling_indices: Optional[Sequence] = None, 
            rng: Optional[np.random.Generator] = None,
            rng_full: Optional[np.random.Generator] = None,
            device: Optional[Any] = None):
        """
        Parameters
        ----------
        image_dataset : sequence or iterable
            Image data. The methods :meth:`__len__` and :meth:`__getitem__`
            directly use the respective functions of ``image_dataset`` and will
            fail if they are not supported. The method :meth:`__iter__` simply
            iterates over ``image_dataset`` and thus will only stop when
            ``image_dataset`` is exhausted.
        ray_trafo : :class:`bayes_dip.data.BaseRayTrafo`
            Ray trafo.
        white_noise_rel_stddev : float
            Relative standard deviation of the noise that is added.
        use_fixed_seeds_starting_from : int, optional
            If an int, the fixed random seed
            ``use_fixed_seeds_starting_from + idx`` is used for sample ``idx``.
            Must be ``None`` if a custom ``rng`` is used. If ``ray_trafo_full`` is specified, 
            ``use_fixed_seeds_starting_from + idx + 1000`` is used for full observation.
            The default is ``1``.
        ray_trafo_full : :class:`bayes_dip.data.BaseRayTrafo`, optional 
            Ray trafo operator for full oberservation data.
            If specified, `observation_full` and `filtbackproj_full` 
            are returned as well. 
        subsampling_indices : sequence, optinal 
            Indeces to subsampled observation. It is required if ``ray_trafo_full``
            is specified. 
        rng : :class:`np.random.Generator`, optional
            Custom random number generator used to simulate noise; it will be
            advanced every time an item is accessed.
            Cannot be combined with ``use_fixed_seeds_starting_from``.
            If both ``rng`` and ``use_fixed_seeds_starting_from`` are ``None``,
            a new generator ``np.random.default_rng()`` is used.
        rng_full : :class:`np.random.Generator`, optional
            Custom random number generator used to simulate noise for full observation 
            if ``ray_trafo_full`` is specified; it will be  advanced every time 
            an item is accessed. Cannot be combined with ``use_fixed_seeds_starting_from``.
            If both ``rng`` and ``use_fixed_seeds_starting_from`` are ``None``,
            a new generator ``np.random.default_rng()`` is used. 
        device : str or torch.device, optional
            If specified, data will be moved to the device. ``ray_trafo``
            (including ``ray_trafo.fbp``) must support tensors on the device.
        """
        super().__init__()

        self.image_dataset = image_dataset
        self.ray_trafo = ray_trafo
        self.white_noise_rel_stddev = white_noise_rel_stddev
        if rng is not None:
            assert use_fixed_seeds_starting_from is None, (
                    'must not use fixed seeds when passing a custom rng')
        self.rng = rng
        self.use_fixed_seeds_starting_from = use_fixed_seeds_starting_from
        
        self.ray_trafo_full = ray_trafo_full
        self.subsampling_indices = subsampling_indices
        if self.ray_trafo_full is not None: 
            assert self.subsampling_indices is not None 
            if not isinstance(self.subsampling_indices, tuple): 
                self.subsampling_indices = (self.subsampling_indices, )
            if rng_full is not None:
                assert use_fixed_seeds_starting_from is None, (
                        'must not use fixed seeds when passing a custom rng')
            self.rng_full = rng_full

        self.device = device

    def __len__(self) -> Union[int, float]:
        return len(self.image_dataset)

    def _generate_item(self, idx: int, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:

        if self.rng is None:
            seed = (self.use_fixed_seeds_starting_from + idx
                    if self.use_fixed_seeds_starting_from is not None else None)
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng

        if self.ray_trafo_full is not None:
            offset = 1000
            if self.rng_full is None:
                seed = (self.use_fixed_seeds_starting_from + idx + offset
                        if self.use_fixed_seeds_starting_from is not None else None)
                rng_full = np.random.default_rng(seed)
            else:
                rng_full = self.rng_full

        x = x.to(device=self.device)
        noisy_observation = simulate(x[None],
                ray_trafo=self.ray_trafo,
                white_noise_rel_stddev=self.white_noise_rel_stddev,
                rng=rng)[0].to(device=self.device)
        filtbackproj = self.ray_trafo.fbp(noisy_observation[None])[0].to(
                device=self.device)

        if self.ray_trafo_full is not None:

            noisy_observation_full = simulate(x[None],
                ray_trafo=self.ray_trafo_full,
                white_noise_rel_stddev=self.white_noise_rel_stddev,
                rng=rng_full)[0].to(device=self.device)
            noisy_observation_full[(slice(None), *self.subsampling_indices)] = noisy_observation
            filtbackproj_full = self.ray_trafo_full.fbp(
                noisy_observation_full[None])[0].to(device=self.device)

            return noisy_observation, x, filtbackproj, noisy_observation_full, filtbackproj_full

        return noisy_observation, x, filtbackproj

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        for idx, x in enumerate(self.image_dataset):
            yield self._generate_item(idx, x)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self._generate_item(idx, self.image_dataset[idx])
