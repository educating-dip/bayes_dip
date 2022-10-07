"""
Provides :class:`RectanglesDataset`.
"""

from typing import Iterable, Iterator, Union, Tuple
from itertools import repeat
import numpy as np
import torch
from torch import Tensor
from skimage.draw import polygon
from skimage.transform import downscale_local_mean

def _rect_coords(shape, a1, a2, x, y, rot):
    # convert [-1., 1.]^2 coordinates to [0., shape[0]] x [0., shape[1]]
    x, y = 0.5 * shape[0] * (x + 1.), 0.5 * shape[1] * (y + 1.)
    a1, a2 = 0.5 * shape[0] * a1, 0.5 * shape[1] * a2
    # rotate side vector [a1, a2] to rot_mat @ [a1, a2]
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    coord_diffs = np.array([
            rot_mat @ [a1, a2],
            rot_mat @ [a1, -a2],
            rot_mat @ [-a1, -a2],
            rot_mat @ [-a1, a2]])
    coords = np.array([x, y])[None, :] + coord_diffs
    return coords


def _rect_phantom(
        shape: Tuple[int, int],
        rects: Iterable[Tuple[float, float, float, float, float, float]],
        smooth_sr_fact: int = 8,
        blend_mode: str = 'add') -> np.ndarray:

    sr_shape = (shape[0] * smooth_sr_fact, shape[1] * smooth_sr_fact)
    img = np.zeros(sr_shape, dtype='float32')
    for rect in rects:
        v, a1, a2, x, y, rot = rect
        coords = _rect_coords(sr_shape, a1, a2, x, y, rot)
        p_rr, p_cc = polygon(coords[:, 1], coords[:, 0], shape=sr_shape)
        if blend_mode == 'add':
            img[p_rr, p_cc] += v
        elif blend_mode == 'set':
            img[p_rr, p_cc] = v
    if smooth_sr_fact != 1:
        img = downscale_local_mean(img, (smooth_sr_fact, smooth_sr_fact))
    return img


class RectanglesDataset(torch.utils.data.IterableDataset):
    """
    Dataset with images of multiple random rectangles.
    The images are normalized to have a value range of ``[0., 1.]`` with a
    background value of ``0.``. Each image has shape ``(1,) + shape``.
    """
    def __init__(self,
            shape: Tuple[int, int] = (128, 128),
            num_rects: int = 3,
            num_angle_modes: int = 1,
            angle_modes_sigma: float = 0.05,
            length: int = 32000,
            fixed_seed: int = 1,
            smooth_sr_fact: int = 8):
        """
        shape : 2-tuple of int, optional
            Image shape.
            The default is ``(128, 128)``.
        num_rects : int, optional
            Number of rectangles (overlayed additively).
            The default is ``3``.
        num_angle_modes : int, optional
            Number of Gaussian modes from which angles can be sampled.
            For each rectangle, one of the modes is selected (with equal
            probability for each mode).
            The default is ``1``.
        angle_modes_sigma : float, optional
            Scale of each Gaussian mode.
            The default is ``0.05``.
        length : int, optional
            Number of images in the dataset.
            The default is ``32000``.
        fixed_seed : int, optional
            Fixed random seed.
            The default is ``1``.
        smooth_sr_fact : int, optional
            Super-resolution factor for the image generation.
            A higher factor leads to smoother edges (if not aligned with the
            pixel grid).
            The default is ``8``.
        """

        super().__init__()
        self.shape = shape
        # defining discretization space ODL
        self.num_rects = num_rects
        self.num_angle_modes = num_angle_modes or num_rects
        self.angle_modes_sigma = angle_modes_sigma
        self.length = length
        self.smooth_sr_fact = smooth_sr_fact
        self.rects_data = []
        fixed_seed = None if fixed_seed in [False, None] else int(fixed_seed)
        self.rng = np.random.RandomState(fixed_seed)

    def __len__(self) -> Union[int, float]:
        return self.length if self.length is not None else float('inf')

    def _extend_rects_data(self, min_length: int) -> None:
        n_to_generate = max(min_length - len(self.rects_data), 0)
        for _ in range(n_to_generate):
            v = self.rng.uniform(0.5, 1.0, (self.num_rects,))
            a1 = self.rng.uniform(0.1, .8, (self.num_rects,))
            a2 = self.rng.uniform(0.1, .8, (self.num_rects,))
            x = self.rng.uniform(-.75, .75, (self.num_rects,))
            y = self.rng.uniform(-.75, .75, (self.num_rects,))
            angle_modes = self.rng.uniform(0., np.pi, (self.num_angle_modes,))
            angle_modes_per_rect = angle_modes[self.rng.randint(
                    0, self.num_angle_modes, (self.num_rects,))]
            rot = self.rng.normal(angle_modes_per_rect, self.angle_modes_sigma)
            rot = np.mod(rot, np.pi)
            rects = np.stack((v, a1, a2, x, y, rot), axis=1)
            self.rects_data.append(rects)

    def _generate_item(self, idx: int) -> Tensor:
        image = _rect_phantom(self.shape, self.rects_data[idx], self.smooth_sr_fact)
        # normalize the foreground (all non-zero pixels) to [0., 1.]
        image[np.array(image) != 0.] -= np.min(image)
        image /= np.max(image)
        return torch.from_numpy(image[None])  # add channel dim

    def __iter__(self) -> Iterator[Tensor]:
        it = repeat(None, self.length) if self.length is not None else repeat(None)
        for idx, _ in enumerate(it):
            self._extend_rects_data(idx + 1)
            yield self._generate_item(idx)

    def __getitem__(self, idx: int) -> Tensor:
        self._extend_rects_data(idx + 1)
        return self._generate_item(idx)
