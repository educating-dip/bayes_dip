"""
Provides random probe generation.
"""
from typing import Optional
import torch
from torch import Tensor

def generate_probes_bernoulli(
        side_length : int,
        num_probes : int,
        dtype=None,
        device=None,
        jacobi_vector : Optional[Tensor] = None
        ) -> Tensor:
    """
    Return Bernoulli-distributed random probes.

    Parameters
    ----------
    side_length : int
        Size of each probe vector.
    num_probes : int
        Number of probes.
    dtype : str or torch.dtype, optional
        Data type.
    device : str or torch.device, optional
        Device.
    jacobi_vector : Tensor, optional
        If specified, multiply the probes with ``jacobi_vector.pow(0.5)``.

    Returns
    -------
    probe_vectors : Tensor
        Probe vectors. Shape: ``(side_length, num_probes)``.
    """
    probe_vectors = torch.empty(side_length, num_probes, dtype=dtype, device=device)
    probe_vectors.bernoulli_().mul_(2).add_(-1)
    if jacobi_vector is not None:
        assert len(jacobi_vector.shape) == 1
        probe_vectors *= jacobi_vector.pow(0.5).unsqueeze(1)
    return probe_vectors  # side_length, num_probes
