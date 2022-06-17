"""
Provides data generation, data access, and the ray transform.
"""

from .datasets import RectanglesDataset
from .trafo import (
        BaseRayTrafo, MatmulRayTrafo,
        get_odl_ray_trafo_parallel_beam_2d, ParallelBeam2DRayTrafo,
        get_odl_ray_trafo_parallel_beam_2d_matrix,
        get_parallel_beam_2d_matmul_ray_trafo)
from .simulation import simulate, SimulatedDataset
