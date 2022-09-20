"""
Provides the utils for the baseline MC-Dropout Deep Image Prior (MCDO-DIP).
"""
from .mcdo_dip_utils import (
    bayesianize_unet_architecture, conv2d_dropout, sample_from_bayesianized_model, approx_kernel_density)