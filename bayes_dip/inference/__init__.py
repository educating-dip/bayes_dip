"""
Provides inference based on the predictive posterior.
"""
from .base_predictive_posterior import BasePredictivePosterior
from .sample_based_predictive_posterior import SampleBasedPredictivePosterior, log_prob_patches
from .exact_predictive_posterior import ExactPredictivePosterior
from .utils import get_image_patch_mask_inds
