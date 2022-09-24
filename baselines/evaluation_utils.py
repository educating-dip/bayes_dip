import os
import torch
from omegaconf import OmegaConf
from bayes_dip.utils.experiment_utils import load_samples

def compute_mcdo_reconstruction(
        run_path: str, sample_idx: int,
        device=None,
        ) -> float:
    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
    device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
    dtype = torch.double if cfg.use_double else torch.float
    samples = load_samples(
            path=cfg.baseline.load_samples_from_path, i=sample_idx,
            num_samples=cfg.baseline.num_samples
        ).to(dtype=dtype, device=device)
    mean_recon = samples.mean(dim=0, keepdim=True)
    return mean_recon
