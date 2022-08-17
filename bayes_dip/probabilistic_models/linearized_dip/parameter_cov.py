"""Provides :class:`ParameterCov`"""
from typing import Dict, Tuple, List, Callable, Optional
import torch
from torch import nn, Tensor
from bayes_dip.utils import get_modules_by_names

class ParameterCov(nn.Module):
    """
    Covariance in parameter space.
    """

    def __init__(
        self,
        nn_model: nn.Module,
        prior_assignment_dict: Dict[str, Tuple[Callable, List[str]]],
        hyperparams_init_dict: Dict[str, Dict],
        device=None
        ):

        super().__init__()
        self.prior_assignment_dict = prior_assignment_dict
        self.hyperparams_init_dict = hyperparams_init_dict
        self.device = device or torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.priors = self._create_prior_dict(nn_model)
        self.params_per_prior_type = self._ordered_params_per_prior_type()
        self.priors_per_prior_type = self._ordered_priors_per_prior_type()
        self.params_numel_per_prior_type = self._params_numel_per_prior_type()
        self.params_slices_per_prior = self._params_slices_per_prior()

    @property
    def ordered_nn_params(self, ):
        ordered_params_list = []
        for params in self.params_per_prior_type.values():
            ordered_params_list.extend(params)

        return ordered_params_list

    @property
    def log_variances(self, ):
        return [prior.log_variance for prior in self.priors.values()]

    def _create_prior_dict(self, nn_model: nn.Module):

        priors = {}
        for prior_name, (prior_type, layer_names) in self.prior_assignment_dict.items():
            init_hyperparams = self.hyperparams_init_dict[prior_name]
            modules = get_modules_by_names(nn_model, layer_names)
            priors[prior_name] = prior_type(
                    init_hyperparams=init_hyperparams, modules=modules, device=self.device)

        return nn.ModuleDict(priors)

    def _ordered_params_per_prior_type(self, ):

        params_per_prior_type = {}
        for _, prior in self.priors.items():
            params_per_prior_type.setdefault(type(prior), [])
            params_per_prior_type[type(prior)].extend(prior.get_params_under_prior())

        return params_per_prior_type

    def _ordered_priors_per_prior_type(self, ):

        priors_per_prior_type = {}
        for _, prior in self.priors.items():
            priors_per_prior_type.setdefault(type(prior), [])
            priors_per_prior_type[type(prior)].append(prior)

        return priors_per_prior_type

    def _params_numel_per_prior_type(self, ):

        params_numel_per_prior_type = {}
        for prior_type, params in self.params_per_prior_type.items():
            params_numel_per_prior_type[prior_type] = sum(p.data.numel() for p in params)

        return params_numel_per_prior_type

    def _params_slices_per_prior(self, ):
        params_slices_per_prior = {}
        params_cnt = 0
        for _, priors in self.priors_per_prior_type.items():
            for prior in priors:
                params_numel = sum(p.data.numel() for p in prior.get_params_under_prior())
                params_slices_per_prior[prior] = slice(params_cnt, params_cnt + params_numel)
                params_cnt += params_numel

        return params_slices_per_prior

    def forward(self,
            v: Tensor,
            **kwargs
        ) -> Tensor:

        v_parameter_cov_mul = []
        params_cnt = 0
        for (prior_type, priors), len_params in zip(
                self.priors_per_prior_type.items(), self.params_numel_per_prior_type.values()
            ):

            v_parameter_cov_mul.append(prior_type.batched_cov_mul(
                    priors=priors,
                    v=v[:, params_cnt:params_cnt+len_params],
                    **kwargs
                )
            )
            params_cnt += len_params

        return torch.cat(v_parameter_cov_mul, dim=-1)

    def sample(self,
        num_samples: int = 10,
        mean: Optional[Tensor] = None,
        sample_only_from_prior: nn.Module = None,
        ) -> Tensor:
        samples = torch.randn(num_samples, self.shape[0],
            device=self.device
            )
        samples = self.forward(samples, use_cholesky=True)
        if sample_only_from_prior is not None:
            # zero all values except those in self.params_slices_per_prior[sample_only_from_prior]
            samples[:, :self.params_slices_per_prior[sample_only_from_prior].start] = 0.
            samples[:, self.params_slices_per_prior[sample_only_from_prior].stop:] = 0.
        if mean is not None:
            samples = samples + mean
        return samples

    @property
    def shape(self) -> Tuple[int, int]:
        return (sum(self.params_numel_per_prior_type.values()),) * 2
