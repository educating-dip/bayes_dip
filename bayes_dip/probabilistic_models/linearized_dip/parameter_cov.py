import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Tuple, List, Callable
from bayes_dip.utils import get_modules_by_names

class ParameterCov(nn.Module):

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
        self.params_per_prior_type = self._ordered_params_under_prior()
        self.priors_per_prior_type = self._ordered_priors_per_prior_type()
        self.params_numel_per_prior_type = self._params_numel_per_prior_type()
    
    @property
    def ordered_nn_params(self, ):
        ordered_params_list = []
        for params in self.params_per_prior_type.values():
            ordered_params_list.extend(params)

        return ordered_params_list

    def _create_prior_dict(self, nn_model: nn.Module):
        
        priors = {}
        for prior_name, (prior_type, layer_names) in self.prior_assignment_dict.items():
            init_hyperparams = self.hyperparams_init_dict[prior_name]
            modules = get_modules_by_names(nn_model, layer_names)
            priors[prior_name] = prior_type(init_hyperparams=init_hyperparams, modules=modules, device=self.device) 
            
        return nn.ModuleDict(priors)
    
    def _ordered_params_under_prior(self, ):

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