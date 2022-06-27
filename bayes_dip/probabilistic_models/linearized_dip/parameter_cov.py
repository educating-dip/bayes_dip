import torch.nn as nn
from typing import Dict, Tuple, List, Callable
from functools import reduce

class ParamaterCov(nn.Module):

    def __init__(
        self,
        nn_model: nn.Module,
        prior_assignment_dict: Dict[str, Tuple[Callable, List[str]]], 
        hyperparams_init_dict: Dict[str, Dict],
        device=None
        ):
        
        super().__init__()
        self.nn_model = nn_model
        self.prior_assignment_dict = prior_assignment_dict
        self.hyperparams_init_dict = hyperparams_init_dict
        self.device = device 
        self.priors = self._create_prior_dict(
            self.nn_model, 
            self.prior_assignment_dict,
            self.hyperparams_init_dict
        )
        self.params_per_prior_type = self._ordered_params_under_prior()
    
    @property
    def ordered_nn_params(self, ):
        ordered_params_list = []
        for params in self.params_per_prior_type.values():
            ordered_params_list.extend(params)

        return ordered_params_list
    
    def _get_modules_by_names(self, 
            layer_names: List[str]
            ) -> List[nn.Module]:
        return [reduce(getattr, layer_name.split(sep='.'),
                            self.nn_model)  for layer_name in layer_names]
             
    def _create_prior_dict(self, ): 
        
        priors = {}
        for prior_name, (prior_type, layer_names) in self.prior_assignment_dict.items():
            init_hyperparams = self.hyperparams_init_dict[prior_name]
            modules = self._get_modules_by_names(layer_names)
            priors[prior_name] = prior_type(init_hyperparams=init_hyperparams, modules=modules, device=self.device) 
            
        return nn.ModuleDict(priors)
    
    def _ordered_params_under_prior(self, ):

        params_per_prior_type = {}
        for _, prior in self.priors.items():
            params_per_prior_type.setdefault([type(prior)], [])
            params_per_prior_type[type(prior)].extend(prior.get_params_under_prior())
        
        return params_per_prior_type


            
    