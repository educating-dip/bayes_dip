import torch
import torch.nn as nn
import numpy as np
import torch.linalg as linalg
from typing import Dict, Tuple, List, Callable
from functools import reduce

from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from .priors import GPprior, RadialBasisFuncCov, NormalPrior

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