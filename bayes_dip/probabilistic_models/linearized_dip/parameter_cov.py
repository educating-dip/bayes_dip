"""Provides :class:`ParameterCov`"""
from typing import Dict, Tuple, List, Callable, Optional
import torch
from torch import nn, Tensor
from bayes_dip.utils import get_modules_by_names
from .parameter_priors.priors import BaseGaussPrior

class ParameterCov(nn.Module):
    """
    Covariance in parameter space.
    """

    def __init__(
        self,
        nn_model: nn.Module,
        prior_assignment_dict: Dict[str, Tuple[Callable[..., BaseGaussPrior], List[str]]],
        hyperparams_init_dict: Dict[str, Dict],
        device=None
        ):
        """
        Parameters
        ----------
        nn_model : :class:`nn.Module`
            Network.
        prior_assignment_dict : dict
            Dictionary specifying the assignment of priors over network modules.
            The keys are prior names, and each value is a tuple ``(constructor, module_names)``,
            where ``constructor`` is a callable with parameters
            ``init_hyperparams: Dict, modules: List[nn.Module], device: Union[str, torch.device]``
            and ``module_names`` is a list of module names in ``nn_model`` used to create the
            ``modules`` list passed to ``constructor``. The prior object created by ``constructor``
            must be an instance of :class:`BaseGaussPrior`.
        hyperparams_init_dict : dict
            Initial hyperparameter values for the priors.
            The keys are prior names (same as ``prior_assignment_dict``), and each value specifies
            the ``init_hyperparams`` argument to ``constructor`` (see ``prior_assignment_dict``).
        device : str or torch.device, optional
            Device. If ``None`` (the default), ``'cuda:0'`` is chosen if available or ``'cpu'``
            otherwise.
        """
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
    def ordered_nn_params(self) -> List[nn.Parameter]:
        """
        List of all parameters under prior, in the order expected by :meth:`forward`.

        The parameters are ordered first (outer-most) by prior type, then by prior in the order of
        occurence in :attr:`priors` (same order as in the ``prior_assignment_dict``), and finally
        (inner-most) in the order returned by ``prior.get_params_under_prior()``.
        """
        ordered_params_list = []
        for params in self.params_per_prior_type.values():
            ordered_params_list.extend(params)

        return ordered_params_list

    @property
    def log_variances(self) -> List[nn.Parameter]:
        """
        List of all ``log_variance`` parameters, in the order of :attr:`self.priors` (same order as
        in the ``prior_assignment_dict``).
        """
        return [prior.log_variance for prior in self.priors.values()]

    def _create_prior_dict(self, nn_model: nn.Module):

        priors = {}
        for prior_name, (prior_type, layer_names) in self.prior_assignment_dict.items():
            init_hyperparams = self.hyperparams_init_dict[prior_name]
            modules = get_modules_by_names(nn_model, layer_names)
            priors[prior_name] = prior_type(
                    init_hyperparams=init_hyperparams, modules=modules, device=self.device)

        return nn.ModuleDict(priors)

    def _ordered_params_per_prior_type(self):
        params_per_prior_type = {}
        for _, prior in self.priors.items():
            params_per_prior_type.setdefault(type(prior), [])
            params_per_prior_type[type(prior)].extend(prior.get_params_under_prior())

        return params_per_prior_type

    def _ordered_priors_per_prior_type(self):
        priors_per_prior_type = {}
        for _, prior in self.priors.items():
            priors_per_prior_type.setdefault(type(prior), [])
            priors_per_prior_type[type(prior)].append(prior)

        return priors_per_prior_type

    def _params_numel_per_prior_type(self):
        params_numel_per_prior_type = {}
        for prior_type, params in self.params_per_prior_type.items():
            params_numel_per_prior_type[prior_type] = sum(p.data.numel() for p in params)

        return params_numel_per_prior_type

    def _params_slices_per_prior(self):
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
            only_prior: nn.Module = None,
            **kwargs
        ) -> Tensor:
        """
        Multiply with the covariance "matrix".

        I.e., evaluate ``(cov @ v.T).T`` where ``cov`` is a matrix representation of ``self``.

        Parameters
        ----------
        v : Tensor
            Parameter sets.
            Shape: ``(batch_size, num_params)``, where ``num_params`` is ``self.shape[0]`` if
            ``only_prior is None`` and otherwise the restricted parameter number
            ``params_slice.stop - params_slice.start`` with
            ``params_slice == self.params_slices_per_prior[only_prior]``.

        Returns
        -------
        Tensor
            Products. Shape: same as ``v``.
        """
        if only_prior is None:
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

            out = torch.cat(v_parameter_cov_mul, dim=-1)
        else:
            out = type(only_prior).batched_cov_mul(
                priors=[only_prior],
                v=v,
                **kwargs
            )

        return out

    def sample(self,
        num_samples: int = 10,
        mean: Optional[Tensor] = None,
        sample_only_from_prior: nn.Module = None,
        ) -> Tensor:
        """
        Sample from a Gaussian with this covariance.

        By default, the mean is zero, but a custom ``mean`` can be specified.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples. The default is ``10``.
        mean : Tensor, optional
            Mean. If ``None`` (the default), the mean is zero.
            Shape: ``(batch_size, num_params)`` (same shape as the return value).
        sample_only_from_prior : :class:`nn.Module`, optional
            If specified, return only samples for the parameters under this prior.

        Returns
        -------
        samples : Tensor
            Samples. Shape: ``(batch_size, num_params)``, where ``num_params`` is ``self.shape[0]``
            if ``sample_only_from_prior is None`` and otherwise the restricted parameter number
            ``params_slice.stop - params_slice.start`` with
            ``params_slice == self.params_slices_per_prior[sample_only_from_prior]``.
        """
        num_params = (
                self.shape[0] if sample_only_from_prior is None else
                (self.params_slices_per_prior[sample_only_from_prior].stop -
                 self.params_slices_per_prior[sample_only_from_prior].start))
        samples = torch.randn(
            num_samples, num_params,
            device=self.device
            )
        samples = self.forward(samples, use_cholesky=True, only_prior=sample_only_from_prior)
        if mean is not None:
            samples = samples + mean
        return samples

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the (theoretical) matrix representation."""
        return (sum(self.params_numel_per_prior_type.values()),) * 2
