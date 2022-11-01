import pytest
import torch
from torch import nn, autograd
import numpy as np
from bayes_dip.utils import tv_loss, batch_tv_grad

@pytest.fixture(scope='session')
def params_and_jac_and_image_and_weight_samples():
    dp = 30
    im_shape = (8, 8)
    dx = np.prod(im_shape)
    torch.manual_seed(1)
    offset_params = nn.Parameter(torch.randn(dp))
    scale_params = nn.Parameter(torch.rand(dp))
    params = (offset_params, scale_params)
    num_samples = 1000
    jac = torch.randn(dx, dp)
    weight_samples = torch.randn(num_samples, dp) * scale_params + offset_params
    image_mean = torch.rand(1, 1, *im_shape)
    x_samples = (weight_samples @ jac.T).view(num_samples, 1, *im_shape) + image_mean

    return params, jac, x_samples, weight_samples

@pytest.fixture(scope='session')
def params_and_image_rsamples_and_jac_and_manual_weight_and_image_samples():
    dp = 30
    im_shape = (4, 4)
    dx = np.prod(im_shape)
    torch.manual_seed(1)
    image_mean = torch.rand(1, 1, *im_shape)
    jac = torch.randn(dx, dp)

    scale_params = nn.Parameter(torch.rand(dp))
    params = (scale_params,)

    # params ~ N(0, diag(scale_params))
    # x ~ N(image_mean, J @ diag(scale_params) @ J.T)

    cov_ff = jac @ torch.diag(scale_params) @ jac.T
    dist = \
        torch.distributions.multivariate_normal.MultivariateNormal(
            loc=image_mean.flatten(),
            scale_tril=torch.linalg.cholesky(cov_ff)
        )

    num_samples = 1000000
    x_samples = dist.rsample((num_samples,)).view(num_samples, 1, *im_shape)

    # manual way of getting x_samples and weight_samples
    manual_weight_samples = torch.randn(num_samples, dp) * scale_params**0.5
    manual_x_samples = (manual_weight_samples @ jac.T).view(num_samples, 1, *im_shape) + image_mean

    return params, x_samples, jac, manual_weight_samples, manual_x_samples

def test_first_derivative(params_and_jac_and_image_and_weight_samples):
    params, jac, x_samples, weight_samples = params_and_jac_and_image_and_weight_samples

    assert x_samples.ndim == 4
    assert x_samples.shape[1] == 1
    num_samples = x_samples.shape[0]
    im_shape = x_samples.shape[2:]
    dx = np.prod(im_shape)
    assert weight_samples.ndim == 2
    assert weight_samples.shape[0] == num_samples
    # dp = weight_samples.shape[1]

    tv_x_samples = batch_tv_grad(x_samples)
    jac_tv_x_samples = tv_x_samples.view(-1, dx) @ jac
    jac_tv_x_samples = jac_tv_x_samples.detach()

    shifted_loss = (weight_samples * jac_tv_x_samples).sum(dim=1).mean(dim=0)
    first_derivative_grads = autograd.grad(
        shifted_loss,
        params,
        allow_unused=True,
        create_graph=True,
        retain_graph=True
    )

    loss = tv_loss(x_samples) / num_samples  # tv_loss sums over batch dim
    first_derivative_grads_via_autograd = autograd.grad(
        loss,
        params,
        allow_unused=True,
        create_graph=True,
        retain_graph=True)

    for grad, grad_via_autograd in zip(first_derivative_grads, first_derivative_grads_via_autograd):
        assert torch.allclose(grad, grad_via_autograd)

def test_first_derivative_dist_rsamples(params_and_image_rsamples_and_jac_and_manual_weight_and_image_samples):
    params, x_samples, jac, manual_weight_samples, manual_x_samples = (
            params_and_image_rsamples_and_jac_and_manual_weight_and_image_samples)

    assert x_samples.ndim == 4
    assert x_samples.shape[1] == 1
    num_samples = x_samples.shape[0]
    im_shape = x_samples.shape[2:]
    dx = np.prod(im_shape)
    assert manual_weight_samples.ndim == 2
    assert manual_weight_samples.shape[0] == num_samples
    dp = manual_weight_samples.shape[1]
    assert manual_x_samples.shape == x_samples.shape

    assert torch.allclose(
            torch.mean(x_samples, dim=0).mean(), torch.mean(manual_x_samples, dim=0).mean(),
            rtol=1e-3)
    assert torch.allclose(
            torch.std(x_samples, dim=0).mean(), torch.std(manual_x_samples, dim=0).mean(),
            rtol=1e-3)

    loss = tv_loss(x_samples) / num_samples  # tv_loss sums over batch dim
    first_derivative_grads = autograd.grad(
        loss,
        params,
        allow_unused=True,
        create_graph=True,
        retain_graph=True)

    manual_loss = tv_loss(manual_x_samples) / num_samples  # tv_loss sums over batch dim
    manual_first_derivative_grads = autograd.grad(
        manual_loss,
        params,
        allow_unused=True,
        create_graph=True,
        retain_graph=True)

    for grad, manual_grad in zip(first_derivative_grads, manual_first_derivative_grads):
        assert torch.allclose(grad, manual_grad, rtol=1e-2)
