import pytest
import torch
import scipy
from scipy.stats import ortho_group
import numpy as np

from bayes_dip.utils import cg
from bayes_dip.marginal_likelihood_optim.preconditioner import (
    JacobiPreconditioner, IncompleteCholeskyPreconditioner)
from bayes_dip.marginal_likelihood_optim.preconditioner_utils import (
        approx_diag, pivoted_cholesky)

@pytest.fixture(scope='session')
def linear_system():
    n = 5
    rng = np.random.default_rng(0)
    U = ortho_group.rvs(n, random_state=rng)
    L = np.array([2.1, 1.7, 1.4, 0.7, 0.4])  # np.exp(-1. * rng.normal(size=n))
    mat = U @ (L[:, None] * U.T)
    mat = torch.from_numpy(mat)
    noise = torch.from_numpy(np.asarray(1e-5))
    try:
        torch.linalg.cholesky(mat + noise * torch.eye(mat.shape[0]))
    except:
        raise RuntimeError('System matrix is not choleskable, cannot run test.')
    rhs = rng.normal(size=n)
    rhs = torch.from_numpy(rhs)
    # `(mat, noise)` should be interpreted as `mat + noise * torch.eye(mat.shape[0])`
    return (mat, noise), rhs

@pytest.fixture(scope='session')
def jacobi_precon_closure(linear_system):
    (mat, noise), rhs = linear_system
    diag = mat.diag() + noise

    precon = JacobiPreconditioner(vector=diag)
    precon_closure = precon.get_closure()

    return precon_closure

@pytest.fixture(scope='session')
def approx_jacobi_precon_closure(linear_system):
    (mat, noise), rhs = linear_system

    torch.manual_seed(0)
    vector = approx_diag(
            lambda v: mat @ v + noise * v,
            size=rhs.shape[0], num_samples=2, batch_size=2, dtype=rhs.dtype, device=rhs.device)

    precon = JacobiPreconditioner(vector=vector)
    precon_closure = precon.get_closure()

    return precon_closure

@pytest.fixture(scope='session')
def ichol_precon_closure(linear_system):
    (mat, noise), rhs = linear_system

    ichol, _ = pivoted_cholesky(
            closure=lambda v: mat @ v,
            size=mat.shape[0],
            max_iter=3,
            matrix_diag=mat.diag(),
            batch_size=2,
            error_tol=1e-3,
            dtype=mat.dtype,
            device=mat.device,
    )

    precon = IncompleteCholeskyPreconditioner(
            incomplete_cholesky=ichol, log_noise_variance=noise.log())
    precon_closure = precon.get_closure()

    return precon_closure

@pytest.fixture(scope='session')
def ichol_approx_diag_precon_closure(linear_system):
    (mat, noise), rhs = linear_system

    torch.manual_seed(0)
    ichol, _ = pivoted_cholesky(
            closure=lambda v: mat @ v,
            size=mat.shape[0],
            max_iter=3,
            approx_diag_num_samples=300,
            batch_size=2,
            error_tol=1e-3,
            dtype=mat.dtype,
            device=mat.device,
    )

    precon = IncompleteCholeskyPreconditioner(
            incomplete_cholesky=ichol, log_noise_variance=noise.log())
    precon_closure = precon.get_closure()

    return precon_closure

def _residual_norm(mat, noise, rhs, solution):
    residual = (mat @ solution + noise * solution - rhs).squeeze()
    assert residual.ndim == 1
    return residual.norm()

def test_cg(linear_system):
    (mat, noise), rhs = linear_system
    print()
    print('mat:', mat)
    print('noise:', noise)
    print('rhs:', rhs)

    scipy_cg_solution = scipy.sparse.linalg.cg(
            mat.numpy() + noise.item() * np.eye(mat.shape[0]), rhs.numpy(), atol=0.)[0]
    scipy_cg_solution = torch.from_numpy(scipy_cg_solution)
    print('residual norm of scipy CG solution:', _residual_norm(
            mat, noise, rhs, scipy_cg_solution))

    precon_closure = lambda v: v.clone()

    cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            )
    cg_solution = cg_solution[:, 0]
    print('residual norm of CG solution:', _residual_norm(
            mat, noise, rhs, cg_solution))

    assert torch.allclose(cg_solution, scipy_cg_solution)

def test_jacobi_preconditioner(linear_system, jacobi_precon_closure):
    (mat, noise), rhs = linear_system
    print()
    print('mat:', mat)
    print('noise:', noise)
    print('rhs:', rhs)

    scipy_cg_solution = scipy.sparse.linalg.cg(
            mat.numpy() + noise.item() * np.eye(mat.shape[0]), rhs.numpy(), atol=0.)[0]
    scipy_cg_solution = torch.from_numpy(scipy_cg_solution)
    print('residual norm of scipy CG solution:', _residual_norm(
            mat, noise, rhs, scipy_cg_solution))

    precon_closure = jacobi_precon_closure

    cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            )
    cg_solution = cg_solution[:, 0]
    print('residual norm of CG solution:', _residual_norm(
            mat, noise, rhs, cg_solution))

    assert torch.allclose(cg_solution, scipy_cg_solution)

    zero_residual_norm = _residual_norm(mat, noise, rhs, torch.zeros_like(cg_solution))

    # with the exact jacobi preconditioning, the system is mostly solved after 3 iterations

    for max_niter, max_rel_residual_norm in [(1, 0.8), (2, 0.8), (3, 0.04), (4, 0.02), (5, 1e-6)]:
        cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            max_niter=max_niter,
            rtol=0.
            )
        cg_solution = cg_solution[:, 0]
        rel_residual_norm = _residual_norm(mat, noise, rhs, cg_solution) / zero_residual_norm
        assert rel_residual_norm < max_rel_residual_norm

def test_approx_jacobi_preconditioner(linear_system, approx_jacobi_precon_closure):
    (mat, noise), rhs = linear_system
    print()
    print('mat:', mat)
    print('noise:', noise)
    print('rhs:', rhs)

    scipy_cg_solution = scipy.sparse.linalg.cg(
            mat.numpy() + noise.item() * np.eye(mat.shape[0]), rhs.numpy(), atol=0.)[0]
    scipy_cg_solution = torch.from_numpy(scipy_cg_solution)
    print('residual norm of scipy CG solution:', _residual_norm(
            mat, noise, rhs, scipy_cg_solution))

    precon_closure = approx_jacobi_precon_closure

    cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            )
    cg_solution = cg_solution[:, 0]
    print('residual norm of CG solution:', _residual_norm(
            mat, noise, rhs, cg_solution))

    assert torch.allclose(cg_solution, scipy_cg_solution)

    zero_residual_norm = _residual_norm(mat, noise, rhs, torch.zeros_like(cg_solution))

    # with the approximate jacobi preconditioning, the system is mostly solved after 3 iterations

    for max_niter, max_rel_residual_norm in [(1, 0.8), (2, 0.8), (3, 0.06), (4, 0.02), (5, 1e-6)]:
        cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            max_niter=max_niter,
            rtol=0.
            )
        cg_solution = cg_solution[:, 0]
        rel_residual_norm = _residual_norm(mat, noise, rhs, cg_solution) / zero_residual_norm
        assert rel_residual_norm < max_rel_residual_norm

def test_ichol_preconditioner(linear_system, ichol_precon_closure):
    (mat, noise), rhs = linear_system
    print()
    print('mat:', mat)
    print('noise:', noise)
    print('rhs:', rhs)

    scipy_cg_solution = scipy.sparse.linalg.cg(
            mat.numpy() + noise.item() * np.eye(mat.shape[0]), rhs.numpy(), atol=0.)[0]
    scipy_cg_solution = torch.from_numpy(scipy_cg_solution)
    print('residual norm of scipy CG solution:', _residual_norm(
            mat, noise, rhs, scipy_cg_solution))

    precon_closure = ichol_precon_closure

    cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            )
    cg_solution = cg_solution[:, 0]
    print('residual norm of CG solution:', _residual_norm(
            mat, noise, rhs, cg_solution))

    assert torch.allclose(cg_solution, scipy_cg_solution)

    zero_residual_norm = _residual_norm(mat, noise, rhs, torch.zeros_like(cg_solution))

    # with the rank-3-ichol preconditioning, the system is solved after 3 iterations

    for max_niter, max_rel_residual_norm in [(1, 0.8), (2, 0.8), (3, 1e-6), (4, 1e-6), (5, 1e-6)]:
        cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            max_niter=max_niter,
            rtol=0.
            )
        cg_solution = cg_solution[:, 0]
        rel_residual_norm = _residual_norm(mat, noise, rhs, cg_solution) / zero_residual_norm
        assert rel_residual_norm < max_rel_residual_norm

def test_ichol_approx_diag_preconditioner(linear_system, ichol_approx_diag_precon_closure):
    (mat, noise), rhs = linear_system
    print()
    print('mat:', mat)
    print('noise:', noise)
    print('rhs:', rhs)

    scipy_cg_solution = scipy.sparse.linalg.cg(
            mat.numpy() + noise.item() * np.eye(mat.shape[0]), rhs.numpy(), atol=0.)[0]
    scipy_cg_solution = torch.from_numpy(scipy_cg_solution)
    print('residual norm of scipy CG solution:', _residual_norm(
            mat, noise, rhs, scipy_cg_solution))

    precon_closure = ichol_approx_diag_precon_closure

    cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            )
    cg_solution = cg_solution[:, 0]
    print('residual norm of CG solution:', _residual_norm(
            mat, noise, rhs, cg_solution))

    assert torch.allclose(cg_solution, scipy_cg_solution)

    zero_residual_norm = _residual_norm(mat, noise, rhs, torch.zeros_like(cg_solution))

    # with the rank-3-ichol preconditioning using the approximate diagonal estimate, the system is
    # solved after 3 iterations

    for max_niter, max_rel_residual_norm in [(1, 0.8), (2, 0.8), (3, 1e-6), (4, 1e-6), (5, 1e-6)]:
        cg_solution, _ = cg(
            lambda v: mat @ v + noise * v, rhs[:, None], precon_closure=precon_closure,
            use_log_re_variant=False,  # Maddox often produces nan
            max_niter=max_niter,
            rtol=0.
            )
        cg_solution = cg_solution[:, 0]
        rel_residual_norm = _residual_norm(mat, noise, rhs, cg_solution) / zero_residual_norm
        assert rel_residual_norm < max_rel_residual_norm
