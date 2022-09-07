import pytest
import torch
import scipy
from scipy.stats import ortho_group
import numpy as np
try:
    from linear_operator import pivoted_cholesky
except:
    # for gyptorch < 1.9
    from gpytorch import pivoted_cholesky

from bayes_dip.utils import cg

@pytest.fixture(scope='function')
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

@pytest.fixture(scope='function')
def ichol_precon_closure(linear_system):
    (mat, noise), rhs = linear_system
    ichol = pivoted_cholesky(mat, rank=3)

    # use the incomplete cholesky factorization as a preconditioner like implemented in
    # https://github.com/cornellius-gp/linear_operator/blob/987df55260afea79eb0590c7e546b221cfec3fe5/linear_operator/operators/added_diag_linear_operator.py#L84

    n, k = ichol.shape

    _q_cache, _r_cache = torch.linalg.qr(
        torch.cat((ichol, noise.sqrt() * torch.eye(k, dtype=ichol.dtype)), dim=-2)
    )
    _q_cache = _q_cache[:n, :]

    def precon_closure(v):
        qqt_v = _q_cache.matmul(_q_cache.mT.matmul(v))
        return (1. / noise) * (v - qqt_v)

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
