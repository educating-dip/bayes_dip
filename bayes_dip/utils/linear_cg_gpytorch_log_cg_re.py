"""
Clone of :func:`linear_log_cg_re` from
https://github.com/AndPotap/halfpres_gps/blob/6aead66d9d9efc30b5e3ee3a49697d660a8c4043/core/gpytorch_log_cg_re.py
also returning the residual, a re-orthogonalizing CG variant introduced in [1]_.

.. [1] W.J. Maddox, A. Potapczynski, and A.G. Wilson, 2022, "Low Precision Arithmetic for Fast
        Gaussian Processes". The 38th Conference on Uncertainty in Artificial Intelligence.
        https://openreview.net/forum?id=S3NOX_Ij9xc
"""
import torch
import logging

# pylint: disable=all

def _default_preconditioner(x):
    return x.clone()


def linear_log_cg_re(
    matmul_closure,
    rhs,
    tolerance,
    max_iter,
    initial_guess=None,
    preconditioner=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_tridiag_iter=0,
):
    x0 = initial_guess if initial_guess is not None else torch.zeros_like(rhs)
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
    rhs = rhs.div(rhs_norm)

    if preconditioner is None:
        preconditioner = _default_preconditioner

    state = initialize_log_re(matmul_closure, rhs, preconditioner, x0, max_iter)
    for k in range(max_iter):
        state = take_cg_step_log_re(state, matmul_closure, preconditioner)
        if cond_fun(state, tolerance, max_iter):
            break

    x0 = state[0]
    r0 = state[1]
    x0 = x0.mul(rhs_norm)
    logging.info(f'CG Iters: {k + 1}')
    return x0, r0.norm(2, dim=-2, keepdim=True)


def initialize_log_re(A, b, preconditioner, x0, max_iters):
    r0 = b - A(x0)
    z0 = preconditioner(r0)
    p0 = z0
    log_gamma0 = update_log_gamma_unclipped(r=r0, z=z0)
    u_all = torch.zeros(size=(max_iters,) + b.shape, dtype=x0.dtype, device=x0.device)
    return (x0, r0, log_gamma0, p0, u_all, torch.tensor(0, dtype=torch.int32))

def re_orthogonalization(x, k, u_all):
    for i in range(k):
        dotprod = torch.sum(x * u_all[i], dim=-2) * u_all[i]
        x = x - dotprod
    return x

def take_cg_step_log_re(state, A, preconditioner):
    x0, r0, log_gamma0, p0, u_all, k = state
    has_converged = torch.linalg.norm(r0, axis=0) < torch.tensor(1.e-8, dtype=p0.dtype)
    Ap0 = A(p0)
    alpha = update_alpha_log_unclipped(log_gamma0, p0, Ap0, has_converged)

    x1 = x0 + alpha * p0
    r1 = r0 - alpha * Ap0

    r1 = re_orthogonalization(r1, k, u_all)
    z1 = preconditioner(r1)

    log_gamma1, beta = update_log_gamma_beta_unclipped(
        r1, z1, log_gamma0, has_converged)
    u_all[k] = r1 / r1.norm(dim=0)
    p1 = z1 + beta * p0
    # print_progress(k, alpha, r1, torch.exp(log_gamma1), beta)
    return (x1, r1, log_gamma1, p1, u_all, k + 1)

def update_alpha_log_unclipped(log_gamma, p, Ap, has_converged):
    log_alpha_abs, sign = compute_robust_denom_unclipped(p, Ap)
    log_denom = logsumexp(tensor=log_alpha_abs, dim=0, mask=sign)
    alpha = torch.exp(log_gamma - log_denom)
    alpha = torch.where(has_converged, torch.zeros_like(alpha), alpha)
    return alpha

def compute_robust_denom_unclipped(p, Ap):
    p_abs = torch.clip(torch.abs(p), min=torch.tensor(1.e-8, device=p.device))
    Ap_abs = torch.clip(torch.abs(Ap), min=torch.tensor(1.e-8, device=Ap.device))
    sign = torch.sign(p) * torch.sign(Ap)
    log_alpha_abs = torch.log(p_abs) + torch.log(Ap_abs)
    return log_alpha_abs, sign


def update_log_gamma_beta_unclipped(r, z, log_gamma0, has_converged):
    log_gamma1 = update_log_gamma_unclipped(r, z)
    beta = torch.exp(log_gamma1 - log_gamma0)
    beta = torch.where(has_converged, torch.zeros_like(beta), beta)
    return log_gamma1, beta


def update_log_gamma_unclipped(r, z, min_val=1e-45):
    r_abs = torch.abs(r).clip(min=min_val)
    z_abs = torch.abs(z).clip(min=min_val)
    sign = torch.sign(r) * torch.sign(z)
    log_gamma_abs = torch.log(r_abs) + torch.log(z_abs)
    log_gamma = logsumexp(tensor=log_gamma_abs, dim=0, mask=sign)
    return log_gamma


def cond_fun(state, tolerance, max_iters):
    _, r, *_, k = state
    rs = torch.linalg.norm(r, axis=0)
    res_meet = torch.mean(rs) < tolerance
    min_val = torch.minimum(torch.tensor(10, dtype=torch.int32),
                            torch.tensor(max_iters, dtype=torch.int32))
    flag = ((res_meet) & (k >= min_val) | (k > max_iters))
    return flag


def logsumexp(tensor, dim=-1, mask=None, return_sign=False, min_val=1e-45):
    max_entry = torch.max(tensor, dim, keepdim=True)[0]
    summ = torch.sum((tensor - max_entry).exp() * mask, dim)
    out = max_entry + summ.clip(min=min_val).log()
    return out
