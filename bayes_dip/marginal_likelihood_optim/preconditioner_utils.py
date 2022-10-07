"""
Provides utilities for the preconditioners in :mod:`.preconditioner`.
"""
from typing import Callable, Optional
from math import ceil
import torch
from torch import Tensor
from tqdm import tqdm
from .random_probes import generate_probes_bernoulli

def approx_diag(
        closure: Callable, size: int, num_samples: int, batch_size: int = 1,
        dtype=None, device=None):
    """
    Return an approximation of a matrix diagonal, estimated using matrix-vector products.

    This implements the matrix-free estimator described in Figure 1 in [1]_.

    .. [1] C. Bekas, E. Kokiopoulou, and Y. Saad, 2007, "An estimator for the diagonal of a matrix".
           Applied Numerical Mathematics. https://doi.org/10.1016/J.APNUM.2007.01.003

    Parameters
    ----------
    closure : callable
        Matmul closure. The closure receives and returns tensors of shape ``(size, batch_size)``.
    size : int
        Matrix size (side length).
    num_samples : int
        Number of samples to use for the estimation.
    batch_size : int, optional
        Batch size for evaluating the closure. The default is ``1``.
    dtype : str or torch.dtype, optional
        Data type.
    device : str or torch.device, optional
        Device.

    Returns
    -------
    estimated_diag : Tensor
        Matrix diagonal estimate. Shape: ``(size,)``.
    """
    num_batches = ceil(num_samples / batch_size)
    t = torch.zeros(size, dtype=dtype, device=device)
    q = torch.zeros(size, dtype=dtype, device=device)
    for _ in tqdm(range(num_batches), desc='approx_diag', miniters=num_batches//100):
        v = generate_probes_bernoulli(
                side_length=size,
                num_probes=batch_size,
                dtype=dtype,
                device=device,
                jacobi_vector=None)  # (size, batch_size)
        t += (closure(v) * v).sum(dim=1)
        q += (v * v).sum(dim=1)
    d = t / q
    return d

def pivoted_cholesky(
            closure: Callable,
            size: int,
            max_iter: int,
            approx_diag_num_samples: int = 100,
            batch_size: int = 1,
            error_tol: float = 1e-3,
            recompute_max_diag_values: bool = True,
            matrix_diag: Optional[Tensor] = None,
            verbose: bool = True,
            dtype=None,
            device=None,
        ):  # pylint: disable=too-many-arguments
    """
    Simplified clone of the Pivoted Cholesky decomposition implementation from
    https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/functions/_pivoted_cholesky.py
    that directly uses a closure.

    Changes made to the original implementation:

        * use closure instead of LinearOperator instance
        * do not support batch dims (we just need one matrix)
        * estimate diagonal with :func:`approx_diag` or use manually passed diagonal
        * use square root of the *exact* diagonal value for each selected pivot to populate
            ``L[m, m]`` (the row is computed anyways)
    """
    # pylint: disable=too-many-locals,too-many-statements

    matrix_shape = (size, size)

    # Need to get diagonals. This is easy if it's a LinearOperator, since
    # LinearOperator.diagonal() operates in batch mode.
    if matrix_diag is None:
        if verbose:
            print(
                f"Estimating the diagonal of a {matrix_shape} matrix using "
                f"{approx_diag_num_samples} samples."
            )
        matrix_diag = approx_diag(
                closure,
                size=size,
                num_samples=approx_diag_num_samples,
                batch_size=batch_size,
                dtype=dtype,
                device=device)
    matrix_diag = matrix_diag.to(device=device)
    # Store the term to be subtracted from the diagonal separately in `matrix_diag_minuend`, so
    # we can easily replace potentially approximate diagonal values (in `matrix_diag`) with
    # exact ones when populating L[m, m]
    matrix_diag_minuend = torch.zeros_like(matrix_diag)

    # Make sure max_iter isn't bigger than the matrix
    max_iter = min(max_iter, matrix_shape[-1])

    # What we're returning
    L = torch.zeros(max_iter, matrix_shape[-1], dtype=dtype, device=device)
    orig_error = torch.max(matrix_diag, dim=-1)[0]
    errors = torch.norm(matrix_diag, 1, dim=-1) / orig_error

    # The permutation
    permutation = torch.arange(0, matrix_shape[-1], dtype=torch.long, device=matrix_diag.device)

    if verbose:
        print(
            f"Running Pivoted Cholesky on a {matrix_shape} matrix for {max_iter} iterations."
        )

    m = 0
    with tqdm(total=max_iter, desc='pivoted_cholesky', miniters=max_iter//100) as pbar:
        while (m == 0) or (m < max_iter and torch.max(errors) > error_tol):
            # Get the maximum diagonal value and index
            # This will serve as the next diagonal entry of the Cholesky,
            # as well as the next entry in the permutation matrix
            permuted_diags = torch.gather(matrix_diag - matrix_diag_minuend, -1, permutation[m:])
            max_diag_values, max_diag_indices = torch.max(permuted_diags, -1)
            max_diag_indices = max_diag_indices + m

            # Swap pi_m and pi_i in each row, where pi_i is the element of the permutation
            # corresponding to the max diagonal element
            old_pi_m = permutation[m].clone()
            permutation[m].copy_(
                    permutation.gather(-1, max_diag_indices.unsqueeze(-1)).squeeze_(-1))
            permutation.scatter_(-1, max_diag_indices.unsqueeze(-1), old_pi_m.unsqueeze(-1))
            pi_m = permutation[m].contiguous()

            row = None  # can potentially re-use row
            if recompute_max_diag_values:
                e_pi_m = torch.zeros((matrix_shape[-1], 1), dtype=dtype, device=device)
                e_pi_m[pi_m] = 1.
                row = closure(e_pi_m).squeeze(1)
                max_diag_values_to_scatter = row[pi_m] - matrix_diag_minuend[pi_m]
            else:
                max_diag_values_to_scatter = max_diag_values

            # Populate L[m, m] with the sqrt of the max diagonal element
            L_m = L[m, :]  # Will be all zeros -- should we use torch.zeros?
            L_m.scatter_(-1, pi_m.unsqueeze(-1), max_diag_values_to_scatter.sqrt().unsqueeze_(-1))

            # Populater L[m:, m] with L[m:, m] * L[m, m].sqrt()
            if m + 1 < matrix_shape[-1]:
                # Get next row of the permuted matrix
                if row is None:
                    e_pi_m = torch.zeros((matrix_shape[-1], 1), dtype=dtype, device=device)
                    e_pi_m[pi_m] = 1.
                    row = closure(e_pi_m).squeeze(1)
                pi_i = permutation[m + 1 :].contiguous()

                L_m_new = row.gather(-1, pi_i)
                if m > 0:
                    L_prev = L[:m, :].gather(-1, pi_i.unsqueeze(-2).repeat(m, 1))
                    update = L[:m, :].gather(
                        -1, pi_m.view(*pi_m.shape, 1, 1).repeat(m, 1)
                    )
                    L_m_new -= torch.sum(update * L_prev, dim=-2)

                L_m_new /= L_m.gather(-1, pi_m.unsqueeze(-1))
                L_m.scatter_(-1, pi_i, L_m_new)

                matrix_diag_minuend_current = matrix_diag_minuend.gather(-1, pi_i)
                matrix_diag_minuend.scatter_(-1, pi_i, matrix_diag_minuend_current + L_m_new**2)
                L[m, :] = L_m

                # Keep track of errors - for potential early stopping
                errors = torch.norm((
                        matrix_diag - matrix_diag_minuend).gather(-1, pi_i), 1, dim=-1) / orig_error

            m = m + 1
            pbar.update(1)

    return L[:m, :].mT.contiguous(), permutation
