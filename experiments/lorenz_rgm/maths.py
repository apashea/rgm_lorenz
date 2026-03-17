# maths.py
"""
Numerical and tensor utilities for the Lorenz RGM example.

This module adapts the most relevant parts of pymdp.jax.maths for our
RGM/Lorenz setting:
- stable entropy / cross-entropy / log
- factorized tensor contractions (via opt_einsum)
- log-likelihood and free energy for simple discrete models
- Dirichlet expectation utilities

It is intentionally lightweight and can later be extended or replaced
by a more general RGM maths module.
"""

from functools import partial
from typing import Optional, Tuple, List

import jax
import jax.numpy as jnp
from jax import nn, lax, tree_util
from jax.scipy.special import xlogy
from opt_einsum import contract

MINVAL = jnp.finfo(float).eps

# -----------------------------------------------------------------------------
# 1. Stable scalar functions
# -----------------------------------------------------------------------------

def stable_xlogx(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute x * log(x) with clipping to avoid NaNs when x ~ 0.
    """
    return xlogy(x, jnp.clip(x, MINVAL))


def stable_entropy(x: jnp.ndarray) -> jnp.ndarray:
    """
    Shannon entropy H[x] = - sum_i x_i log x_i
    """
    return - stable_xlogx(x).sum()


def stable_cross_entropy(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Cross-entropy H(x, y) = - sum_i x_i log y_i
    """
    return - xlogy(x, jnp.clip(y, MINVAL)).sum()


def log_stable(x: jnp.ndarray) -> jnp.ndarray:
    """
    Log with lower bound to avoid -inf.
    """
    return jnp.log(jnp.clip(x, a_min=MINVAL))

# -----------------------------------------------------------------------------
# 2. Factorized tensor contractions
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=['keep_dims'])
def factor_dot(
    M: jnp.ndarray,
    xs: List[jnp.ndarray],
    keep_dims: Optional[Tuple[int]] = None
) -> jnp.ndarray:
    """
    Dot product of a multidimensional array M with list of factors xs.

    M.ndim == len(xs) + len(keep_dims)
    xs[f] is contracted along dims specified implicitly by M layout.
    """
    d = len(keep_dims) if keep_dims is not None else 0
    assert M.ndim == len(xs) + d
    keep_dims = () if keep_dims is None else keep_dims
    dims = tuple((i,) for i in range(M.ndim) if i not in keep_dims)
    return factor_dot_flex(M, xs, dims, keep_dims=keep_dims)


@partial(jax.jit, static_argnames=['dims', 'keep_dims'])
def factor_dot_flex(
    M: jnp.ndarray,
    xs: List[jnp.ndarray],
    dims: List[Tuple[int]],
    keep_dims: Optional[Tuple[int]] = None
) -> jnp.ndarray:
    """
    Flexible factorized dot product using opt_einsum.

    Parameters
    ----------
    M : array
        High-dimensional tensor.
    xs : list of arrays
        Factor arrays.
    dims : list of tuples
        For each factor xs[f], dims[f] lists the axes in M where it applies.
    keep_dims : tuple of ints
        Axes in M to keep in the output.

    Returns
    -------
    Y : array
        Result of contracting M with xs, leaving keep_dims.
    """
    all_dims = tuple(range(M.ndim))
    matrix = [[xs[f], dims[f]] for f in range(len(xs))]
    args = [M, all_dims]
    for row in matrix:
        args.extend(row)

    keep_dims = () if keep_dims is None else keep_dims
    args += [keep_dims]
    return contract(*args, backend='jax')

# -----------------------------------------------------------------------------
# 3. Likelihoods and free energy (simple discrete case)
# -----------------------------------------------------------------------------

def get_likelihood_single_modality(
    o_m: jnp.ndarray,
    A_m: jnp.ndarray,
    distr_obs: bool = True
) -> jnp.ndarray:
    """
    Return observation likelihood P(o_m | s) for a single modality.

    o_m:
      - if distr_obs=True: probability vector over outcomes (O,)
      - else: integer index of outcome
    A_m: (S, O) or (O, S) depending on convention; here we assume (S, O)
    """
    if distr_obs:
        # o_m is (O,), A_m is (S, O)
        expanded_obs = jnp.expand_dims(o_m, axis=0)  # (1, O)
        likelihood = (expanded_obs * A_m).sum(axis=1)  # (S,)
    else:
        # o_m is scalar index
        likelihood = A_m[:, o_m]
    return likelihood


def compute_log_likelihood_single_modality(
    o_m: jnp.ndarray,
    A_m: jnp.ndarray,
    distr_obs: bool = True
) -> jnp.ndarray:
    """
    Compute log P(o_m | s) for a single modality.
    """
    return log_stable(get_likelihood_single_modality(o_m, A_m, distr_obs=distr_obs))


def compute_log_likelihood(
    obs: List[jnp.ndarray],
    A: List[jnp.ndarray],
    distr_obs: bool = True
) -> jnp.ndarray:
    """
    Compute log-likelihood over hidden states across observation modalities.

    obs: list of modality observations
    A: list of modality-specific likelihood arrays
    Returns:
      ll: (S,) log-likelihood over hidden states
    """
    result = tree_util.tree_map(
        lambda o, a: compute_log_likelihood_single_modality(o, a, distr_obs=distr_obs),
        obs,
        A
    )

    ll = jnp.sum(jnp.stack(result, axis=0), axis=0)
    return ll


def compute_accuracy(
    qs: List[jnp.ndarray],
    obs: List[jnp.ndarray],
    A: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    Accuracy term (expected log-likelihood under Q(s)).
    """
    log_likelihood = compute_log_likelihood(obs, A)
    x = qs[0]
    for q in qs[1:]:
        x = jnp.expand_dims(x, -1) * q
    joint = log_likelihood * x
    return joint.sum()


def compute_free_energy(
    qs: List[jnp.ndarray],
    prior: List[jnp.ndarray],
    obs: List[jnp.ndarray],
    A: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    Variational free energy:

      F = E_q[log q(s)] - E_q[log p(s)] - E_q[log p(o|s)]

    Using:
      -H[Q] + H_{Q}[P] - accuracy
    """
    vfe = 0.0
    for q, p in zip(qs, prior):
        negH_q = - stable_entropy(q)
        xH_qp = stable_cross_entropy(q, p)
        vfe += (negH_q + xH_qp)

    vfe -= compute_accuracy(qs, obs, A)
    return vfe

# -----------------------------------------------------------------------------
# 4. Dirichlet utilities
# -----------------------------------------------------------------------------

def spm_wnorm(A: jnp.ndarray) -> jnp.ndarray:
    """
    Returns a simple approximation to E[log Dirichlet] over columns of A.

    This is a lightweight analogue of SPM's wnorm function.
    """
    norm = 1.0 / A.sum(axis=0)
    avg = 1.0 / (A + MINVAL)
    wA = norm - avg
    return wA


def dirichlet_expected_value(dir_arr: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the expected value of Dirichlet parameters over a set of
    categorical distributions stored in columns of dir_arr.

    E[theta] = alpha / sum(alpha)
    """
    dir_arr = jnp.clip(dir_arr, a_min=MINVAL)
    expected_val = dir_arr / dir_arr.sum(axis=0, keepdims=True)
    return expected_val
