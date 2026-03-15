# lorenz_inference.py
"""
Variational message passing for the Lorenz hierarchy.

This module provides:
  - A JAX-friendly inference routine that updates posterior beliefs over
    states at each spatial level in the LorenzHierarchy, given:
      * lowest-level observations (quantized patch codes)
      * current model parameters (A, B, D, E)
  - Free-energy computation for diagnostics.

This is a Lorenz-specific instance of the RGM-style inference and is
structured to be easily generalized to full RGM message passing later.
"""

from typing import List, Dict, Any, Tuple

import jax
import jax.numpy as jnp
from jax import nn

from .lorenz_model import LorenzHierarchy, LorenzLevel
from .lorenz_data import encode_quantized_coeffs
from . import maths as rgm_maths  # assume we adapt pymdp.maths to rgm.maths

# If you keep pymdp.jax.maths as a reference, you can temporarily import from there:
# from pymdp.jax.maths import compute_log_likelihood_single_modality as compute_ll_single


# -----------------------------------------------------------------------------
# 1. Utilities: observations and A-dependencies for lowest level
# -----------------------------------------------------------------------------

def build_lowest_level_observations(
    lorenz_data_dict: Dict[str, Any],
) -> jnp.ndarray:
    """
    Build an observation array suitable for lowest-level inference.

    For now, we treat the quantized coefficients as "observations" in a
    one-hot form that is consistent with the A built in lorenz_model.py.

    Returns:
        obs: (T * H_blocks * W_blocks, O) array, where O = K * L,
             each row is a one-hot (or distribution) over O.
    """
    q_coeffs = lorenz_data_dict["q_coeffs"]  # (N, K)
    K = int(lorenz_data_dict["K"])
    L = int(lorenz_data_dict["L"])
    N = q_coeffs.shape[0]
    O = K * L

    # Same encoding scheme as build_lowest_level_A
    indices = (
        jnp.arange(K, dtype=jnp.int32)[None, :] * L + q_coeffs
    )  # (N, K)

    def one_hot_row(idx_row: jnp.ndarray) -> jnp.ndarray:
        oh = jnp.zeros((O,), dtype=jnp.float32)
        def body_fun(i, arr):
            return arr.at[idx_row[i]].add(1.0)
        oh = jax.lax.fori_loop(0, K, body_fun, oh)
        return oh / (oh.sum() + 1e-8)

    obs = jax.vmap(one_hot_row)(indices)
    return obs  # (N, O)


# -----------------------------------------------------------------------------
# 2. Single-level VMP using A and B (no spatial coupling yet)
# -----------------------------------------------------------------------------

def vmp_single_level(
    A: jnp.ndarray,
    B: jnp.ndarray,
    E: jnp.ndarray,
    obs: jnp.ndarray,
    num_iter: int = 8,
) -> Tuple[jnp.ndarray, float]:
    """
    Variational message passing for a single chain of hidden states with:
      - emission model A (shared across time)
      - transition model B(s'|s)
      - prior over initial state E

    This is a simplified version of pymdp.jax.algos.run_vmp specialized
    to a 1D chain with one hidden factor and one observation modality.

    Args:
        A: (S, O) emission matrix P(o|s)
        B: (S, S) transition matrix P(s'|s)
        E: (S,) prior over s_0
        obs: (T, O) observations as one-hot/distributions over O
        num_iter: number of VMP iterations

    Returns:
        qs: (T, S) posterior marginals over states
        F: scalar free energy (approximation)
    """
    T = obs.shape[0]
    S = A.shape[0]

    # Precompute log-likelihoods ln P(o_t | s_t) for all t, s
    # likelihood_t[s] = sum_o obs[t, o] * A[s, o]
    # log_likelihoods: (T, S)
    def likelihood_single(o_t: jnp.ndarray) -> jnp.ndarray:
        # o_t: (O,)
        # A: (S, O)
        lik = (o_t[None, :] * A).sum(axis=1)  # (S,)
        return jnp.log(jnp.clip(lik, a_min=1e-16))

    log_liks = jax.vmap(likelihood_single)(obs)  # (T, S)

    # Initialize q(s_t) ~ uniform
    qs = jnp.full((T, S), 1.0 / S, dtype=jnp.float32)
    ln_prior = jnp.log(jnp.clip(E, 1e-16))
    ln_B = jnp.log(jnp.clip(B, 1e-16))

    def forward_messages(qs_: jnp.ndarray) -> jnp.ndarray:
        """
        Compute forward messages m_+(s_t) using B and prior.
        m_+(s_0) = ln_prior
        m_+(s_t) = ln sum_s' [ q(s'_{t-1}) * B(s_t | s'_{t-1}) ]
        """
        def step(carry, t):
            prev_q = carry  # (S,)
            # message to time t: log(B @ prev_q)
            msg = jnp.log(jnp.clip(B @ prev_q, 1e-16))
            return qs_[t], msg

        # First message is prior at t=0
        msgs = []
        prev_q = qs_[0]
        msgs.append(ln_prior)
        for t in range(1, T):
            msg = jnp.log(jnp.clip(B @ prev_q, 1e-16))
            msgs.append(msg)
            prev_q = qs_[t]
        return jnp.stack(msgs, axis=0)  # (T, S)

    def backward_messages(qs_: jnp.ndarray) -> jnp.ndarray:
        """
        Compute backward messages m_-(s_t) using B.
        m_-(s_T) = 0
        m_-(s_t) = ln sum_s' [ B(s' | s_t) * q(s'_{t+1}) ]
        """
        msgs = []
        next_q = qs_[-1]
        msgs.append(jnp.zeros_like(next_q))
        for t in range(T - 2, -1, -1):
            msg = jnp.log(jnp.clip(B.T @ next_q, 1e-16))
            msgs.append(msg)
            next_q = qs_[t]
        msgs = msgs[::-1]
        return jnp.stack(msgs, axis=0)  # (T, S)

    def vmp_iteration(qs_):
        m_plus = forward_messages(qs_)   # (T, S)
        m_minus = backward_messages(qs_) # (T, S)
        ln_qs = log_liks + m_plus + m_minus
        qs_next = nn.softmax(ln_qs, axis=1)
        return qs_next

    def body_fun(i, qs_):
        return vmp_iteration(qs_)

    qs = jax.lax.fori_loop(0, num_iter, body_fun, qs)

    # Free energy (simple approximation) using pymdp-style helper
    # We treat obs as one-hot distributions; A as emission model; prior as E.
    # Package obs into the "modality" format expected by compute_free_energy.
    obs_list = [obs]  # single modality
    A_list = [A]
    qs_list = [qs_t for qs_t in qs.T]  # treat each state as factor? simple hack

    # For now, we just compute a crude free energy ignoring transitions,
    # as a diagnostic. A full chain-based F would include transition terms.
    F = rgm_maths.compute_free_energy(qs_list, [E], obs_list, A_list)

    return qs, F


# -----------------------------------------------------------------------------
# 3. Multi-level Lorenz inference stub
# -----------------------------------------------------------------------------

def infer_lorenz_states(
    hierarchy: LorenzHierarchy,
    lorenz_data_dict: Dict[str, Any],
    num_iter: int = 8,
) -> Dict[str, Any]:
    """
    Run variational inference over states in the Lorenz hierarchy.

    Current simplifications:
      - We infer posteriors only at the lowest level using A and B,
        ignoring top-down influences from higher levels.
      - Higher levels are kept as placeholders; later we will add
        messages from D and possibly path factors.

    Args:
        hierarchy: LorenzHierarchy instance
        lorenz_data_dict: output of build_lorenz_patch_dataset (for obs)
        num_iter: number of VMP iterations

    Returns:
        results dict with:
          'qs_levels': list of posterior arrays per level:
               level 0: (T*H*W, S0) or (T, H, W, S0)
               higher levels: None for now
          'F_levels': list of free energies per level (level 0 only)
    """
    T = hierarchy.T
    H0 = hierarchy.H_blocks
    W0 = hierarchy.W_blocks

    # Level 0 model
    level0: LorenzLevel = hierarchy.levels[0]
    A0 = level0.A     # (S0, O)
    B0 = level0.B_states  # (S0, S0)
    E0 = level0.E_states  # (S0,)

    # Observations for level 0
    obs_flat = build_lowest_level_observations(lorenz_data_dict)  # (N, O)
    N = obs_flat.shape[0]
    S0 = A0.shape[0]

    # For now we treat the entire patch set as one long chain of length N.
    qs0, F0 = vmp_single_level(A0, B0, E0, obs_flat, num_iter=num_iter)

    # Optionally reshape qs0 to (T, H0, W0, S0)
    qs0_grid = qs0.reshape(T, H0, W0, S0)

    # Higher levels: placeholders for now
    num_levels = len(hierarchy.levels)
    qs_levels: List[Any] = [qs0_grid]
    F_levels: List[Any] = [F0]

    for l in range(1, num_levels):
        qs_levels.append(None)
        F_levels.append(None)

    return {
        "qs_levels": qs_levels,
        "F_levels": F_levels,
    }
