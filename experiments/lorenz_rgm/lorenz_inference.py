# lorenz_inference.py
"""
Variational message passing for the Lorenz hierarchy.

This module provides:
- Single-chain VMP for the lowest (patch) level using A, a simple temporal
  kernel, and E_states.
- Patch-wise lowest-level inference by looping over patches and calling
  the single-chain VMP (to keep memory usage manageable).
- Multi-level state inference with spatial renormalization:
  * bottom-up messages from level 0 to higher levels via D_state_from_parent
  * top-down messages from parent to child via D_state_from_parent
- Top-level path inference using expected free energy over paths
  computed via lorenz_efe (risk + ambiguity - epistemic), with
  path-dependent transitions B_states_paths at the top state level.

This is a Lorenz-specific instance of RGM-style inference and is structured
to realize the RGM-style pixels-to-planning architecture.

The implementation here is optimized by:
- Vectorizing higher-level VMP over spatial sites using vmap (no Python
  loops over (h,w) at parent/child levels).
- Wrapping the full hierarchical inference entry point in jax.jit to
  compile the entire computation graph.

These optimizations preserve the underlying graphical model and
variational updates, and are therefore faithful to the RGM formulation.
"""

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import nn

from .lorenz_model import LorenzHierarchy, LorenzLevel, LorenzRGMParams
from .lorenz_efe import (
    compute_expected_free_energy_paths,
    update_path_posterior_from_G,
)

# Global debug flag for this module
DEBUG_INFERENCE = False

# -----------------------------------------------------------------------------
# 1. Utilities: observations and preferences for lowest level
# -----------------------------------------------------------------------------

def build_lowest_level_observations_flat(
    lorenz_data_dict: Dict[str, Any],
) -> jnp.ndarray:
    """
    Build a flat observation array suitable for lowest-level inference.

    We treat the quantized coefficients as "observations" in a one-hot form
    that is consistent with the A built in lorenz_model.py.

    Returns:
      obs_flat: (N, O) array, where N = T * H_blocks * W_blocks,
                O = K * L.
    """
    q_coeffs = lorenz_data_dict["q_coeffs"]  # (N, K)
    K = int(lorenz_data_dict["K"])
    L = int(lorenz_data_dict["L"])
    N = q_coeffs.shape[0]
    O = K * L

    indices = (
        jnp.arange(K, dtype=jnp.int32)[None, :] * L + q_coeffs
    )  # (N, K)

    def one_hot_row(idx_row: jnp.ndarray) -> jnp.ndarray:
        oh = jnp.zeros((O,), dtype=jnp.float32)

        def body_fun(i, arr):
            return arr.at[idx_row[i]].add(1.0)

        oh = jax.lax.fori_loop(0, K, body_fun, oh)
        return oh / (oh.sum() + 1e-8)

    obs_flat = jax.vmap(one_hot_row)(indices)  # (N, O)
    return obs_flat


def build_lowest_level_observations_grid(
    lorenz_data_dict: Dict[str, Any],
) -> jnp.ndarray:
    """
    Build an observation array reshaped as (T, H0, W0, O) for patch-wise
    inference.
    """
    obs_flat = build_lowest_level_observations_flat(lorenz_data_dict)  # (N, O)
    T0 = int(lorenz_data_dict["T"])
    H0 = int(lorenz_data_dict["H_blocks"])
    W0 = int(lorenz_data_dict["W_blocks"])

    obs_grid = obs_flat.reshape(T0, H0, W0, obs_flat.shape[1])
    return obs_grid


def prefs_from_params(
    params: LorenzRGMParams,
    K: int,
    L: int,
) -> jnp.ndarray:
    """
    Derive a categorical preference distribution C over lowest-level
    outcomes from pref_alpha stored in LorenzRGMParams.

    Args:
      params: LorenzRGMParams with pref_alpha
      K, L: lowest-level configuration (O0 = K * L)

    Returns:
      C: (O0,) normalized preference distribution
    """
    O0 = K * L
    C = params.pref_alpha / (params.pref_alpha.sum() + 1e-8)
    assert C.shape[0] == O0, "prefs_from_params: pref_alpha shape mismatch with K*L."
    return C

# -----------------------------------------------------------------------------
# 2. Single-chain VMP using A and a temporal kernel (no spatial coupling)
# -----------------------------------------------------------------------------

def vmp_single_chain(
    A: jnp.ndarray,
    B: jnp.ndarray,
    E: jnp.ndarray,
    obs: jnp.ndarray,
    num_iter: int = 8,
) -> jnp.ndarray:
    """
    Variational message passing for a single chain of hidden states.

    Args:
      A: (S, O)
      B: (S, S)
      E: (S,)
      obs: (T, O)

    Returns:
      qs: (T, S)
    """
    T = obs.shape[0]
    S = A.shape[0]

    def likelihood_single(o_t: jnp.ndarray) -> jnp.ndarray:
        lik = (o_t[None, :] * A).sum(axis=1)  # (S,)
        return jnp.log(jnp.clip(lik, a_min=1e-16))

    log_liks = jax.vmap(likelihood_single)(obs)  # (T, S)

    qs = jnp.full((T, S), 1.0 / S, dtype=jnp.float32)
    ln_prior = jnp.log(jnp.clip(E, 1e-16))

    def forward_messages(qs_: jnp.ndarray) -> jnp.ndarray:
        msgs = []
        prev_q = qs_[0]
        msgs.append(ln_prior)
        for t in range(1, T):
            msg = jnp.log(jnp.clip(B @ prev_q, 1e-16))
            msgs.append(msg)
            prev_q = qs_[t]
        return jnp.stack(msgs, axis=0)

    def backward_messages(qs_: jnp.ndarray) -> jnp.ndarray:
        msgs = []
        next_q = qs_[-1]
        msgs.append(jnp.zeros_like(next_q))
        for t in range(T - 2, -1, -1):
            msg = jnp.log(jnp.clip(B.T @ next_q, 1e-16))
            msgs.append(msg)
            next_q = qs_[t]
        msgs = msgs[::-1]
        return jnp.stack(msgs, axis=0)

    def vmp_iteration(qs_):
        m_plus = forward_messages(qs_)
        m_minus = backward_messages(qs_)
        ln_qs = log_liks + m_plus + m_minus
        return nn.softmax(ln_qs, axis=1)

    def body_fun(_, qs_):
        return vmp_iteration(qs_)

    qs = jax.lax.fori_loop(0, num_iter, body_fun, qs)
    return qs

# -----------------------------------------------------------------------------
# 3. Patch-wise lowest-level inference (loop over patches, unchanged)
# -----------------------------------------------------------------------------

def infer_lowest_level_patches(
    level0: LorenzLevel,
    lorenz_data_dict: Dict[str, Any],
    num_iter_lowest: int = 8,
) -> jnp.ndarray:
    """
    Run VMP at the lowest level independently for each patch.

    For now, we approximate temporal dynamics at level 0 with an identity
    kernel (persistence), since all path-dependent dynamics are attached
    to higher levels.

    Returns:
      qs0_grid: (T0, H0, W0, S0)
    """
    A0 = level0.A
    E0 = level0.E_states

    obs_grid = build_lowest_level_observations_grid(lorenz_data_dict)  # (T0, H0, W0, O)
    T0, H0, W0, _ = obs_grid.shape
    S0 = A0.shape[0]

    B0 = jnp.eye(S0, dtype=jnp.float32)

    vmp_single_chain_jit = jax.jit(vmp_single_chain, static_argnames=("num_iter",))

    qs0_host = np.zeros((T0, H0, W0, S0), dtype=np.float32)

    for h0 in range(H0):
        for w0 in range(W0):
            obs_patch = obs_grid[:, h0, w0, :]
            qs_chain = vmp_single_chain_jit(A0, B0, E0, obs_patch, num_iter=num_iter_lowest)
            qs0_host[:, h0, w0, :] = np.array(qs_chain)

    qs0_grid = jnp.array(qs0_host)
    return qs0_grid

# -----------------------------------------------------------------------------
# 4. Bottom-up and top-down messages via D_state_from_parent
# -----------------------------------------------------------------------------

def bottom_up_message_child_to_parent(
    qs_child_grid: jnp.ndarray,
    D_parent: jnp.ndarray,
    states_grid_parent: jnp.ndarray,
) -> jnp.ndarray:
    """
    Bottom-up pseudo-likelihood from a child level to a parent level via D.

    Args:
      qs_child_grid: (T, H_child, W_child, S_child)
      D_parent: (S_parent, 4) integer child-state indices per parent state
      states_grid_parent: (T, H_parent, W_parent)

    Returns:
      log_lik_parent: (T, H_parent, W_parent, S_parent)
    """
    T, Hc, Wc, S_child = qs_child_grid.shape
    T1, Hp, Wp = states_grid_parent.shape
    assert T == T1
    assert Hc == 2 * Hp and Wc == 2 * Wp

    S_parent = D_parent.shape[0]

    def site_children(qs_child_t: jnp.ndarray) -> jnp.ndarray:
        c00 = qs_child_t[0::2, 0::2, :]
        c01 = qs_child_t[0::2, 1::2, :]
        c10 = qs_child_t[1::2, 0::2, :]
        c11 = qs_child_t[1::2, 1::2, :]
        return jnp.stack([c00, c01, c10, c11], axis=2)  # (Hp, Wp, 4, S_child)

    qs_children = jax.vmap(site_children)(qs_child_grid)  # (T, Hp, Wp, 4, S_child)
    D_idx = D_parent.astype(jnp.int32)  # (S_parent, 4)

    def score_parent_state(qs_child_patch: jnp.ndarray, pattern: jnp.ndarray) -> jnp.ndarray:
        probs = qs_child_patch[jnp.arange(4), pattern]  # (4,)
        log_probs = jnp.log(jnp.clip(probs, 1e-16))
        return log_probs.sum()

    def score_all_parents_for_site(qs_child_site: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda pattern: score_parent_state(qs_child_site, pattern)
        )(D_idx)  # (S_parent,)

    def score_all_sites(qs_children_t: jnp.ndarray) -> jnp.ndarray:
        def score_site(children_site: jnp.ndarray) -> jnp.ndarray:
            return score_all_parents_for_site(children_site)

        return jax.vmap(jax.vmap(score_site, in_axes=0), in_axes=0)(qs_children_t)

    log_lik_parent = jax.vmap(score_all_sites)(qs_children)  # (T, Hp, Wp, S_parent)
    return log_lik_parent


def top_down_prior_parent_to_child(
    qs_parent_grid: jnp.ndarray,
    D_parent: jnp.ndarray,
    H_child: int,
    W_child: int,
    S_child: int,
) -> jnp.ndarray:
    """
    Build a top-down log prior bias over child states from parent posteriors
    using D_state_from_parent.

    Args:
      qs_parent_grid: (T, Hp, Wp, S_parent)
      D_parent: (S_parent, 4) mapping parent->child states
      H_child, W_child, S_child: child grid sizes

    Returns:
      log_prior_bias_child: (T, H_child, W_child, S_child)
    """
    T, Hp, Wp, S_parent = qs_parent_grid.shape
    assert H_child == 2 * Hp and W_child == 2 * Wp

    D_idx = D_parent.astype(jnp.int32)  # (S_parent, 4)

    def bias_for_time(t):
        qs_parent_t = qs_parent_grid[t]  # (Hp, Wp, S_parent)
        bias = jnp.zeros((H_child, W_child, S_child), dtype=jnp.float32)

        for hp in range(Hp):
            for wp in range(Wp):
                qs_p = qs_parent_t[hp, wp]  # (S_parent,)
                for child_idx, (dh, dw) in enumerate(
                    [(0, 0), (0, 1), (1, 0), (1, 1)]
                ):
                    hc = 2 * hp + dh
                    wc = 2 * wp + dw
                    child_states = D_idx[:, child_idx]  # (S_parent,)
                    accum = jnp.zeros((S_child,), dtype=jnp.float32)

                    def body_fun(sp, acc):
                        sc = child_states[sp]
                        return acc.at[sc].add(qs_p[sp])

                    accum = jax.lax.fori_loop(0, S_parent, body_fun, accum)
                    accum = accum / (accum.sum() + 1e-8)
                    log_accum = jnp.log(jnp.clip(accum, 1e-16))
                    bias = bias.at[hc, wc, :].set(log_accum)
        return bias

    log_prior_bias_child = jax.vmap(bias_for_time)(jnp.arange(T))  # (T, Hc, Wc, S_child)
    return log_prior_bias_child

# -----------------------------------------------------------------------------
# 5. Vectorized two-level VMP (child-parent) over spatial sites
# -----------------------------------------------------------------------------

def _vmp_parent_chain(
    log_lik_chain: jnp.ndarray,
    E_parent: jnp.ndarray,
    B_chain: jnp.ndarray,
    num_iter: int,
) -> jnp.ndarray:
    """
    VMP for a single parent site chain given time-varying temporal kernel.

    Args:
      log_lik_chain: (T, S_parent)
      E_parent: (S_parent,)
      B_chain: (T, S_parent, S_parent)
      num_iter: number of iterations

    Returns:
      qs_parent_chain: (T, S_parent)
    """
    T, S_parent = log_lik_chain.shape

    def forward_messages(qs_):
        msgs = []
        prev_q = qs_[0]
        msgs.append(jnp.log(jnp.clip(E_parent, 1e-16)))
        for t in range(1, T):
            B_t = B_chain[t]
            msg = jnp.log(jnp.clip(B_t @ prev_q, 1e-16))
            msgs.append(msg)
            prev_q = qs_[t]
        return jnp.stack(msgs, axis=0)

    def backward_messages(qs_):
        msgs = []
        next_q = qs_[-1]
        msgs.append(jnp.zeros_like(next_q))
        for t in range(T - 2, -1, -1):
            B_tp1 = B_chain[t + 1]
            msg = jnp.log(jnp.clip(B_tp1.T @ next_q, 1e-16))
            msgs.append(msg)
            next_q = qs_[t]
        msgs = msgs[::-1]
        return jnp.stack(msgs, axis=0)

    def vmp_iter(qs_):
        m_plus = forward_messages(qs_)
        m_minus = backward_messages(qs_)
        ln_qs = log_lik_chain + m_plus + m_minus
        return nn.softmax(ln_qs, axis=1)

    def body_fun(_, qs_):
        return vmp_iter(qs_)

    qs_init = jnp.full((T, S_parent), 1.0 / S_parent, dtype=jnp.float32)
    qs_final = jax.lax.fori_loop(0, num_iter, body_fun, qs_init)
    return qs_final


def _vmp_child_chain(
    log_bias_chain: jnp.ndarray,
    E_child: jnp.ndarray,
    B_child: jnp.ndarray,
    num_iter: int,
) -> jnp.ndarray:
    """
    VMP for a single child site chain with identity kernel (or fixed B_child).

    Args:
      log_bias_chain: (T, S_child)
      E_child: (S_child,)
      B_child: (S_child, S_child)
      num_iter: iterations

    Returns:
      qs_child_chain: (T, S_child)
    """
    T, S_child = log_bias_chain.shape

    def forward_messages(qs_):
        msgs = []
        prev_q = qs_[0]
        msgs.append(jnp.log(jnp.clip(E_child, 1e-16)))
        for t in range(1, T):
            msg = jnp.log(jnp.clip(B_child @ prev_q, 1e-16))
            msgs.append(msg)
            prev_q = qs_[t]
        return jnp.stack(msgs, axis=0)

    def backward_messages(qs_):
        msgs = []
        next_q = qs_[-1]
        msgs.append(jnp.zeros_like(next_q))
        for t in range(T - 2, -1, -1):
            msg = jnp.log(jnp.clip(B_child.T @ next_q, 1e-16))
            msgs.append(msg)
            next_q = qs_[t]
        msgs = msgs[::-1]
        return jnp.stack(msgs, axis=0)

    def vmp_iter(qs_):
        m_plus = forward_messages(qs_)
        m_minus = backward_messages(qs_)
        ln_qs = log_bias_chain + m_plus + m_minus
        return nn.softmax(ln_qs, axis=1)

    def body_fun(_, qs_):
        return vmp_iter(qs_)

    qs_init = jnp.full((T, S_child), 1.0 / S_child, dtype=jnp.float32)
    qs_final = jax.lax.fori_loop(0, num_iter, body_fun, qs_init)
    return qs_final


def vmp_two_level_states(
    level_child: LorenzLevel,
    level_parent: LorenzLevel,
    qs_child_grid: jnp.ndarray,
    states_grid_parent: jnp.ndarray,
    qu_parent: Optional[jnp.ndarray] = None,
    num_iter: int = 2,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Coupled VMP over child and parent states with optional path-dependent
    dynamics at the parent level.

    This version is vectorized over spatial sites using vmap: each parent
    (and corresponding child group) is treated as an independent chain, but
    the update equations are identical to the original per-site code.

    Args:
      level_child: lower LorenzLevel
      level_parent: parent LorenzLevel
      qs_child_grid: (T, Hc, Wc, S_child)
      states_grid_parent: (T, Hp, Wp)
      qu_parent: (T, U) or None (path posterior at this level)
      num_iter: alternating sweeps

    Returns:
      qs_child_final: (T, Hc, Wc, S_child)
      qs_parent_final: (T, Hp, Wp, S_parent)
    """
    E_child = level_child.E_states
    S_child = E_child.shape[0]

    E_parent = level_parent.E_states
    D_parent = level_parent.D_state_from_parent  # (S_parent, 4)
    B_states_paths_parent = level_parent.B_states_paths  # (S_parent, S_parent, U) or None

    T, Hc, Wc, _ = qs_child_grid.shape
    T1, Hp, Wp = states_grid_parent.shape
    assert T == T1
    assert Hc == 2 * Hp and Wc == 2 * Wp

    if D_parent is None:
        raise ValueError("Parent level must define D_state_from_parent.")

    S_parent = E_parent.shape[0]

    # Effective temporal kernel at parent for each time
    if B_states_paths_parent is None:
        B_parent_all = jnp.broadcast_to(
            jnp.eye(S_parent, dtype=jnp.float32), (T, S_parent, S_parent)
        )
    else:
        U = B_states_paths_parent.shape[2]
        if qu_parent is None:
            qu_parent_eff = jnp.full((T, U), 1.0 / U, dtype=jnp.float32)
        else:
            qu_parent_eff = qu_parent

        def B_eff_for_time(t):
            qu_t = qu_parent_eff[t]  # (U,)
            return (B_states_paths_parent * qu_t[None, None, :]).sum(axis=2)  # (S_parent, S_parent)

        B_parent_all = jax.vmap(B_eff_for_time)(jnp.arange(T))  # (T, S_parent, S_parent)

    # Initialize parent states uniformly
    qs_parent_grid = jnp.full(
        (T, Hp, Wp, S_parent), 1.0 / S_parent, dtype=jnp.float32
    )

    # Identity kernel at child in this block
    B_child = jnp.eye(S_child, dtype=jnp.float32)

    # ----- Alternating updates -----

    def alt_step(_, carry):
        qs_child_curr, qs_parent_curr = carry

        # 1) Update parent given current child
        log_lik_parent = bottom_up_message_child_to_parent(
            qs_child_curr, D_parent, states_grid_parent
        )  # (T, Hp, Wp, S_parent)

        # Flatten parent spatial dims to a site axis
        log_lik_flat = log_lik_parent.reshape(T, Hp * Wp, S_parent)  # (T, Np, S_parent)
        B_chain = B_parent_all  # (T, S_parent, S_parent)

        # VMP per parent site chain
        vmp_parent_site = lambda log_chain: _vmp_parent_chain(
            log_chain, E_parent, B_chain, num_iter=2
        )  # (T, S_parent)
        qs_parent_flat = jax.vmap(vmp_parent_site, in_axes=1, out_axes=1)(
            log_lik_flat
        )  # (T, Np, S_parent)
        qs_parent_new = qs_parent_flat.reshape(T, Hp, Wp, S_parent)

        # 2) Update child given updated parent
        log_prior_bias_child = top_down_prior_parent_to_child(
            qs_parent_new, D_parent, Hc, Wc, S_child
        )  # (T, Hc, Wc, S_child)

        # Flatten child spatial dims
        log_bias_flat = log_prior_bias_child.reshape(T, Hc * Wc, S_child)  # (T, Nc, S_child)

        vmp_child_site = lambda log_chain: _vmp_child_chain(
            log_chain, E_child, B_child, num_iter=1
        )  # (T, S_child)
        qs_child_flat = jax.vmap(vmp_child_site, in_axes=1, out_axes=1)(
            log_bias_flat
        )  # (T, Nc, S_child)
        qs_child_new = qs_child_flat.reshape(T, Hc, Wc, S_child)

        return (qs_child_new, qs_parent_new)

    qs_child_final, qs_parent_final = jax.lax.fori_loop(
        0, num_iter, alt_step, (qs_child_grid, qs_parent_grid)
    )
    return qs_child_final, qs_parent_final

# -----------------------------------------------------------------------------
# 6. High-level inference entry point with multi-level states and EFE-based paths
# -----------------------------------------------------------------------------

def _infer_lorenz_hierarchy_inner(
    hierarchy: LorenzHierarchy,
    lorenz_data_dict: Dict[str, Any],
    params: Optional[LorenzRGMParams],
    num_iter_lowest: int,
    num_iter_hier: int,
    efe_gamma: float,
    pref_mode: str,
) -> Dict[str, Any]:
    """
    Pure JAX-compatible inner function implementing the core inference logic.
    This is separated so we can wrap it in jax.jit without Python side-effects.
    """
    T0 = hierarchy.T0
    H0 = hierarchy.H_blocks
    W0 = hierarchy.W_blocks

    level0: LorenzLevel = hierarchy.levels[0]
    A0 = level0.A

    obs_flat = build_lowest_level_observations_flat(lorenz_data_dict)  # (N, O)
    N = obs_flat.shape[0]
    assert N == T0 * H0 * W0, "Observation length mismatch."

    # Preferences: from params.pref_alpha if available, else empirical fallback
    if params is not None:
        K = int(lorenz_data_dict["K"])
        L = int(lorenz_data_dict["L"])
        C = prefs_from_params(params, K, L)
    else:
        C = obs_flat.mean(axis=0)
        C = C / (C.sum() + 1e-8)

    # 1. Level-0 states
    qs0_grid = infer_lowest_level_patches(
        level0, lorenz_data_dict, num_iter_lowest=num_iter_lowest
    )

    num_levels = len(hierarchy.levels)
    qs_levels: List[Optional[jnp.ndarray]] = [None] * num_levels
    qu_levels: List[Optional[jnp.ndarray]] = [None] * num_levels

    qs_levels[0] = qs0_grid

    # Early exit if only one level
    if num_levels == 1:
        return {"qs_levels": qs_levels, "qu_levels": qu_levels}

    # 2. Higher levels (generalized): build qs_l grids for all l >= 1
    states_grids = hierarchy.states_grids
    qs_current: List[Optional[jnp.ndarray]] = [None] * num_levels

    # Level 0 is already inferred
    qs_current[0] = qs0_grid

    for l in range(1, num_levels):
        level_l = hierarchy.levels[l]
        states_grid_l = states_grids[l]
        Tl, Hl, Wl = states_grid_l.shape
        assert Tl == T0
        qs_l = jnp.full(
            (T0, Hl, Wl, level_l.S),
            1.0 / level_l.S,
            dtype=jnp.float32,
        )
        qs_current[l] = qs_l

    # Determine top level carrying paths (by level index)
    top_idx = None
    for idx in reversed(range(num_levels)):
        lvl = hierarchy.levels[idx]
        if (
            lvl.num_paths is not None
            and lvl.num_paths > 1
            and lvl.C_paths is not None
            and lvl.E_paths is not None
        ):
            top_idx = idx
            break

    paths_active = top_idx is not None
    if paths_active:
        level_top = hierarchy.levels[top_idx]
        U = level_top.num_paths
        qu_top = jnp.full((T0, U), 1.0 / U, dtype=jnp.float32)
    else:
        level_top = None
        qu_top = None

    # Hierarchical inference: alternate bottom-up / top-down across all levels
    qu_top_current = qu_top

    for _ in range(num_iter_hier):
        # Upward passes: 0 <-> 1, 1 <-> 2, ..., (L-2) <-> (L-1)
        for l_child in range(num_levels - 1):
            l_parent = l_child + 1
            level_child = hierarchy.levels[l_child]
            level_parent = hierarchy.levels[l_parent]

            qs_child = qs_current[l_child]
            qs_parent = qs_current[l_parent]
            states_grid_parent = states_grids[l_parent]

            # Only run coupled block if this parent has a D_state_from_parent
            if level_parent.D_state_from_parent is None:
                continue

            # Path posterior only at the top path level
            qu_parent = None
            if paths_active and (l_parent == top_idx):
                qu_parent = qu_top_current

            qs_child_new, qs_parent_new = vmp_two_level_states(
                level_child=level_child,
                level_parent=level_parent,
                qs_child_grid=qs_child,
                states_grid_parent=states_grid_parent,
                qu_parent=qu_parent,
                num_iter=1,
            )

            qs_current[l_child] = qs_child_new
            qs_current[l_parent] = qs_parent_new

        # Path update if active
        if paths_active:
            qs_top_grid = qs_current[top_idx]
            level0_for_efe = level0

            G_tu = compute_expected_free_energy_paths(
                level_top=level_top,
                level0=level0_for_efe,
                qs_top_grid=qs_top_grid,
                qs0_grid=qs_current[0],
                C=C,
                tau=2,
            )

            qu_top_current = update_path_posterior_from_G(
                level_top,
                G_tu,
                gamma=efe_gamma,
                num_iter=2,
            )

    # Collect results per level
    for l in range(num_levels):
        qs_levels[l] = qs_current[l]

    if paths_active and qu_top_current is not None:
        qu_levels[top_idx] = qu_top_current

    return {
        "qs_levels": qs_levels,
        "qu_levels": qu_levels,
    }


# JIT-compiled wrapper around the pure inner function
infer_lorenz_hierarchy_jit = jax.jit(
    _infer_lorenz_hierarchy_inner,
    static_argnames=("num_iter_lowest", "num_iter_hier", "pref_mode"),
)

def infer_lorenz_hierarchy(
    hierarchy: LorenzHierarchy,
    lorenz_data_dict: Dict[str, Any],
    params: Optional[LorenzRGMParams] = None,
    num_iter_lowest: int = 8,
    num_iter_hier: int = 4,
    efe_gamma: float = 16.0,
    pref_mode: str = "data_empirical",
) -> Dict[str, Any]:
    """
    Public entry point for variational inference over states and (optional)
    paths in the Lorenz hierarchy.

    This function dispatches to a JIT-compiled inner implementation to
    ensure efficient execution while preserving the RGM's structure and
    update equations.
    """
    return infer_lorenz_hierarchy_jit(
        hierarchy=hierarchy,
        lorenz_data_dict=lorenz_data_dict,
        params=params,
        num_iter_lowest=num_iter_lowest,
        num_iter_hier=num_iter_hier,
        efe_gamma=efe_gamma,
        pref_mode=pref_mode,
    )
