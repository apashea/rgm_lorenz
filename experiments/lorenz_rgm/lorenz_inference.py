# lorenz_inference.py
"""
Variational message passing for the Lorenz hierarchy.

This module provides:
- Single-chain VMP for the lowest (patch) level using A, a simple temporal
  kernel, and E_states.
- Patch-wise lowest-level inference by looping over patches and calling
  the single-chain VMP (to keep memory usage manageable).
- Multi-level state inference with spatial renormalization:
  * bottom-up messages from level 0 to level 1 via D_state_from_parent
  * bottom-up messages from level 1 to level 2 via D_state_from_parent
  * top-down messages from parent to child via D_state_from_parent
- Top-level path inference using expected free energy over paths
  computed via lorenz_efe (risk + ambiguity - epistemic), with
  path-dependent transitions B_states_paths at the top state level.

This is a Lorenz-specific instance of RGM-style inference and is structured
to realize the RGM-style pixels-to-planning architecture.
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
    O = obs_flat.shape[1]

    obs_grid = obs_flat.reshape(T0, H0, W0, O)
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
# 3. Patch-wise lowest-level inference (loop over patches)
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
# 5. Two-level VMP building block (child-parent)
# -----------------------------------------------------------------------------

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

    # Effective temporal kernel at parent
    if B_states_paths_parent is None:
        B_parent_all = jnp.broadcast_to(jnp.eye(S_parent, dtype=jnp.float32), (T, S_parent, S_parent))
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
    qs_parent_grid = jnp.full((T, Hp, Wp, S_parent), 1.0 / S_parent, dtype=jnp.float32)

    # ----- Alternating updates -----

    def update_parent(qs_child_curr, qs_parent_curr):
        # bottom-up from child
        log_lik_parent = bottom_up_message_child_to_parent(
            qs_child_curr, D_parent, states_grid_parent
        )  # (T, Hp, Wp, S_parent)

        def update_site_chain(
            qs_parent_chain: jnp.ndarray,
            log_lik_chain: jnp.ndarray,
            B_chain: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            qs_parent_chain: (T, S_parent)
            log_lik_chain: (T, S_parent)
            B_chain: (T, S_parent, S_parent)
            """

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

            qs_init = qs_parent_chain
            qs_final = jax.lax.fori_loop(0, 2, body_fun, qs_init)
            return qs_final

        H_p_, W_p_ = qs_parent_curr.shape[1], qs_parent_curr.shape[2]
        h_idx = jnp.arange(H_p_)
        w_idx = jnp.arange(W_p_)

        def update_site(h, w, qs_parent_all, log_lik_all, B_all):
            qs_chain = qs_parent_all[:, h, w, :]      # (T, S_parent)
            log_chain = log_lik_all[:, h, w, :]       # (T, S_parent)
            B_chain = B_all                           # (T, S_parent, S_parent)
            return update_site_chain(qs_chain, log_chain, B_chain)

        def update_row(h, qs_parent_all, log_lik_all, B_all):
            return jax.vmap(
                lambda w: update_site(h, w, qs_parent_all, log_lik_all, B_all)
            )(w_idx)  # (W_p_, T, S_parent)

        qs_parent_rows = jax.vmap(
            lambda h: update_row(h, qs_parent_curr, log_lik_parent, B_parent_all)
        )(h_idx)  # (H_p_, W_p_, T, S_parent)

        qs_parent_new = jnp.transpose(qs_parent_rows, (2, 0, 1, 3))  # (T, Hp, Wp, S_parent)
        return qs_parent_new

    def update_child(qs_child_curr, qs_parent_curr):
        H_p_, W_p_ = qs_parent_curr.shape[1], qs_parent_curr.shape[2]
        H_c_, W_c_ = qs_child_curr.shape[1], qs_child_curr.shape[2]
        assert H_c_ == 2 * H_p_ and W_c_ == 2 * W_p_

        log_prior_bias_child = top_down_prior_parent_to_child(
            qs_parent_curr, D_parent, H_c_, W_c_, S_child
        )  # (T, Hc, Wc, S_child)

        # Identity kernel at child in this block
        B_child = jnp.eye(S_child, dtype=jnp.float32)

        def update_site_chain(
            qs_child_chain: jnp.ndarray,
            log_bias_chain: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            qs_child_chain: (T, S_child)
            log_bias_chain: (T, S_child)
            """

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

            qs_init = qs_child_chain
            qs_final = jax.lax.fori_loop(0, 1, body_fun, qs_init)
            return qs_final

        update_site_chain_vmap = jax.vmap(
            jax.vmap(update_site_chain, in_axes=(1, 1), out_axes=1),
            in_axes=(1, 1),
            out_axes=1,
        )

        qs_child_new = update_site_chain_vmap(qs_child_curr, log_prior_bias_child)
        return qs_child_new

    qs_child_current = qs_child_grid
    qs_parent_current = qs_parent_grid

    def alt_step(_, carry):
        qs_child_c, qs_parent_c = carry
        qs_parent_new = update_parent(qs_child_c, qs_parent_c)
        qs_child_new = update_child(qs_child_c, qs_parent_new)
        return (qs_child_new, qs_parent_new)

    qs_child_final, qs_parent_final = jax.lax.fori_loop(
        0, num_iter, alt_step, (qs_child_current, qs_parent_current)
    )

    return qs_child_final, qs_parent_final

# -----------------------------------------------------------------------------
# 6. High-level inference entry point with multi-level states and EFE-based paths
# -----------------------------------------------------------------------------

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
    Run variational inference over states and (optional) paths in the Lorenz hierarchy.

    If the highest level that carries a path factor (num_paths > 1 and C_paths/E_paths
    are not None) exists, we run EFE-based path inference at that level, using
    its own state posterior as qs_top_grid in compute_expected_free_energy_paths.

    Preferences C are derived from params.pref_alpha when params is provided.
    Otherwise, an empirical fallback is used.
    """
    T0 = hierarchy.T0
    H0 = hierarchy.H_blocks
    W0 = hierarchy.W_blocks

    level0: LorenzLevel = hierarchy.levels[0]
    A0 = level0.A
    O = A0.shape[1]

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

    qs_levels: List[Optional[jnp.ndarray]] = [qs0_grid]
    qu_levels: List[Optional[jnp.ndarray]] = [None]

    # Early exit if only one level
    if len(hierarchy.levels) == 1:
        return {"qs_levels": qs_levels, "qu_levels": qu_levels}

    # 2. Level-1 and (optionally) level-2 states

    # Level 1
    level1: LorenzLevel = hierarchy.levels[1]
    states_grid1 = hierarchy.states_grids[1]

    # Initialize q(s^1)
    T1, H1, W1 = states_grid1.shape
    assert T1 == T0
    qs1_grid = jnp.full(
        (T0, H1, W1, level1.S),
        1.0 / level1.S,
        dtype=jnp.float32,
    )

    # Level 2 if present
    has_level2 = len(hierarchy.levels) > 2
    if has_level2:
        level2: LorenzLevel = hierarchy.levels[2]
        states_grid2 = hierarchy.states_grids[2]
        T2, H2, W2 = states_grid2.shape
        assert T2 == T0
        qs2_grid = jnp.full(
            (T0, H2, W2, level2.S),
            1.0 / level2.S,
            dtype=jnp.float32,
        )
    else:
        level2 = None
        qs2_grid = None

    # Determine top level carrying paths
    top_idx = None
    for idx in reversed(range(len(hierarchy.levels))):
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

    # Hierarchical inference:
    # We alternate block updates:
    #   0 <-> 1  (without paths)
    #   1 <-> 2  (with paths if level 2 is top)

    qs0_current = qs0_grid
    qs1_current = qs1_grid
    qs2_current = qs2_grid
    qu_top_current = qu_top

    for _ in range(num_iter_hier):
        # 0 <-> 1 (no explicit paths at level 1 in this configuration)
        qs0_current, qs1_current = vmp_two_level_states(
            level_child=level0,
            level_parent=level1,
            qs_child_grid=qs0_current,
            states_grid_parent=states_grid1,
            qu_parent=None,
            num_iter=1,
        )

        # 1 <-> 2 if level 2 exists
        if has_level2:
            # paths at level 2 if top_idx == 2
            qu_for_level2 = qu_top_current if (paths_active and top_idx == 2) else None
            qs1_current, qs2_current = vmp_two_level_states(
                level_child=level1,
                level_parent=level2,
                qs_child_grid=qs1_current,
                states_grid_parent=states_grid2,
                qu_parent=qu_for_level2,
                num_iter=1,
            )

        # Path update if active
        if paths_active:
            # top-level state grid for EFE is from the level that carries paths
            if top_idx == 2:
                qs_top_grid = qs2_current
            elif top_idx == 1:
                qs_top_grid = qs1_current
            else:
                qs_top_grid = qs0_current  # (not expected in current config)

            level0_for_efe = level0
            G_tu = compute_expected_free_energy_paths(
                level_top=level_top,
                level0=level0_for_efe,
                qs_top_grid=qs_top_grid,
                qs0_grid=qs0_current,
                C=C,
                tau=2,
            )
            qu_top_current = update_path_posterior_from_G(
                level_top,
                G_tu,
                gamma=efe_gamma,
                num_iter=2,
            )

    # Collect results
    qs_levels[0] = qs0_current
    qs_levels.append(qs1_current)
    if has_level2:
        qs_levels.append(qs2_current)

    if paths_active:
        qu_levels.append(qu_top_current)
    else:
        qu_levels.append(None)

    # Pad qu_levels to match number of levels (one qu entry per level)
    while len(qu_levels) < len(hierarchy.levels):
        qu_levels.append(None)

    return {
        "qs_levels": qs_levels,
        "qu_levels": qu_levels,
    }
