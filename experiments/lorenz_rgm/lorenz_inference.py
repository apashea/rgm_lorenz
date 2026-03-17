# lorenz_inference.py
"""
Variational message passing for the Lorenz hierarchy.

This module provides:
- Single-chain VMP for the lowest (patch) level using A, B_states_paths (U=1),
  and E_states.
- Patch-wise lowest-level inference by looping over patches and calling
  the single-chain VMP (to keep memory usage manageable).
- Two-level state inference with spatial renormalization:
    * bottom-up messages from level 0 to level 1 via D_state_from_parent
    * top-down messages from level 1 to level 0 via D_state_from_parent
  using path-marginal (effective) dynamics at level 1.
- Top-level path inference using expected free energy over paths computed
  via lorenz_efe (risk + ambiguity - epistemic), with path-dependent
  transitions B_states_paths and path dynamics C_paths,E_paths.

NOTE: Temporal renormalization (block structure using T0,T1,T2,K0,K1) is
encoded in LorenzHierarchy but not yet exploited here; all levels currently
use T0 as the time index. This will be extended in the next stage.
"""

from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import nn

from .lorenz_model import LorenzHierarchy, LorenzLevel
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
    T = int(lorenz_data_dict["T"])
    H0 = int(lorenz_data_dict["H_blocks"])
    W0 = int(lorenz_data_dict["W_blocks"])
    O = obs_flat.shape[1]

    obs_grid = obs_flat.reshape(T, H0, W0, O)
    return obs_grid


def build_preference_distribution_lowest(
    O: int,
    mode: str = "data_empirical",
    obs_flat: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Build a preference distribution C over lowest-level outcomes (O-dim).
    """
    if mode == "uniform" or obs_flat is None:
        C = jnp.ones((O,), dtype=jnp.float32)
        C = C / C.sum()
        return C

    C = obs_flat.mean(axis=0)
    C = C / (C.sum() + 1e-8)
    return C


# -----------------------------------------------------------------------------
# 2. Single-chain VMP using A and B (no spatial coupling)
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
      B: (S, S) with B[s_next, s] = P(s_next | s)
      E: (S,) initial state prior
      obs: (T, O) one-hot or categorical observations

    Returns:
      qs: (T, S) approximate posterior over states
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

    Returns:
      qs0_grid: (T, H0, W0, S0)
    """
    A0 = level0.A
    if level0.B_states_paths is None:
        raise ValueError("level0 must define B_states_paths with U0=1.")
    B0 = level0.B_states_paths[:, :, 0]  # (S0, S0)
    E0 = level0.E_states

    obs_grid = build_lowest_level_observations_grid(lorenz_data_dict)  # (T, H0, W0, O)
    T, H0, W0, O = obs_grid.shape
    S0 = A0.shape[0]

    vmp_single_chain_jit = jax.jit(vmp_single_chain, static_argnames=("num_iter",))

    qs0_host = np.zeros((T, H0, W0, S0), dtype=np.float32)

    for h0 in range(H0):
        for w0 in range(W0):
            obs_patch = obs_grid[:, h0, w0, :]
            qs_chain = vmp_single_chain_jit(A0, B0, E0, obs_patch, num_iter=num_iter_lowest)
            qs0_host[:, h0, w0, :] = np.array(qs_chain)

    qs0_grid = jnp.array(qs0_host)
    return qs0_grid


# -----------------------------------------------------------------------------
# 4. Two-level hierarchical state inference via D (with path-dependent B1)
# -----------------------------------------------------------------------------

def bottom_up_message_level0_to_level1(
    qs0_grid: jnp.ndarray,
    D1: jnp.ndarray,
    states_grid1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Bottom-up pseudo-likelihood from Level 0 to Level 1 via D.

    Args:
      qs0_grid: (T, H0, W0, S0)
      D1: (S1, 4) mapping each parent state to its 4 child indices
      states_grid1: (T, H1, W1)

    Returns:
      log_lik1: (T, H1, W1, S1)
    """
    T, H0, W0, S0 = qs0_grid.shape
    T1, H1, W1 = states_grid1.shape
    assert T == T1
    assert H0 == 2 * H1 and W0 == 2 * W1

    S1 = D1.shape[0]

    def site_children(qs0_t: jnp.ndarray) -> jnp.ndarray:
        c00 = qs0_t[0::2, 0::2, :]
        c01 = qs0_t[0::2, 1::2, :]
        c10 = qs0_t[1::2, 0::2, :]
        c11 = qs0_t[1::2, 1::2, :]
        return jnp.stack([c00, c01, c10, c11], axis=2)  # (H1, W1, 4, S0)

    qs0_children = jax.vmap(site_children)(qs0_grid)  # (T, H1, W1, 4, S0)
    D1_indices = D1.astype(jnp.int32)  # (S1, 4)

    def score_parent_state(qs0_child: jnp.ndarray, pattern: jnp.ndarray) -> jnp.ndarray:
        probs = qs0_child[jnp.arange(4), pattern]  # (4,)
        log_probs = jnp.log(jnp.clip(probs, 1e-16))
        return log_probs.sum()

    def score_all_parents_for_site(qs0_child_site: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda pattern: score_parent_state(qs0_child_site, pattern)
        )(D1_indices)

    def score_all_sites(qs0_children_t: jnp.ndarray) -> jnp.ndarray:
        def score_site(children_site: jnp.ndarray) -> jnp.ndarray:
            return score_all_parents_for_site(children_site)

        return jax.vmap(jax.vmap(score_site, in_axes=0), in_axes=0)(qs0_children_t)

    log_lik1 = jax.vmap(score_all_sites)(qs0_children)  # (T, H1, W1, S1)
    return log_lik1


def vmp_two_level_states(
    level0: LorenzLevel,
    level1: LorenzLevel,
    qs0_grid: jnp.ndarray,
    states_grid1: jnp.ndarray,
    qu_top: Optional[jnp.ndarray] = None,
    num_iter: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Coupled VMP over Level-0 and Level-1 states with path-dependent dynamics
    at level 1.

    Args:
      level0: lowest-level LorenzLevel
      level1: parent-level LorenzLevel
      qs0_grid: (T, H0, W0, S0)
      states_grid1: (T, H1, W1)
      qu_top: (T, U) or None (path posterior at this level)
      num_iter: alternating sweeps

    Returns:
      qs0_final: (T, H0, W0, S0)
      qs1_final: (T, H1, W1, S1)
    """
    A0 = level0.A
    if level0.B_states_paths is None:
        raise ValueError("level0 must define B_states_paths with U0=1.")
    B0 = level0.B_states_paths[:, :, 0]  # (S0, S0)
    E0 = level0.E_states

    B1_base = None  # no separate stored B; we derive B_eff from B_states_paths
    E1 = level1.E_states
    D1 = level1.D_state_from_parent
    if D1 is None:
        raise ValueError("level1 must define D_state_from_parent.")
    B_states_paths = level1.B_states_paths  # (S1, S1, U) or None

    T, H0, W0, S0 = qs0_grid.shape
    T1, H1, W1 = states_grid1.shape
    assert T == T1
    assert H0 == 2 * H1 and W0 == 2 * W1

    if B_states_paths is not None:
        S1 = int(B_states_paths.shape[0])
    else:
        # fallback: treat as single-path with identity transitions
        S1 = int(D1.shape[0])
    qs1_grid = jnp.full((T, H1, W1, S1), 1.0 / S1, dtype=jnp.float32)

    # Effective path posterior
    if B_states_paths is None:
        U = 0
        qu_top_eff = None
    else:
        U = B_states_paths.shape[2]
        if qu_top is None:
            qu_top_eff = jnp.full((T, U), 1.0 / U, dtype=jnp.float32)
        else:
            qu_top_eff = qu_top

    # ----- Level-1 update -----

    def update_level1(
        qs0_grid_current: jnp.ndarray,
        qs1_grid_current: jnp.ndarray,
        qu_current: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        log_lik1 = bottom_up_message_level0_to_level1(
            qs0_grid_current, D1, states_grid1
        )  # (T, H1, W1, S1)

        if B_states_paths is not None and qu_current is not None:
            # B_eff[t] = sum_u q(u_t=u) * B_states_paths[:,:,u]
            def B_eff_for_time(t):
                qu_t = qu_current[t]  # (U,)
                return (B_states_paths * qu_t[None, None, :]).sum(axis=2)  # (S1, S1)

            B_eff_all = jax.vmap(B_eff_for_time)(jnp.arange(T))  # (T, S1, S1)
        else:
            # Use identity (no dynamics) if no B_states_paths available
            B_eff_all = jnp.broadcast_to(
                jnp.eye(S1, dtype=jnp.float32), (T, S1, S1)
            )

        def update_site_chain(
            qs1_chain: jnp.ndarray,
            log_lik_chain: jnp.ndarray,
            B_eff_chain: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            qs1_chain: (T, S1)
            log_lik_chain: (T, S1)
            B_eff_chain: (T, S1, S1)
            """

            def forward_messages(qs_):
                msgs = []
                prev_q = qs_[0]
                msgs.append(jnp.log(jnp.clip(E1, 1e-16)))
                for t in range(1, T):
                    B_t = B_eff_chain[t]
                    msg = jnp.log(jnp.clip(B_t @ prev_q, 1e-16))
                    msgs.append(msg)
                    prev_q = qs_[t]
                return jnp.stack(msgs, axis=0)

            def backward_messages(qs_):
                msgs = []
                next_q = qs_[-1]
                msgs.append(jnp.zeros_like(next_q))
                for t in range(T - 2, -1, -1):
                    B_tp1 = B_eff_chain[t + 1]
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

            qs_init = qs1_chain
            qs_final = jax.lax.fori_loop(0, 2, body_fun, qs_init)
            return qs_final

        H1_, W1_ = qs1_grid_current.shape[1], qs1_grid_current.shape[2]
        h_indices = jnp.arange(H1_)
        w_indices = jnp.arange(W1_)

        def update_site(h_idx, w_idx, qs1_curr, log_lik1_all, B_all):
            qs_chain = qs1_curr[:, h_idx, w_idx, :]        # (T, S1)
            log_chain = log_lik1_all[:, h_idx, w_idx, :]   # (T, S1)
            B_chain = B_all                                # (T, S1, S1)
            return update_site_chain(qs_chain, log_chain, B_chain)

        def update_row(h_idx, qs1_curr, log_lik1_all, B_all):
            return jax.vmap(
                lambda w_idx: update_site(h_idx, w_idx, qs1_curr, log_lik1_all, B_all)
            )(w_indices)  # (W1, T, S1)

        qs1_rows = jax.vmap(
            lambda h_idx: update_row(h_idx, qs1_grid_current, log_lik1, B_eff_all)
        )(h_indices)  # (H1, W1, T, S1)

        qs1_grid_new = jnp.transpose(qs1_rows, (2, 0, 1, 3))  # (T, H1, W1, S1)
        return qs1_grid_new

    # ----- Level-0 update -----

    def update_level0(
        qs0_grid_current: jnp.ndarray,
        qs1_grid_current: jnp.ndarray,
    ) -> jnp.ndarray:
        H1_, W1_ = qs1_grid_current.shape[1], qs1_grid_current.shape[2]

        def build_topdown_prior() -> jnp.ndarray:
            def bias_for_time(t):
                qs1_t = qs1_grid_current[t]  # (H1, W1, S1)
                bias = jnp.zeros((H0, W0, S0), dtype=jnp.float32)

                for h1 in range(H1_):
                    for w1 in range(W1_):
                        qs1_hw = qs1_t[h1, w1]  # (S1,)
                        for child_idx, (dh, dw) in enumerate(
                            [(0, 0), (0, 1), (1, 0), (1, 1)]
                        ):
                            h0 = 2 * h1 + dh
                            w0 = 2 * w1 + dw
                            child_states = D1[:, child_idx]  # (S1,)
                            accum = jnp.zeros((S0,), dtype=jnp.float32)

                            def body_fun(s1, acc):
                                s0_pref = child_states[s1]
                                return acc.at[s0_pref].add(qs1_hw[s1])

                            accum = jax.lax.fori_loop(0, S1, body_fun, accum)
                            accum = accum / (accum.sum() + 1e-8)
                            log_accum = jnp.log(jnp.clip(accum, 1e-16))
                            bias = bias.at[h0, w0, :].set(log_accum)
                return bias

            return jax.vmap(bias_for_time)(jnp.arange(T))  # (T, H0, W0, S0)

        log_prior_bias0 = build_topdown_prior()

        def update_site_chain(
            qs0_chain: jnp.ndarray,
            log_bias_chain: jnp.ndarray,
        ) -> jnp.ndarray:
            """
            qs0_chain: (T, S0)
            log_bias_chain: (T, S0)
            """

            def forward_messages(qs_):
                msgs = []
                prev_q = qs_[0]
                msgs.append(jnp.log(jnp.clip(E0, 1e-16)))
                for t in range(1, T):
                    msg = jnp.log(jnp.clip(B0 @ prev_q, 1e-16))
                    msgs.append(msg)
                    prev_q = qs_[t]
                return jnp.stack(msgs, axis=0)

            def backward_messages(qs_):
                msgs = []
                next_q = qs_[-1]
                msgs.append(jnp.zeros_like(next_q))
                for t in range(T - 2, -1, -1):
                    msg = jnp.log(jnp.clip(B0.T @ next_q, 1e-16))
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

            qs_init = qs0_chain
            qs_final = jax.lax.fori_loop(0, 1, body_fun, qs_init)
            return qs_final

        update_site_chain_vmap = jax.vmap(
            jax.vmap(update_site_chain, in_axes=(1, 1), out_axes=1),
            in_axes=(1, 1),
            out_axes=1,
        )

        qs0_grid_new = update_site_chain_vmap(qs0_grid_current, log_prior_bias0)
        return qs0_grid_new

    # Alternating updates
    qs0_current = qs0_grid
    qs1_current = qs1_grid

    def alt_step(_, carry):
        qs0_c, qs1_c = carry
        qs1_new = update_level1(qs0_c, qs1_c, qu_top_eff if "qu_top_eff" in locals() else None)
        qs0_new = update_level0(qs0_c, qs1_new)
        return (qs0_new, qs1_new)

    qs0_final, qs1_final = jax.lax.fori_loop(
        0, num_iter, alt_step, (qs0_current, qs1_current)
    )

    return qs0_final, qs1_final


# -----------------------------------------------------------------------------
# 5. High-level inference entry point with EFE-based paths
# -----------------------------------------------------------------------------

def infer_lorenz_hierarchy(
    hierarchy: LorenzHierarchy,
    lorenz_data_dict: Dict[str, Any],
    num_iter_lowest: int = 8,
    num_iter_hier: int = 4,
    efe_gamma: float = 16.0,
    pref_mode: str = "data_empirical",
) -> Dict[str, Any]:
    """
    Run variational inference over states and (optional) paths in the Lorenz hierarchy.

    For now:
      - We infer states at level 0 and (if present) level 1 over the T0 time index.
      - If the top level has a proper path factor (num_paths > 1 and C_paths/E_paths
        are not None), we run EFE-based path inference at that level; otherwise
        we only infer states.
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

    C_pref = build_preference_distribution_lowest(O, mode=pref_mode, obs_flat=obs_flat)

    # 1. Level-0 states
    qs0_grid = infer_lowest_level_patches(
        level0, lorenz_data_dict, num_iter_lowest=num_iter_lowest
    )  # (T0, H0, W0, S0)

    qs_levels: List[Optional[jnp.ndarray]] = [qs0_grid]
    qu_levels: List[Optional[jnp.ndarray]] = [None]

    qs1_grid = None
    qu_top = None

    # For now we only implement a 2-level state hierarchy (0 and 1) with
    # paths potentially at the highest level in hierarchy.levels.
    if len(hierarchy.levels) > 1:
        level1: LorenzLevel = hierarchy.levels[1]
        states_grid1 = hierarchy.states_grids[1]

        top_idx = len(hierarchy.levels) - 1
        level_top = hierarchy.levels[top_idx]

        # Only treat paths as active if we have a full path factor:
        # more than one path and both C_paths and E_paths defined.
        paths_active = (
            level_top.num_paths is not None
            and level_top.num_paths > 1
            and level_top.C_paths is not None
            and level_top.E_paths is not None
        )

        if paths_active:
            U = level_top.num_paths
            qu_top = jnp.full((T0, U), 1.0 / U, dtype=jnp.float32)
        else:
            qu_top = None

        qs0_current = qs0_grid
        qs1_current = jnp.full(
            (T0, states_grid1.shape[1], states_grid1.shape[2], level1.S),
            1.0 / level1.S,
            dtype=jnp.float32,
        )

        for _ in range(num_iter_hier):
            qs0_current, qs1_current = vmp_two_level_states(
                level0,
                level1,
                qs0_current,
                states_grid1,
                qu_top=qu_top,
                num_iter=1,
            )

            if paths_active:
                G_tu = compute_expected_free_energy_paths(
                    level_top,
                    level0,
                    qs1_current,
                    qs0_current,
                    C_pref,
                    tau=2,  # planning horizon (can be tuned)
                )

                qu_top = update_path_posterior_from_G(
                    level_top,
                    G_tu,
                    gamma=efe_gamma,
                    num_iter=2,
                )

        qs0_grid_h = qs0_current
        qs1_grid = qs1_current

        qs_levels[0] = qs0_grid_h
        qs_levels.append(qs1_grid)
        qu_levels.append(qu_top)
    else:
        if len(qu_levels) < len(hierarchy.levels):
            qu_levels += [None] * (len(hierarchy.levels) - len(qu_levels))

    return {
        "qs_levels": qs_levels,
        "qu_levels": qu_levels,
    }
