# lorenz_inference.py
"""
Variational message passing for the Lorenz hierarchy.

This module provides:
  - Single-chain VMP for the lowest (patch) level using A, B_states, E_states.
  - Patch-wise lowest-level inference by vmapping the single-chain VMP over
    all spatial patches (H0, W0).
  - Two-level state inference with spatial renormalization:
      * bottom-up messages from level 0 to level 1 via D
      * top-down messages from level 1 to level 0 via D
  - Top-level path inference using expected free energy over paths
    computed via lorenz_efe (risk + ambiguity - epistemic).

This is a Lorenz-specific instance of RGM-style inference and is structured
to be extended further for full RGM behaviour.
"""
import numpy as np

from typing import List, Dict, Any, Tuple, Optional

import jax
import jax.numpy as jnp
from jax import nn

from .lorenz_model import LorenzHierarchy, LorenzLevel
from . import maths as rgm_maths  # adapted from pymdp.jax.maths
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
    Build a flat observation array suitable for lowest-level inference,
    primarily for computing empirical preferences.

    We treat the quantized coefficients as "observations" in a one-hot form
    that is consistent with the A built in lorenz_model.py.

    Returns:
        obs_flat: (N, O) array, where N = T * H_blocks * W_blocks,
                  O = K * L; each row is a one-hot (or distribution) over O.
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

    Returns:
        obs_grid: (T, H0, W0, O)
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

    Args:
        O: outcome dimension at lowest level (O = K * L)
        mode: how to set preferences:
              - "uniform": no particular preferred outcome
              - "data_empirical": empirical distribution over training obs
        obs_flat: (N, O) one-hot/distribution observations (required for empirical)

    Returns:
        C: (O,) preference distribution
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
) -> Tuple[jnp.ndarray, float]:
    """
    Variational message passing for a single chain of hidden states with:
      - emission model A (shared across time),
      - transition model B(s'|s),
      - prior over initial state E.

    Args:
        A: (S, O) emission matrix P(o|s)
        B: (S, S) transition matrix P(s'|s)
        E: (S,) prior over s_0
        obs: (T, O) observations (one-hot / distribution over O)
        num_iter: number of VMP iterations

    Returns:
        qs: (T, S) posterior marginals over states
        F: scalar free energy (approximate, observation-only)
    """
    T = obs.shape[0]
    S = A.shape[0]

    # Precompute log-likelihoods ln P(o_t | s_t) for all t, s
    def likelihood_single(o_t: jnp.ndarray) -> jnp.ndarray:
        lik = (o_t[None, :] * A).sum(axis=1)  # (S,)
        return jnp.log(jnp.clip(lik, a_min=1e-16))

    log_liks = jax.vmap(likelihood_single)(obs)  # (T, S)

    # Initialize q(s_t) ~ uniform
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

    # Observation-only free energy (for diagnostics, using a single-chain view)
    obs_list = [obs]
    A_list = [A]
    qs_list = [qs.mean(axis=0)]  # crude aggregation
    F = rgm_maths.compute_free_energy(qs_list, [E], obs_list, A_list)

    return qs, F


# -----------------------------------------------------------------------------
# 3. Patch-wise lowest-level inference (vmapped chains)
# -----------------------------------------------------------------------------

def infer_lowest_level_patches(
    level0: LorenzLevel,
    lorenz_data_dict: Dict[str, Any],
    num_iter_lowest: int = 8,
) -> Tuple[jnp.ndarray, float]:
    """
    Run VMP at the lowest level independently for each patch, using a
    Python loop over (h0, w0) to keep memory usage manageable.

      - obs_grid: (T, H0, W0, O)
      - for each (h0, w0): run vmp_single_chain on (T, O) chain
      - stack results into qs0_grid: (T, H0, W0, S0)

    Args:
        level0: lowest-level LorenzLevel (A, B_states, E_states)
        lorenz_data_dict: dataset dict
        num_iter_lowest: iterations for single-chain VMP

    Returns:
        qs0_grid: (T, H0, W0, S0) posterior over lowest-level states
        F_avg: average free energy across patches (diagnostic)
    """
    A0 = level0.A        # (S0, O)
    B0 = level0.B_states # (S0, S0)
    E0 = level0.E_states # (S0,)

    obs_grid = build_lowest_level_observations_grid(lorenz_data_dict)  # (T, H0, W0, O)
    T, H0, W0, O = obs_grid.shape
    S0 = A0.shape[0]

    # Jitted single-chain inference
    vmp_single_chain_jit = jax.jit(vmp_single_chain, static_argnames=("num_iter",))

    # We'll accumulate results in host (NumPy) arrays to avoid excessive device memory
    qs0_host = np.zeros((T, H0, W0, S0), dtype=np.float32)
    F_sum = 0.0
    num_patches = H0 * W0

    for h0 in range(H0):
        for w0 in range(W0):
            obs_patch = obs_grid[:, h0, w0, :]         # (T, O)
            qs_chain, F_chain = vmp_single_chain_jit(A0, B0, E0, obs_patch, num_iter=num_iter_lowest)
            qs_chain_host = np.array(qs_chain)         # move to host
            qs0_host[:, h0, w0, :] = qs_chain_host
            F_sum += float(F_chain)

    qs0_grid = jnp.array(qs0_host)          # back to device
    F_avg = F_sum / float(num_patches)

    return qs0_grid, F_avg


# -----------------------------------------------------------------------------
# 4. Two-level hierarchical state inference via D
# -----------------------------------------------------------------------------

def bottom_up_message_level0_to_level1(
    qs0_grid: jnp.ndarray,
    D1: jnp.ndarray,
    states_grid1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute a bottom-up "pseudo-likelihood" for each Level-1 state
    from Level-0 posterior beliefs and the deterministic D mapping.

    Args:
        qs0_grid: (T, H0, W0, S0) posterior over Level-0 states per patch.
        D1: (S1, child_config_dim=4) array of child state patterns.
        states_grid1: (T, H1, W1) integer states for Level-1 (layout).

    Returns:
        log_lik1: (T, H1, W1, S1) log "likelihood" contributions at Level 1.
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
        return jax.vmap(lambda pattern: score_parent_state(qs0_child_site, pattern))(D1_indices)

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
    num_iter: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform coupled VMP over Level-0 and Level-1 states.

    - Level 0: updated by A0, B0, E0 and top-down log messages from Level 1.
    - Level 1: updated by bottom-up log messages from Level 0 via D1,
               plus B1 and E1.

    Current simplifications:
      - Level-1 spatial sites are treated as independent copies of the
        same chain model (same B1, E1).
    """
    A0 = level0.A
    B0 = level0.B_states
    E0 = level0.E_states

    B1 = level1.B_states
    E1 = level1.E_states
    D1 = level1.D

    T, H0, W0, S0 = qs0_grid.shape
    T1, H1, W1 = states_grid1.shape
    assert T == T1
    assert H0 == 2 * H1 and W0 == 2 * W1

    S1 = int(B1.shape[0])
    qs1_grid = jnp.full((T, H1, W1, S1), 1.0 / S1, dtype=jnp.float32)

    def update_level1(qs0_grid_current: jnp.ndarray,
                      qs1_grid_current: jnp.ndarray) -> jnp.ndarray:
        log_lik1 = bottom_up_message_level0_to_level1(qs0_grid_current, D1, states_grid1)

        def update_site(qs1_site: jnp.ndarray,
                        log_lik1_site: jnp.ndarray) -> jnp.ndarray:
            def fwd_bwd(qs_):
                def forward_messages(qs_):
                    msgs = []
                    prev_q = qs_[0]
                    msgs.append(jnp.log(jnp.clip(E1, 1e-16)))
                    for t in range(1, T):
                        msg = jnp.log(jnp.clip(B1 @ prev_q, 1e-16))
                        msgs.append(msg)
                        prev_q = qs_[t]
                    return jnp.stack(msgs, axis=0)

                def backward_messages(qs_):
                    msgs = []
                    next_q = qs_[-1]
                    msgs.append(jnp.zeros_like(next_q))
                    for t in range(T - 2, -1, -1):
                        msg = jnp.log(jnp.clip(B1.T @ next_q, 1e-16))
                        msgs.append(msg)
                        next_q = qs_[t]
                    msgs = msgs[::-1]
                    return jnp.stack(msgs, axis=0)

                m_plus = forward_messages(qs_)
                m_minus = backward_messages(qs_)
                ln_qs = log_lik1_site + m_plus + m_minus
                return nn.softmax(ln_qs, axis=1)

            def body_fun(_, qs_):
                return fwd_bwd(qs_)

            return jax.lax.fori_loop(0, 2, body_fun, qs1_site)

        qs1_grid_new = jax.vmap(
            jax.vmap(update_site, in_axes=(0, 0)),
            in_axes=(0, 0)
        )(qs1_grid_current, log_lik1)

        return qs1_grid_new

    def update_level0(qs0_grid_current: jnp.ndarray,
                      qs1_grid_current: jnp.ndarray) -> jnp.ndarray:
        H1, W1 = qs1_grid_current.shape[1], qs1_grid_current.shape[2]

        def build_topdown_prior() -> jnp.ndarray:
            def bias_for_time(t):
                qs1_t = qs1_grid_current[t]  # (H1, W1, S1)
                bias = jnp.zeros((H0, W0, S0), dtype=jnp.float32)

                for h1 in range(H1):
                    for w1 in range(W1):
                        qs1_hw = qs1_t[h1, w1]  # (S1,)
                        for child_idx, (dh, dw) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
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

            return jax.vmap(bias_for_time)(jnp.arange(T))

        log_prior_bias0 = build_topdown_prior()

        def update_site(qs0_site: jnp.ndarray,
                        log_bias_site: jnp.ndarray) -> jnp.ndarray:
            def fwd_bwd(qs_):
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

                m_plus = forward_messages(qs_)
                m_minus = backward_messages(qs_)
                ln_qs = log_bias_site + m_plus + m_minus
                return nn.softmax(ln_qs, axis=1)

            def body_fun(_, qs_):
                return fwd_bwd(qs_)

            return jax.lax.fori_loop(0, 1, body_fun, qs0_site)

        qs0_grid_new = jax.vmap(
            jax.vmap(update_site, in_axes=(0, 0)),
            in_axes=(0, 0)
        )(qs0_grid_current, log_prior_bias0)

        return qs0_grid_new

    qs0_current = qs0_grid
    qs1_current = qs1_grid

    def alt_step(_, carry):
        qs0_c, qs1_c = carry
        qs1_new = update_level1(qs0_c, qs1_c)
        qs0_new = update_level0(qs0_c, qs1_new)
        return (qs0_new, qs1_new)

    qs0_final, qs1_final = jax.lax.fori_loop(0, num_iter, alt_step, (qs0_current, qs1_current))

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
    Run variational inference over states and paths in the Lorenz hierarchy.

    Steps:
      1. Infer level-0 states patch-wise using A0, B0, E0 (one chain per patch).
      2. If a higher level exists:
           - Run two-level hierarchical state inference using D between
             level 0 and 1 (bottom-up + top-down).
      3. Build a preference distribution C over lowest-level outcomes.
      4. If the top level has a path factor, compute expected free energy
         G_tu for paths via lorenz_efe, and update q(u_t) using G and
         B_paths/E_paths.

    Args:
        hierarchy: LorenzHierarchy instance
        lorenz_data_dict: output of build_lorenz_patch_dataset
        num_iter_lowest: iterations for lowest-level chain VMP
        num_iter_hier: iterations for hierarchical alternating updates
        efe_gamma: precision over expected free energy for paths
        pref_mode: "uniform" or "data_empirical" for preferences over outcomes

    Returns:
        dict with:
          'qs_levels': list of posterior arrays per level:
               level 0: (T, H0, W0, S0)
               level 1: (T, H1, W1, S1) if exists
          'qu_levels': list with path posteriors or None per level
          'F_lowest': average free energy at lowest level (observation-only approximation)
    """
    T = hierarchy.T
    H0 = hierarchy.H_blocks
    W0 = hierarchy.W_blocks

    # Level 0 model
    level0: LorenzLevel = hierarchy.levels[0]
    A0 = level0.A
    S0 = A0.shape[0]
    O = A0.shape[1]

    # Observations for level 0 (flat) for preferences
    obs_flat = build_lowest_level_observations_flat(lorenz_data_dict)  # (N, O)
    N = obs_flat.shape[0]
    assert N == T * H0 * W0, "Observation length mismatch."

    # Preferences over outcomes at lowest level
    C = build_preference_distribution_lowest(O, mode=pref_mode, obs_flat=obs_flat)

    # 1. Infer Level-0 states patch-wise
    qs0_grid, F0_avg = infer_lowest_level_patches(level0, lorenz_data_dict,
                                                  num_iter_lowest=num_iter_lowest)

    qs_levels: List[Optional[jnp.ndarray]] = [qs0_grid]
    qu_levels: List[Optional[jnp.ndarray]] = [None]

    qs1_grid = None

    # 2. Hierarchical inference if there is at least one higher level
    if len(hierarchy.levels) > 1:
        level1: LorenzLevel = hierarchy.levels[1]
        states_grid1 = hierarchy.states_grids[1]  # (T, H1, W1)

        qs0_grid_h, qs1_grid = vmp_two_level_states(
            level0,
            level1,
            qs0_grid,
            states_grid1,
            num_iter=num_iter_hier,
        )

        qs_levels[0] = qs0_grid_h
        qs_levels.append(qs1_grid)
        qu_levels.append(None)

    # 3. Top-level path factor with EFE
    top_idx = len(hierarchy.levels) - 1
    level_top = hierarchy.levels[top_idx]
    if level_top.num_paths > 0 and qs1_grid is not None:
        G_tu = compute_expected_free_energy_paths(
            level_top,
            level0,
            qs1_grid,
            qs_levels[0],  # updated lowest-level qs0_grid_h
            C,
        )
        qu_top = update_path_posterior_from_G(
            level_top,
            G_tu,
            gamma=efe_gamma,
            num_iter=2,
        )
        qu_levels[top_idx] = qu_top
    else:
        if len(qu_levels) < len(hierarchy.levels):
            qu_levels += [None] * (len(hierarchy.levels) - len(qu_levels))

    return {
        "qs_levels": qs_levels,
        "qu_levels": qu_levels,
        "F_lowest": F0_avg,
    }
