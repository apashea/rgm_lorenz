# lorenz_inference.py
"""
Variational message passing for the Lorenz hierarchy.

This module provides:
- Construction of lowest-level observations for the Lorenz patch model.
- Single-chain VMP over time using A, B_states, E_states.
- Patch-wise lowest-level inference by looping over patches.
- Two-level hierarchical state inference with spatial renormalization:
  * bottom-up messages from level 0 to level 1 via D
  * top-down messages from level 1 to level 0 via D
- Top-level path inference using expected free energy over paths
  computed via lorenz_efe (risk + ambiguity - epistemic), with
  path-dependent transitions B_states_paths at the top state level.

Current limitations:
- Spatial hierarchy is fixed at 2 levels (0 and 1).
- D operators are fixed (no structure learning yet).

CONVENTIONS (consistent with lorenz_model.py, lorenz_learning.py):

- For each level ℓ with S_l states and U_l path/control values:

    B_states_paths_l[s_next, s, u] = P(s_next | s, u_l = u)

  with shape (S_l, S_l, U_l). For levels without multiple paths we set
  U_l = 1 and always use u=0, so the effective B_states_l matrix is:

    B_states_l = B_states_paths_l[:, :, 0]   # (S_l, S_l)

- In this file:
    - level.B_states is always the effective (S, S) matrix B[s_next, s]
      used when no explicit path dimension is needed.
    - level.B_states_paths may be (S, S, U) at the top level (when a path
      factor exists) or (S, S, 1) / None at other levels.
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
# 1. Lowest-level observations and preferences
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

    # q_coeffs[n, k] ∈ {0,...,L-1} is the quantized bin index for component k
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
# 2. Generic VMP for a single temporal chain
# -----------------------------------------------------------------------------

def vmp_temporal_chain(
    log_lik_chain: jnp.ndarray,
    B_chain: jnp.ndarray,
    E: jnp.ndarray,
    num_iter: int = 4,
    q_init: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Variational message passing for a single chain of hidden states with
    time-homogeneous transitions B(s_next|s) and prior E, given per-time
    log-likelihood contributions.

    Args:
        log_lik_chain: (T, S) log p(o_t | s_t) or log-bias messages
        B_chain: (S, S) transition matrix with B[s_next, s] = P(s_next | s)
        E: (S,) prior over s_0
        num_iter: number of VMP iterations
        q_init: optional initial posterior (T, S); if None, start uniform

    Returns:
        qs: (T, S) posterior marginals over states
    """
    T, S = log_lik_chain.shape

    if q_init is None:
        qs = jnp.full((T, S), 1.0 / S, dtype=jnp.float32)
    else:
        qs = q_init

    ln_prior = jnp.log(jnp.clip(E, 1e-16))

    def forward_messages(qs_):
        msgs = []
        prev_q = qs_[0]
        msgs.append(ln_prior)
        for t in range(1, T):
            msg = jnp.log(jnp.clip(B_chain @ prev_q, 1e-16))
            msgs.append(msg)
            prev_q = qs_[t]
        return jnp.stack(msgs, axis=0)

    def backward_messages(qs_):
        msgs = []
        next_q = qs_[-1]
        msgs.append(jnp.zeros_like(next_q))
        for t in range(T - 2, -1, -1):
            msg = jnp.log(jnp.clip(B_chain.T @ next_q, 1e-16))
            msgs.append(msg)
            next_q = qs_[t]
        msgs = msgs[::-1]
        return jnp.stack(msgs, axis=0)

    def vmp_iteration(qs_):
        m_plus = forward_messages(qs_)
        m_minus = backward_messages(qs_)
        ln_qs = log_lik_chain + m_plus + m_minus
        return nn.softmax(ln_qs, axis=1)

    def body_fun(_, qs_):
        return vmp_iteration(qs_)

    qs = jax.lax.fori_loop(0, num_iter, body_fun, qs)
    return qs


# -----------------------------------------------------------------------------
# 3. Single-chain VMP with A and B (used for lowest-level patch chains)
# -----------------------------------------------------------------------------

def vmp_single_chain(
    A: jnp.ndarray,
    B: jnp.ndarray,
    E: jnp.ndarray,
    obs: jnp.ndarray,
    num_iter: int = 8,
) -> jnp.ndarray:
    """
    Variational message passing for a single chain of hidden states with:
      - emission model A (shared across time),
      - transition model B(s_next|s),
      - prior over initial state E.

    Args:
        A: (S, O) emission matrix P(o|s)
        B: (S, S) transition matrix P(s_next|s)
        E: (S,) prior over s_0
        obs: (T, O) observations (one-hot / distribution over O)
        num_iter: number of VMP iterations

    Returns:
        qs: (T, S) posterior marginals over states
    """
    S, O = A.shape
    T = obs.shape[0]

    def likelihood_single(o_t: jnp.ndarray) -> jnp.ndarray:
        lik = (o_t[None, :] * A).sum(axis=1)  # (S,)
        return jnp.log(jnp.clip(lik, a_min=1e-16))

    log_liks = jax.vmap(likelihood_single)(obs)  # (T, S)

    qs = vmp_temporal_chain(
        log_lik_chain=log_liks,
        B_chain=B,
        E=E,
        num_iter=num_iter,
        q_init=None,
    )
    return qs


# -----------------------------------------------------------------------------
# 4. Patch-wise lowest-level inference (loop over patches)
# -----------------------------------------------------------------------------

def infer_lowest_level_patches(
    level0: LorenzLevel,
    lorenz_data_dict: Dict[str, Any],
    num_iter_lowest: int = 8,
) -> jnp.ndarray:
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
    """
    A0 = level0.A               # (S0, O0)
    B0 = level0.B_states        # (S0, S0)
    E0 = level0.E_states        # (S0,)

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
# 5. Bottom-up and top-down messages between levels (0 and 1)
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
        D1: (S1, 4) array of child state patterns.
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
        # qs0_child: (4, S0), pattern: (4,)
        probs = qs0_child[jnp.arange(4), pattern]  # (4,)
        log_probs = jnp.log(jnp.clip(probs, 1e-16))
        return log_probs.sum()

    def score_all_parents_for_site(qs0_child_site: jnp.ndarray) -> jnp.ndarray:
        # qs0_child_site: (4, S0)
        return jax.vmap(lambda pattern: score_parent_state(qs0_child_site, pattern))(D1_indices)

    def score_all_sites(qs0_children_t: jnp.ndarray) -> jnp.ndarray:
        def score_site(children_site: jnp.ndarray) -> jnp.ndarray:
            return score_all_parents_for_site(children_site)
        return jax.vmap(jax.vmap(score_site, in_axes=0), in_axes=0)(qs0_children_t)

    log_lik1 = jax.vmap(score_all_sites)(qs0_children)  # (T, H1, W1, S1)
    return log_lik1


# -----------------------------------------------------------------------------
# 6. Two-level coupled VMP for states (0 ↔ 1)
# -----------------------------------------------------------------------------

def vmp_two_level_states(
    level0: LorenzLevel,
    level1: LorenzLevel,
    qs0_grid: jnp.ndarray,
    states_grid1: jnp.ndarray,
    qu_top: Optional[jnp.ndarray] = None,
    num_iter: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform coupled VMP over Level-0 and Level-1 states.

    - Level 0: updated by A0, B0, E0 and top-down log messages from Level 1.
    - Level 1: updated by bottom-up log messages from Level 0 via D1,
               plus (optionally) path-dependent transitions B_states_paths
               and E1, mixed according to current path posterior qu_top.

    This function is still written specifically for levels 0 and 1,
    but uses the new B semantics (B[s_next, s]) and is structured so
    that it can be generalized to arbitrary adjacent levels.

    Args:
        level0: lowest-level LorenzLevel
        level1: parent-level LorenzLevel
        qs0_grid: (T, H0, W0, S0) initial qs at level 0
        states_grid1: (T, H1, W1) layout grid for level 1
        qu_top: (T, U) path posterior; if None, ignored (no path dependence)
        num_iter: alternating updates between levels

    Returns:
        qs0_final: (T, H0, W0, S0)
        qs1_final: (T, H1, W1, S1)
    """
    A0 = level0.A
    B0 = level0.B_states      # (S0, S0)
    E0 = level0.E_states

    B1 = level1.B_states      # (S1, S1)
    E1 = level1.E_states
    D1 = level1.D
    B_states_paths = level1.B_states_paths  # currently (S1, S1, 1) for level 1

    T, H0, W0, S0 = qs0_grid.shape
    T1, H1, W1 = states_grid1.shape
    assert T == T1
    assert H0 == 2 * H1 and W0 == 2 * W1

    S1 = int(B1.shape[0])
    qs1_grid = jnp.full((T, H1, W1, S1), 1.0 / S1, dtype=jnp.float32)

    # At present, level 1 does not have a real path factor (num_paths=1),
    # so B_states_paths is effectively (S1, S1, 1) and qu_top is used only
    # when paths are attached at a higher level in future refactors.
    use_path_dependent_B1 = (
        B_states_paths is not None
        and qu_top is not None
        and B_states_paths.shape[2] > 1
    )

    # ----- Level-1 update: per-site chains of length T -----

    def update_level1(
        qs0_grid_current: jnp.ndarray,
        qs1_grid_current: jnp.ndarray,
        qu_current: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        # Bottom-up log-likelihood from level 0
        log_lik1 = bottom_up_message_level0_to_level1(qs0_grid_current, D1, states_grid1)
        # log_lik1: (T, H1, W1, S1)

        def update_site_chain(
            qs1_chain: jnp.ndarray,
            log_lik_chain: jnp.ndarray,
        ) -> jnp.ndarray:
            qs_chain_updated = vmp_temporal_chain(
                log_lik_chain=log_lik_chain,
                B_chain=B1,
                E=E1,
                num_iter=2,
                q_init=qs1_chain,
            )
            return qs_chain_updated

        H1_local, W1_local = qs1_grid_current.shape[1], qs1_grid_current.shape[2]
        h_indices = jnp.arange(H1_local)
        w_indices = jnp.arange(W1_local)

        def update_site_for_row(h_idx, qs1_grid_current, log_lik1):
            def update_site_col(w_idx):
                qs1_chain = qs1_grid_current[:, h_idx, w_idx, :]   # (T, S1)
                log_lik_chain = log_lik1[:, h_idx, w_idx, :]       # (T, S1)
                return update_site_chain(qs1_chain, log_lik_chain) # (T, S1)
            return jax.vmap(update_site_col)(w_indices)  # (W1, T, S1)

        qs1_rows = jax.vmap(
            lambda h_idx: update_site_for_row(h_idx, qs1_grid_current, log_lik1)
        )(h_indices)  # (H1, W1, T, S1)

        qs1_grid_new = jnp.transpose(qs1_rows, (2, 0, 1, 3))  # (T, H1, W1, S1)
        return qs1_grid_new

    # ----- Level-0 update: per-patch chains with top-down bias -----

    def update_level0(
        qs0_grid_current: jnp.ndarray,
        qs1_grid_current: jnp.ndarray,
    ) -> jnp.ndarray:
        H1_local, W1_local = qs1_grid_current.shape[1], qs1_grid_current.shape[2]

        def build_topdown_prior() -> jnp.ndarray:
            def bias_for_time(t):
                qs1_t = qs1_grid_current[t]  # (H1, W1, S1)
                bias = jnp.zeros((H0, W0, S0), dtype=jnp.float32)

                for h1 in range(H1_local):
                    for w1 in range(W1_local):
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

        log_prior_bias0 = build_topdown_prior()  # (T, H0, W0, S0)

        def update_site_chain(
            qs0_chain: jnp.ndarray,
            log_bias_chain: jnp.ndarray,
        ) -> jnp.ndarray:
            log_lik_chain = log_bias_chain  # (T, S0)
            qs_chain_updated = vmp_temporal_chain(
                log_lik_chain=log_lik_chain,
                B_chain=B0,
                E=E0,
                num_iter=1,
                q_init=qs0_chain,
            )
            return qs_chain_updated

        update_site_chain_vmap = jax.vmap(
            jax.vmap(update_site_chain, in_axes=(1, 1), out_axes=1),
            in_axes=(1, 1), out_axes=1,
        )
        qs0_grid_new = update_site_chain_vmap(qs0_grid_current, log_prior_bias0)
        return qs0_grid_new

    qs0_current = qs0_grid
    qs1_current = qs1_grid

    def alt_step(_, carry):
        qs0_c, qs1_c = carry
        qs1_new = update_level1(qs0_c, qs1_c, qu_top)
        qs0_new = update_level0(qs0_c, qs1_new)
        return (qs0_new, qs1_new)

    qs0_final, qs1_final = jax.lax.fori_loop(0, num_iter, alt_step, (qs0_current, qs1_current))

    return qs0_final, qs1_final


# -----------------------------------------------------------------------------
# 7. High-level inference entry point with EFE-based paths
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
         G_tu for paths via lorenz_efe and update q(u_t) using G and
         B_paths/E_paths, alternating with state updates.

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
    """
    T = hierarchy.T
    H0 = hierarchy.H_blocks
    W0 = hierarchy.W_blocks

    level0: LorenzLevel = hierarchy.levels[0]
    A0 = level0.A
    O = A0.shape[1]

    obs_flat = build_lowest_level_observations_flat(lorenz_data_dict)  # (N, O)
    N = obs_flat.shape[0]
    assert N == T * H0 * W0, "Observation length mismatch."

    C = build_preference_distribution_lowest(O, mode=pref_mode, obs_flat=obs_flat)

    # 1. Infer Level-0 states patch-wise
    qs0_grid = infer_lowest_level_patches(
        level0, lorenz_data_dict, num_iter_lowest=num_iter_lowest
    )

    qs_levels: List[Optional[jnp.ndarray]] = [qs0_grid]
    qu_levels: List[Optional[jnp.ndarray]] = [None]

    qs1_grid = None
    qu_top = None

    if len(hierarchy.levels) > 1:
        level1: LorenzLevel = hierarchy.levels[1]
        states_grid1 = hierarchy.states_grids[1]

        # Top-level for paths is currently the highest level in the hierarchy.
        top_idx = len(hierarchy.levels) - 1
        level_top = hierarchy.levels[top_idx]

        if level_top.num_paths > 1:
            U = level_top.num_paths
            qu_top = jnp.full((T, U), 1.0 / U, dtype=jnp.float32)
        else:
            qu_top = None

        qs0_current = qs0_grid
        qs1_current = jnp.full(
            (T, states_grid1.shape[1], states_grid1.shape[2], level1.S),
            1.0 / level1.S,
            dtype=jnp.float32,
        )

        for _ in range(num_iter_hier):
            # Two-level state updates
            qs0_current, qs1_current = vmp_two_level_states(
                level0,
                level1,
                qs0_current,
                states_grid1,
                qu_top=qu_top,
                num_iter=1,
            )

            # Path updates (only if num_paths_top > 1 at top level)
            if level_top.num_paths > 1:
                G_tu = compute_expected_free_energy_paths(
                    level_top,
                    level0,
                    qs1_current,
                    qs0_current,
                    C,
                    tau=3,  # can be passed through if needed
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
