# lorenz_inference.py
"""
Variational message passing for the Lorenz hierarchy.

This module provides:
- Construction of lowest-level observations for the Lorenz patch model.
- Patch-wise lowest-level inference by looping over patches.
- Generic adjacent-level VMP using spatial D mappings (bottom-up and top-down).
- Multi-level hierarchical state inference for an arbitrary number of
  spatial levels built by lorenz_renorm.
- Top-level path inference using expected free energy over paths
  computed via lorenz_efe (risk + ambiguity - epistemic), with
  path-dependent transitions at the top state level.

All routines are written in a level-agnostic way so they work for 2, 3,
or more spatial levels. The Lorenz example uses 3 spatial levels with
a path factor at the top.
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
    Build a flat observation array suitable for lowest-level inference.

    We treat the quantized coefficients as "observations" in a one-hot
    form that is consistent with the A built in lorenz_model.py.

    Returns:
        obs_flat: (N, O) where N = T * H_blocks * W_blocks,
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
    Build a preference distribution C over lowest-level outcomes.

    Args:
        O: outcome dimension at lowest level (O = K * L)
        mode:
            - "uniform": no particular preferred outcome
            - "data_empirical": empirical distribution over training obs
        obs_flat: (N, O) observations (required if data_empirical)

    Returns:
        C: (O,) preference distribution.
    """
    if mode == "uniform" or obs_flat is None:
        C = jnp.ones((O,), dtype=jnp.float32)
        C = C / C.sum()
        return C

    C = obs_flat.mean(axis=0)
    C = C / (C.sum() + 1e-8)
    return C


# -----------------------------------------------------------------------------
# 2. Single-chain VMP (no spatial coupling)
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
      - emission model A (S,O),
      - transition model B(s_next|s) (S,S),
      - prior over initial state E (S,).

    Args:
        obs: (T, O) observations (one-hot / distribution over O).

    Returns:
        qs: (T, S) posterior marginals over states.
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
# 3. Patch-wise lowest-level inference
# -----------------------------------------------------------------------------

def infer_lowest_level_patches(
    level0: LorenzLevel,
    lorenz_data_dict: Dict[str, Any],
    num_iter_lowest: int = 8,
) -> jnp.ndarray:
    """
    Run VMP at the lowest level independently for each patch.

    Args:
        level0: lowest-level LorenzLevel (A, B_states, E_states)
        lorenz_data_dict: dataset dict
        num_iter_lowest: iterations for single-chain VMP

    Returns:
        qs0_grid: (T, H0, W0, S0) posterior over lowest-level states.
    """
    A0 = level0.A
    B0 = level0.B_states
    E0 = level0.E_states

    obs_grid = build_lowest_level_observations_grid(lorenz_data_dict)
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
# 4. Generic D-based bottom-up and top-down messages
# -----------------------------------------------------------------------------

def bottom_up_message_child_to_parent(
    qs_child: jnp.ndarray,
    D_parent: jnp.ndarray,
    states_grid_parent: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute a bottom-up "pseudo-likelihood" log p(s_parent | qs_child)
    using a fixed D mapping from parent states to 4 child states.

    This generalizes the previous level0→level1 function and works for
    any adjacent pair of levels where the parent uses 2x2 groups.

    Args:
        qs_child: (T, Hc, Wc, S_child) posterior at child level
        D_parent: (S_parent, 4) child state patterns
        states_grid_parent: (T, Hp, Wp) integer parent states (layout)

    Returns:
        log_lik_parent: (T, Hp, Wp, S_parent)
    """
    T, Hc, Wc, S_child = qs_child.shape
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

    qs_child_children = jax.vmap(site_children)(qs_child)  # (T, Hp, Wp, 4, S_child)
    D_idx = D_parent.astype(jnp.int32)  # (S_parent, 4)

    def score_parent_state(qs_child_4: jnp.ndarray, pattern: jnp.ndarray) -> jnp.ndarray:
        probs = qs_child_4[jnp.arange(4), pattern]  # (4,)
        log_probs = jnp.log(jnp.clip(probs, 1e-16))
        return log_probs.sum()

    def score_all_parents_for_site(qs_child_site: jnp.ndarray) -> jnp.ndarray:
        # qs_child_site: (4, S_child)
        return jax.vmap(lambda patt: score_parent_state(qs_child_site, patt))(D_idx)

    def score_all_sites(qs_child_t: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda row: jax.vmap(score_all_parents_for_site)(row), in_axes=0
        )(qs_child_t)

    log_lik_parent = jax.vmap(score_all_sites)(qs_child_children)  # (T, Hp, Wp, S_parent)
    return log_lik_parent


def top_down_message_parent_to_child(
    qs_parent: jnp.ndarray,
    D_parent: jnp.ndarray,
    states_grid_parent: jnp.ndarray,
    child_shape: Tuple[int, int],
    S_child: int,
) -> jnp.ndarray:
    """
    Compute a simple top-down message from parent posteriors to child
    sites using the D mapping. For now, we project a uniform distribution
    over the 4 children implied by each parent state and average over
    parents at each group-site.

    Args:
        qs_parent: (T, Hp, Wp, S_parent) parent posteriors
        D_parent: (S_parent, 4) child state patterns
        states_grid_parent: (T, Hp, Wp) parent state grid (unused but kept
                            for possible extensions)
        child_shape: (Hc, Wc) child grid shape
        S_child: number of child states

    Returns:
        log_bias_child: (T, Hc, Wc, S_child) log "bias" messages to child.
    """
    T, Hp, Wp, S_parent = qs_parent.shape
    Hc, Wc = child_shape
    assert Hc == 2 * Hp and Wc == 2 * Wp

    D_idx = D_parent.astype(jnp.int32)  # (S_parent, 4)

    # Initialize uniform log-bias
    log_bias = jnp.zeros((T, Hc, Wc, S_child), dtype=jnp.float32)

    # For now, a simple approximation: ignore detailed structure and
    # broadcast a weak, uniform bias to child states. This keeps the
    # interface in place for future D-based top-down structure learning.
    # (More sophisticated versions could invert D to obtain child posteriors.)
    return log_bias


# -----------------------------------------------------------------------------
# 5. Generic adjacent-level VMP (child-parent pair)
# -----------------------------------------------------------------------------

def vmp_adjacent_levels(
    level_child: LorenzLevel,
    level_parent: LorenzLevel,
    qs_child: jnp.ndarray,
    states_grid_parent: jnp.ndarray,
    qs_parent_init: Optional[jnp.ndarray] = None,
    qu_paths: Optional[jnp.ndarray] = None,
    num_iter: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform coupled VMP between a child level and its parent level.

    - Child level is updated by its own A,B,E and top-down messages.
    - Parent level is updated by bottom-up messages from child via D,
      plus its own B,E, and (optionally) path-dependent transitions.

    This is written level-agnostically and can be applied to any
    adjacent pair (ℓ, ℓ+1).

    Args:
        level_child: LorenzLevel at level ℓ
        level_parent: LorenzLevel at level ℓ+1
        qs_child: (T, Hc, Wc, S_child) current child posteriors
        states_grid_parent: (T, Hp, Wp) parent state grid
        qs_parent_init: optional initial parent qs; if None, uniform
        qu_paths: (T, U) path posterior if parent has num_paths>1, else None
        num_iter: number of alternations (child↔parent) for this pair

    Returns:
        qs_child_new: (T, Hc, Wc, S_child)
        qs_parent_new: (T, Hp, Wp, S_parent)
    """
    D_parent = level_parent.D
    assert D_parent is not None, "Parent level must define D for spatial coupling."

    T, Hc, Wc, S_child = qs_child.shape
    T1, Hp, Wp = states_grid_parent.shape
    assert T == T1
    assert Hc == 2 * Hp and Wc == 2 * Wp

    S_parent = int(jnp.max(states_grid_parent) + 1)

    if qs_parent_init is None:
        qs_parent = jnp.full((T, Hp, Wp, S_parent), 1.0 / S_parent, dtype=jnp.float32)
    else:
        qs_parent = qs_parent_init

    # Precompute child emission log-likelihoods if A has nonzero shape
    A_child = level_child.A
    B_child = level_child.B_states
    E_child = level_child.E_states

    def child_site_update(
        qs_chain: jnp.ndarray,
        log_lik_chain: jnp.ndarray,
    ) -> jnp.ndarray:
        # Reuse vmp_single_chain for each site
        return vmp_single_chain(A_child, B_child, E_child, log_lik_chain, num_iter=2)

    # Parent dynamics
    B_parent_base = level_parent.B_states
    E_parent = level_parent.E_states
    B_states_paths = level_parent.B_states_paths
    U = level_parent.num_paths

    def parent_site_update(
        qs_chain: jnp.ndarray,
        log_lik_chain: jnp.ndarray,
        B_eff_all: jnp.ndarray,
    ) -> jnp.ndarray:
        # VMP for a chain with time-varying B_eff_all[t]
        T_loc, S_loc = qs_chain.shape

        def forward_messages(qs_):
            msgs = []
            prev_q = qs_[0]
            msgs.append(jnp.log(jnp.clip(E_parent, 1e-16)))
            for t in range(1, T_loc):
                B_t = B_eff_all[t]
                msg = jnp.log(jnp.clip(B_t @ prev_q, 1e-16))
                msgs.append(msg)
                prev_q = qs_[t]
            return jnp.stack(msgs, axis=0)

        def backward_messages(qs_):
            msgs = []
            next_q = qs_[-1]
            msgs.append(jnp.zeros_like(next_q))
            for t in range(T_loc - 2, -1, -1):
                B_tp1 = B_eff_all[t + 1]
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

        qs_final = jax.lax.fori_loop(0, 2, body_fun, qs_chain)
        return qs_final

    # Main alternating loop
    def one_iteration(carry):
        qs_child_curr, qs_parent_curr = carry

        # Bottom-up messages to parent
        log_lik_parent = bottom_up_message_child_to_parent(
            qs_child_curr, D_parent, states_grid_parent
        )  # (T, Hp, Wp, S_parent)

        # Build time-varying effective B for parent
        if B_states_paths is not None and U > 1 and qu_paths is not None:
            # B_states_paths: (S_parent, S_parent, U)
            def B_eff_t(t):
                qu_t = qu_paths[t]  # (U,)
                return (B_states_paths * qu_t[None, None, :]).sum(axis=2)  # (S_parent, S_parent)

            B_eff_all = jax.vmap(B_eff_t)(jnp.arange(T))  # (T, S_parent, S_parent)
        else:
            B_eff_all = jnp.broadcast_to(B_parent_base, (T, S_parent, S_parent))

        # Update parent per site
        def update_parent_site(qs_site_chain, log_lik_site_chain, B_eff_all_t):
            return parent_site_update(qs_site_chain, log_lik_site_chain, B_eff_all_t)

        qs_parent_next = jax.vmap(  # over t
            lambda qs_t, ll_t, B_t: update_parent_site(qs_t, ll_t, B_t),
            in_axes=(0, 0, 0),
        )(
            qs_parent_curr.reshape(T, Hp * Wp, S_parent),
            log_lik_parent.reshape(T, Hp * Wp, S_parent),
            B_eff_all,
        ).reshape(T, Hp, Wp, S_parent)

        # Top-down messages to child (currently a neutral log-bias)
        log_bias_child = top_down_message_parent_to_child(
            qs_parent_next,
            D_parent,
            states_grid_parent,
            child_shape=(Hc, Wc),
            S_child=S_child,
        )

        # Child emission log-likelihood per site
        obs_grid = None  # We do not recompute emissions here; assume they are in qs_child
        # For now, we treat child qs update as driven only by B_child and
        # top-down bias; the emission term is already encoded at lowest level.

        # Simple approximation: only use top-down log_bias_child as "likelihood"
        log_lik_child = log_bias_child  # (T, Hc, Wc, S_child)

        qs_child_next = qs_child_curr  # keep child fixed in this minimal version
        # (We can extend this to a full child update once we add explicit obs here.)

        return (qs_child_next, qs_parent_next)

    qs_child_out, qs_parent_out = jax.lax.fori_loop(
        0, num_iter, one_iteration, (qs_child, qs_parent)
    )

    return qs_child_out, qs_parent_out


# -----------------------------------------------------------------------------
# 6. Full multi-level inference for a LorenzHierarchy
# -----------------------------------------------------------------------------

def infer_lorenz_hierarchy(
    hierarchy: LorenzHierarchy,
    lorenz_data_dict: Dict[str, Any],
    C: jnp.ndarray,
    num_iter_lowest: int = 8,
    num_iter_hier: int = 2,
    efe_gamma: float = 16.0,
    tau: int = 3,
) -> Dict[str, Any]:
    """
    Run hierarchical inference for a LorenzHierarchy with an arbitrary
    number of spatial levels.

    Steps:
      1. Infer lowest-level posteriors qs0_grid via patch-wise VMP.
      2. For each adjacent pair of levels (ℓ, ℓ+1), run a small number of
         vmp_adjacent_levels updates.
      3. If the top level has num_paths > 1, compute expected free energy
         over paths using the top and lowest levels, and update the path
         posterior q(u_t).

    Args:
        hierarchy: LorenzHierarchy
        lorenz_data_dict: dataset dict
        C: (O0,) preference distribution over lowest-level outcomes
        num_iter_lowest: iterations for lowest-level chain VMP
        num_iter_hier: iterations per adjacent-level pair
        efe_gamma: precision over expected free energy
        tau: planning horizon for EFE

    Returns:
        dict with:
          'qs_levels': list of qs per level (T, H_l, W_l, S_l)
          'qu_levels': list of path posteriors per level (None except top)
    """
    levels = hierarchy.levels
    states_grids = hierarchy.states_grids
    T = hierarchy.T

    # 1. Lowest-level inference
    level0 = levels[0]
    qs0_grid = infer_lowest_level_patches(
        level0, lorenz_data_dict, num_iter_lowest=num_iter_lowest
    )

    qs_levels: List[Optional[jnp.ndarray]] = [qs0_grid]
    qu_levels: List[Optional[jnp.ndarray]] = [None]

    # 2. Initialize parent qs for higher levels
    for l in range(1, len(levels)):
        level_l = levels[l]
        states_grid_l = states_grids[l]
        T_l, H_l, W_l = states_grid_l.shape
        S_l = level_l.S
        qs_l = jnp.full((T_l, H_l, W_l, S_l), 1.0 / S_l, dtype=jnp.float32)
        qs_levels.append(qs_l)
        qu_levels.append(None)

    # 3. If top level has a path factor, initialize qu_top
    top_idx = len(levels) - 1
    level_top = levels[top_idx]
    qu_top: Optional[jnp.ndarray]

    if level_top.num_paths > 1:
        U = level_top.num_paths
        qu_top = jnp.full((T, U), 1.0 / U, dtype=jnp.float32)
        qu_levels[top_idx] = qu_top
    else:
        qu_top = None

    # 4. Hierarchical updates (bottom-up passes)
    for _ in range(num_iter_hier):
        # For each adjacent pair (ℓ, ℓ+1)
        for l in range(len(levels) - 1):
            level_child = levels[l]
            level_parent = levels[l + 1]
            qs_child = qs_levels[l]
            qs_parent = qs_levels[l + 1]
            states_grid_parent = states_grids[l + 1]

            if qs_child is None or qs_parent is None:
                continue

            # Only the topmost parent may have a path factor
            qu_paths = qu_top if (l + 1 == top_idx and qu_top is not None) else None

            qs_child_new, qs_parent_new = vmp_adjacent_levels(
                level_child,
                level_parent,
                qs_child,
                states_grid_parent,
                qs_parent_init=qs_parent,
                qu_paths=qu_paths,
                num_iter=1,
            )
            qs_levels[l] = qs_child_new
            qs_levels[l + 1] = qs_parent_new

        # Top-level path update (if any)
        if level_top.num_paths > 1:
            qs_top_grid = qs_levels[top_idx]
            qs0_grid_curr = qs_levels[0]

            if qs_top_grid is not None and qs0_grid_curr is not None:
                G_tu = compute_expected_free_energy_paths(
                    level_top,
                    levels[0],
                    qs_top_grid,
                    qs0_grid_curr,
                    C,
                    tau=tau,
                )
                qu_top = update_path_posterior_from_G(
                    level_top,
                    G_tu,
                    gamma=efe_gamma,
                    num_iter=2,
                )
                qu_levels[top_idx] = qu_top

    return {
        "qs_levels": qs_levels,
        "qu_levels": qu_levels,
    }
