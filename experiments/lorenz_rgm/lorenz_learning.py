# lorenz_learning.py
"""
Learning routines for the Lorenz RGM.

This module implements:
  - Accumulation of Dirichlet counts for A, B, E, and C from posterior
    beliefs over states and observations.
  - Updates of Dirichlet concentration parameters (A_alpha, B_alpha, E_alpha,
    C_alpha) stored in LorenzRGMParams.
  - Accumulation and updates for path-related Dirichlet parameters:
      * B_states_paths_alpha: path-dependent state transitions at top level
      * B_paths_alpha: path transition dynamics
      * E_paths_alpha: initial path priors
  - A training loop scaffold that alternates between inference and parameter
    updates over one or more Lorenz trajectories.

It assumes:
  - The generative model structure (levels, D matrices, path factors)
    is defined in lorenz_model.py.
  - Inference over states and paths is provided by lorenz_inference.py.
"""

from typing import Dict, Any, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .lorenz_model import (
    LorenzHierarchy,
    LorenzRGMParams,
    build_lorenz_hierarchy_from_params,
)
from .lorenz_inference import (
    build_lowest_level_observations_grid,
    infer_lorenz_hierarchy,
)


# -----------------------------------------------------------------------------
# 1. Sufficient statistics for lowest level (A0, B0, E0, C0)
# -----------------------------------------------------------------------------

def accumulate_A_counts_level0(
    qs0_grid: jnp.ndarray,
    obs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for A0 (lowest-level emission matrix).

    Args:
        qs0_grid: (T, H0, W0, S0) posterior over lowest-level states
        obs_grid: (T, H0, W0, O0) one-hot/distribution over outcomes

    Returns:
        dA0: (S0, O0) Dirichlet count increments for A0
    """
    T, H0, W0, S0 = qs0_grid.shape
    _, _, _, O0 = obs_grid.shape

    qs_flat = qs0_grid.reshape(T * H0 * W0, S0)   # (N, S0)
    obs_flat = obs_grid.reshape(T * H0 * W0, O0)  # (N, O0)

    dA0 = qs_flat.T @ obs_flat  # (S0, O0)
    return dA0


def accumulate_B_counts_level0(
    qs0_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for B0 (lowest-level transitions).

    Approximation:
      dB[s,s'] = sum_{t,h,w} q(s_t = s) * q(s_{t+1} = s').

    Args:
        qs0_grid: (T, H0, W0, S0)

    Returns:
        dB0: (S0, S0) Dirichlet count increments for B0
    """
    T, H0, W0, S0 = qs0_grid.shape

    qt = qs0_grid[:-1].reshape((T - 1) * H0 * W0, S0)   # (N_tr, S0)
    qt1 = qs0_grid[1:].reshape((T - 1) * H0 * W0, S0)   # (N_tr, S0)

    dB0 = qt.T @ qt1  # (S0, S0)
    return dB0


def accumulate_E_counts_level0(
    qs0_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for E0 (prior over initial states).

    Args:
        qs0_grid: (T, H0, W0, S0)

    Returns:
        dE0: (S0,) Dirichlet count increments for E0
    """
    q0 = qs0_grid[0]  # (H0, W0, S0)
    H0, W0, S0 = q0.shape
    q0_flat = q0.reshape(H0 * W0, S0)
    dE0 = q0_flat.sum(axis=0)  # (S0,)
    return dE0


def accumulate_C_counts(
    obs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate counts for lowest-level preferences C0(o) from observations.

    Args:
        obs_grid: (T, H0, W0, O0) one-hot/distribution over outcomes

    Returns:
        dC0: (O0,) Dirichlet count increments for C0
    """
    dC0 = obs_grid.sum(axis=(0, 1, 2))  # (O0,)
    return dC0


# -----------------------------------------------------------------------------
# 2. Sufficient statistics for higher level (B1, E1)
# -----------------------------------------------------------------------------

def accumulate_B_counts_level1(
    qs1_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for B1 (level-1 transitions).

    Args:
        qs1_grid: (T, H1, W1, S1)

    Returns:
        dB1: (S1, S1) Dirichlet count increments for B1
    """
    T, H1, W1, S1 = qs1_grid.shape

    qt = qs1_grid[:-1].reshape((T - 1) * H1 * W1, S1)  # (N_tr, S1)
    qt1 = qs1_grid[1:].reshape((T - 1) * H1 * W1, S1)  # (N_tr, S1)

    dB1 = qt.T @ qt1  # (S1, S1)
    return dB1


def accumulate_E_counts_level1(
    qs1_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for E1 (prior over initial states at level 1).

    Args:
        qs1_grid: (T, H1, W1, S1)

    Returns:
        dE1: (S1,) Dirichlet count increments for E1
    """
    q0 = qs1_grid[0]  # (H1, W1, S1)
    H1, W1, S1 = q0.shape
    q0_flat = q0.reshape(H1 * W1, S1)
    dE1 = q0_flat.sum(axis=0)  # (S1,)
    return dE1


# -----------------------------------------------------------------------------
# 3. Sufficient statistics for path-dependent state transitions and paths
# -----------------------------------------------------------------------------

def accumulate_B_states_paths_counts_level1(
    qs1_grid: jnp.ndarray,
    qu_top: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for path-dependent state transitions
    B_states_paths[u, s, s'] at the top level.

    Approximate formula:
      dB_states_paths[u,s,s'] ≈ sum_t q(u_t = u)
                                  * sum_{h1,w1} q(s^1_{t,h1,w1}=s)
                                               * q(s^1_{t+1,h1,w1}=s').

    For now we use a simple approximation where we compute a base transition
    frequency over level-1 states (independent of u) and weight it by the
    average q(u_t = u) over time.

    Args:
        qs1_grid: (T, H1, W1, S1)
        qu_top: (T, U) posterior over paths at top level

    Returns:
        dB_states_paths: (U, S1, S1) Dirichlet count increments
    """
    T, H1, W1, S1 = qs1_grid.shape
    Tq, U = qu_top.shape
    assert Tq == T, "qu_top and qs1_grid must have same T."

    qt = qs1_grid[:-1].reshape((T - 1) * H1 * W1, S1)   # (N_tr, S1)
    qt1 = qs1_grid[1:].reshape((T - 1) * H1 * W1, S1)   # (N_tr, S1)
    base_trans = qt.T @ qt1                             # (S1, S1)

    # Average path posterior over time steps where transitions occur (0..T-2)
    qu_avg = qu_top[:-1].mean(axis=0)                   # (U,)

    dB_list = []
    for u in range(U):
        dB_list.append(base_trans * qu_avg[u])
    dB_states_paths = jnp.stack(dB_list, axis=0)        # (U, S1, S1)
    return dB_states_paths


def accumulate_B_E_paths(
    qu_top: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Accumulate Dirichlet counts for path transitions B_paths and
    initial path prior E_paths.

    Args:
        qu_top: (T, U) posterior over paths at top level

    Returns:
        dB_paths: (U, U) Dirichlet count increments for B_paths
        dE_paths: (U,) Dirichlet count increments for E_paths
    """
    T, U = qu_top.shape
    qu_t = qu_top[:-1]  # (T-1, U)
    qu_t1 = qu_top[1:]  # (T-1, U)

    # Approximate joint q(u_t, u_{t+1}) as outer product of marginals
    dB_paths = qu_t.T @ qu_t1   # (U, U)
    dE_paths = qu_top[0]        # (U,)
    return dB_paths, dE_paths


# -----------------------------------------------------------------------------
# 4. Update Dirichlet parameters from a single sequence
# -----------------------------------------------------------------------------

def update_dirichlet_from_sequence(
    hierarchy: LorenzHierarchy,
    params: LorenzRGMParams,
    lorenz_data_dict: Dict[str, Any],
    num_iter_lowest: int,
    num_iter_hier: int,
    efe_gamma: float,
    pref_mode: str,
) -> LorenzRGMParams:
    """
    Run inference for a single Lorenz sequence and update Dirichlet
    concentration parameters A_alpha, B_alpha, E_alpha, C_alpha,
    and, if available, B_states_paths_alpha, B_paths_alpha, E_paths_alpha.

    This function:
      1. Builds observations for this sequence.
      2. Runs infer_lorenz_hierarchy with the current hierarchy.
      3. Accumulates counts for:
           - A0, B0, E0, C0 at level 0
           - B1, E1 at level 1 (if present)
           - path-dependent state transitions B_states_paths at top level (if present)
           - path transitions B_paths and initial path prior E_paths at top level
      4. Adds these counts to the corresponding alphas.

    Args:
        hierarchy: LorenzHierarchy with current A/B/E/D/path structure
        params: LorenzRGMParams with current alphas
        lorenz_data_dict: data dict for this sequence
        num_iter_lowest: iterations for lowest-level VMP
        num_iter_hier: iterations for hierarchical VMP
        efe_gamma: precision over expected free energy for paths
        pref_mode: preference mode (used by infer_lorenz_hierarchy)

    Returns:
        Updated LorenzRGMParams
    """
    # 1. Observations as grid
    obs_grid = build_lowest_level_observations_grid(lorenz_data_dict)  # (T, H0, W0, O0)

    # 2. Inference
    results = infer_lorenz_hierarchy(
        hierarchy,
        lorenz_data_dict,
        num_iter_lowest=num_iter_lowest,
        num_iter_hier=num_iter_hier,
        efe_gamma=efe_gamma,
        pref_mode=pref_mode,
    )
    qs_levels = results["qs_levels"]
    qu_levels = results["qu_levels"]

    qs0_grid = qs_levels[0]                      # (T, H0, W0, S0)
    qs1_grid = qs_levels[1] if len(qs_levels) > 1 else None
    qu_top = qu_levels[-1] if len(qu_levels) > 1 else None

    # 3. Accumulate counts for level 0
    dA0 = accumulate_A_counts_level0(qs0_grid, obs_grid)
    dB0 = accumulate_B_counts_level0(qs0_grid)
    dE0 = accumulate_E_counts_level0(qs0_grid)
    dC0 = accumulate_C_counts(obs_grid)

    params.A_alpha[0] = params.A_alpha[0] + dA0
    params.B_alpha[0] = params.B_alpha[0] + dB0
    params.E_alpha[0] = params.E_alpha[0] + dE0
    params.C_alpha = params.C_alpha + dC0

    # 4. Accumulate counts for level 1 (if present)
    if qs1_grid is not None and len(params.B_alpha) > 1:
        dB1 = accumulate_B_counts_level1(qs1_grid)
        dE1 = accumulate_E_counts_level1(qs1_grid)
        params.B_alpha[1] = params.B_alpha[1] + dB1
        params.E_alpha[1] = params.E_alpha[1] + dE1

    # 5. Path-related counts at top level (if path factor present)
    if qu_top is not None and params.B_paths_alpha is not None and params.E_paths_alpha is not None:
        dB_paths, dE_paths = accumulate_B_E_paths(qu_top)
        params.B_paths_alpha = params.B_paths_alpha + dB_paths
        params.E_paths_alpha = params.E_paths_alpha + dE_paths

    # 6. Path-dependent state transitions at top level (if enabled)
    if (
        qs1_grid is not None
        and qu_top is not None
        and params.B_states_paths_alpha is not None
    ):
        dB_states_paths = accumulate_B_states_paths_counts_level1(qs1_grid, qu_top)
        params.B_states_paths_alpha = params.B_states_paths_alpha + dB_states_paths

    return params


# -----------------------------------------------------------------------------
# 5. Normalize Dirichlet parameters to get A/B/E/C/B_states_paths/B_paths/E_paths
# -----------------------------------------------------------------------------

def params_to_categorical(
    params: LorenzRGMParams,
    K: int,
    L: int,
) -> Tuple[
    List[jnp.ndarray],
    List[jnp.ndarray],
    List[jnp.ndarray],
    jnp.ndarray,
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
    Optional[jnp.ndarray],
]:
    """
    Convert Dirichlet concentration parameters into categorical parameters
    for inference: A, B, E, C, and (if present) B_states_paths, B_paths,
    E_paths for the top level.

    Args:
        params: LorenzRGMParams
        K, L: for lowest level O0 = K * L

    Returns:
        A_list: list of A^l matrices (S_l, O_l)
        B_list: list of B^l matrices (S_l, S_l)
        E_list: list of E^l vectors (S_l,)
        C0: preferences over lowest outcomes (O0,)
        B_states_paths: (U, S_top, S_top) or None
        B_paths: (U, U) or None
        E_paths: (U,) or None
    """
    A_list: List[jnp.ndarray] = []
    B_list: List[jnp.ndarray] = []
    E_list: List[jnp.ndarray] = []

    for A_alpha_l, B_alpha_l, E_alpha_l in zip(
        params.A_alpha, params.B_alpha, params.E_alpha
    ):
        if A_alpha_l.size > 0:
            A_l = A_alpha_l / (A_alpha_l.sum(axis=1, keepdims=True) + 1e-8)
        else:
            A_l = A_alpha_l

        B_l = B_alpha_l / (B_alpha_l.sum(axis=1, keepdims=True) + 1e-8)
        E_l = E_alpha_l / (E_alpha_l.sum() + 1e-8)

        A_list.append(A_l)
        B_list.append(B_l)
        E_list.append(E_l)

    O0 = K * L
    C0 = params.C_alpha / (params.C_alpha.sum() + 1e-8)
    assert C0.shape[0] == O0, "C0 shape mismatch with K*L."

    # Path-related categorical parameters
    if params.B_paths_alpha is not None and params.E_paths_alpha is not None:
        B_paths = params.B_paths_alpha / (
            params.B_paths_alpha.sum(axis=1, keepdims=True) + 1e-8
        )
        E_paths = params.E_paths_alpha / (params.E_paths_alpha.sum() + 1e-8)
    else:
        B_paths, E_paths = None, None

    if params.B_states_paths_alpha is not None:
        B_states_paths = params.B_states_paths_alpha / (
            params.B_states_paths_alpha.sum(axis=2, keepdims=True) + 1e-8
        )
    else:
        B_states_paths = None

    return A_list, B_list, E_list, C0, B_states_paths, B_paths, E_paths


# -----------------------------------------------------------------------------
# 6. Training loop scaffold
# -----------------------------------------------------------------------------

def train_lorenz_rgm(
    initial_params: LorenzRGMParams,
    lorenz_spatial_hierarchy: Dict[str, Any],
    build_data_fn,
    K: int,
    L: int,
    num_epochs: int = 1,
    num_sequences_per_epoch: int = 1,
    num_iter_lowest: int = 8,
    num_iter_hier: int = 4,
    efe_gamma: float = 16.0,
    pref_mode: str = "data_empirical",
    num_paths_top: int = 4,
) -> LorenzRGMParams:
    """
    High-level training loop scaffold for the Lorenz RGM.

    At each epoch and for each sequence:
      1. Build Lorenz data (via build_data_fn).
      2. Build a hierarchy from current params and spatial hierarchy.
      3. Run inference (infer_lorenz_hierarchy).
      4. Update Dirichlet parameters from this sequence.

    Args:
        initial_params: LorenzRGMParams with initial Dirichlet alphas
        lorenz_spatial_hierarchy: dict from build_lorenz_spatial_hierarchy
        build_data_fn: function that returns a lorenz_data_dict for each sequence
        K, L: discrete coefficient configuration
        num_epochs: number of training epochs
        num_sequences_per_epoch: sequences per epoch
        num_iter_lowest: lowest-level VMP iterations
        num_iter_hier: hierarchical VMP iterations
        efe_gamma: precision over expected free energy for paths
        pref_mode: preference mode for inference
        num_paths_top: number of path states at top level

    Returns:
        Trained LorenzRGMParams
    """
    params = initial_params

    for epoch in range(num_epochs):
        for seq_idx in range(num_sequences_per_epoch):
            # 1. Build data for this sequence
            lorenz_data_dict = build_data_fn()

            # 2. Build hierarchy from current params and spatial structure
            hierarchy = build_lorenz_hierarchy_from_params(
                lorenz_spatial_hierarchy,
                params,
                num_paths_top=num_paths_top,
            )

            # 3. Update Dirichlet parameters from this sequence
            params = update_dirichlet_from_sequence(
                hierarchy,
                params,
                lorenz_data_dict,
                num_iter_lowest=num_iter_lowest,
                num_iter_hier=num_iter_hier,
                efe_gamma=efe_gamma,
                pref_mode=pref_mode,
            )

    return params
