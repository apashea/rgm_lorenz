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
  * B_paths_alpha: path transition dynamics at top level
  * E_paths_alpha: initial path priors at top level
- A training loop scaffold that alternates between inference and parameter
  updates over one or more Lorenz trajectories.

The model structure (levels, D matrices, path factors) is defined in
lorenz_model.py. Inference over states and paths is provided by
lorenz_inference.py.
"""

from typing import Dict, Any, List, Tuple, Optional

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
        obs_grid: (T, H0, W0, O0) outcomes (one-hot or distributions)

    Returns:
        dA0: (S0, O0) Dirichlet count increments for A0
    """
    T, H0, W0, S0 = qs0_grid.shape
    _, _, _, O0 = obs_grid.shape

    qs_flat = qs0_grid.reshape(T * H0 * W0, S0)   # (N, S0)
    obs_flat = obs_grid.reshape(T * H0 * W0, O0)  # (N, O0)

    dA0 = qs_flat.T @ obs_flat  # (S0, O0)
    return dA0


def accumulate_B_counts_level(
    qs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for B at a single level.

    Approximation:
        dB[s, s_next] = sum_{t,sites} q(s_t = s) * q(s_{t+1} = s_next)

    Args:
        qs_grid: (T, H, W, S)

    Returns:
        dB: (S, S) Dirichlet count increments for B at this level
    """
    T, H, W, S = qs_grid.shape

    qt = qs_grid[:-1].reshape((T - 1) * H * W, S)   # (N_tr, S)
    qt1 = qs_grid[1:].reshape((T - 1) * H * W, S)   # (N_tr, S)

    dB = qt.T @ qt1  # (S, S)
    return dB


def accumulate_E_counts_level(
    qs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for E (prior over initial states) at a level.

    Args:
        qs_grid: (T, H, W, S)

    Returns:
        dE: (S,) Dirichlet count increments for E
    """
    q0 = qs_grid[0]  # (H, W, S)
    H, W, S = q0.shape
    q0_flat = q0.reshape(H * W, S)
    dE = q0_flat.sum(axis=0)  # (S,)
    return dE


def accumulate_C_counts(
    obs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate counts for lowest-level preferences C0(o) from observations.

    Args:
        obs_grid: (T, H0, W0, O0)

    Returns:
        dC0: (O0,) Dirichlet count increments for C0
    """
    dC0 = obs_grid.sum(axis=(0, 1, 2))  # (O0,)
    return dC0


# -----------------------------------------------------------------------------
# 2. Sufficient statistics for path-dependent transitions and paths (top level)
# -----------------------------------------------------------------------------

def accumulate_B_states_paths_counts_top(
    qs_top_grid: jnp.ndarray,
    qu_top: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for path-dependent state transitions
    B_states_paths[s_next, s, u] at the top level.

    Approximation:
        dB_states_paths[s_next, s, u]
          ≈ sum_t q(u_t = u)
             * sum_sites q(s_t = s) q(s_{t+1} = s_next)

    Args:
        qs_top_grid: (T, H, W, S_top) top-level posteriors
        qu_top: (T, U) path posterior at top level

    Returns:
        dB_states_paths: (S_top, S_top, U) Dirichlet count increments
    """
    T, H, W, S_top = qs_top_grid.shape
    Tq, U = qu_top.shape
    assert Tq == T, "qu_top and qs_top_grid must have same T."

    N_sites = H * W
    qs_flat = qs_top_grid.reshape(T, N_sites, S_top)  # (T, N_sites, S_top)

    dB_states_paths = jnp.zeros((S_top, S_top, U), dtype=jnp.float32)

    for t in range(T - 1):
        qs_t = qs_flat[t]      # (N_sites, S_top)
        qs_tp1 = qs_flat[t+1]  # (N_sites, S_top)

        base_t = qs_t.T @ qs_tp1  # (S_top, S_top)

        qu_t = qu_top[t]  # (U,)
        for u in range(U):
            dB_states_paths = dB_states_paths.at[:, :, u].add(base_t * qu_t[u])

    return dB_states_paths  # (S_top, S_top, U)


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
    qu_tp1 = qu_top[1:]  # (T-1, U)

    dB_paths = qu_t.T @ qu_tp1  # (U, U)
    dE_paths = qu_top[0]        # (U,)

    return dB_paths, dE_paths


# -----------------------------------------------------------------------------
# 3. Update Dirichlet parameters from a single sequence
# -----------------------------------------------------------------------------

def update_dirichlet_from_sequence(
    hierarchy: LorenzHierarchy,
    params: LorenzRGMParams,
    lorenz_data_dict: Dict[str, Any],
    num_iter_lowest: int = 8,
    num_iter_hier: int = 2,
    efe_gamma: float = 16.0,
    pref_mode: str = "data_empirical",
) -> LorenzRGMParams:
    """
    Update Dirichlet parameters (A_alpha, B_alpha, E_alpha, C_alpha,
    and optionally B_states_paths_alpha, B_paths_alpha, E_paths_alpha)
    given a single Lorenz sequence.

    This function:
      1. Builds observations at the lowest level.
      2. Runs hierarchical inference to obtain qs_levels and qu_levels.
      3. Accumulates Dirichlet counts for A/B/E/C across levels.
      4. Accumulates path-related counts at the top level if present.
      5. Adds these counts to the current params.

    Args:
        hierarchy: LorenzHierarchy used for inference
        params: current LorenzRGMParams
        lorenz_data_dict: dataset dict for this sequence
        num_iter_lowest: iterations of lowest-level VMP
        num_iter_hier: iterations of hierarchical updates
        efe_gamma: precision for EFE-based path selection
        pref_mode: mode for preferences ("uniform" or "data_empirical")

    Returns:
        Updated LorenzRGMParams.
    """
    # 1. Build lowest-level observations and preferences
    obs_grid0 = build_lowest_level_observations_grid(lorenz_data_dict)  # (T,H0,W0,O0)
    O0 = obs_grid0.shape[-1]

    if pref_mode == "data_empirical":
        C0 = build_lowest_level_observations_grid(lorenz_data_dict).mean(axis=(0, 1, 2))
        C0 = C0 / (C0.sum() + 1e-8)
    else:
        C0 = jnp.ones((O0,), dtype=jnp.float32)
        C0 = C0 / C0.sum()

    # 2. Run hierarchical inference
    infer_res = infer_lorenz_hierarchy(
        hierarchy,
        lorenz_data_dict,
        C0,
        num_iter_lowest=num_iter_lowest,
        num_iter_hier=num_iter_hier,
        efe_gamma=efe_gamma,
    )
    qs_levels = infer_res["qs_levels"]
    qu_levels = infer_res["qu_levels"]

    # 3. Accumulate A/B/E/C counts

    # Lowest level (level 0)
    qs0_grid = qs_levels[0]
    dA0 = accumulate_A_counts_level0(qs0_grid, obs_grid0)
    dB0 = accumulate_B_counts_level(qs0_grid)
    dE0 = accumulate_E_counts_level(qs0_grid)
    dC0 = accumulate_C_counts(obs_grid0)

    params.A_alpha[0] = params.A_alpha[0] + dA0
    params.B_alpha[0] = params.B_alpha[0] + dB0
    params.E_alpha[0] = params.E_alpha[0] + dE0
    params.C_alpha = params.C_alpha + dC0

    # Higher levels
    for l in range(1, len(hierarchy.levels)):
        qs_l = qs_levels[l]
        if qs_l is None:
            continue

        dB_l = accumulate_B_counts_level(qs_l)
        dE_l = accumulate_E_counts_level(qs_l)

        params.B_alpha[l] = params.B_alpha[l] + dB_l
        params.E_alpha[l] = params.E_alpha[l] + dE_l

    # 4. Path-related counts at top level (if any)
    top_idx = len(hierarchy.levels) - 1
    level_top = hierarchy.levels[top_idx]
    qu_top = qu_levels[top_idx] if top_idx < len(qu_levels) else None

    if level_top.num_paths > 1 and qu_top is not None:
        qs_top_grid = qs_levels[top_idx]

        # B_states_paths_alpha: (S_top, S_top, U)
        if params.B_states_paths_alpha is not None:
            dB_states_paths = accumulate_B_states_paths_counts_top(
                qs_top_grid, qu_top
            )
            params.B_states_paths_alpha = params.B_states_paths_alpha + dB_states_paths

        # B_paths_alpha and E_paths_alpha
        if params.B_paths_alpha is not None and params.E_paths_alpha is not None:
            dB_paths, dE_paths = accumulate_B_E_paths(qu_top)
            params.B_paths_alpha = params.B_paths_alpha + dB_paths
            params.E_paths_alpha = params.E_paths_alpha + dE_paths

    return params


# -----------------------------------------------------------------------------
# 4. Outer training loop (multiple epochs / sequences)
# -----------------------------------------------------------------------------

def train_lorenz_rgm_with_tau(
    hierarchy: LorenzHierarchy,
    params: LorenzRGMParams,
    lorenz_data_dict: Dict[str, Any],
    num_epochs: int = 1,
    num_iter_lowest: int = 8,
    num_iter_hier: int = 2,
    efe_gamma: float = 16.0,
    pref_mode: str = "data_empirical",
) -> LorenzRGMParams:
    """
    Simple training loop over a single Lorenz trajectory (or dataset),
    alternating between inference and parameter updates.

    Args:
        hierarchy: initial LorenzHierarchy
        params: initial Dirichlet parameters
        lorenz_data_dict: data dict for training
        num_epochs: number of passes over the data
        num_iter_lowest: iterations for lowest-level VMP
        num_iter_hier: iterations for hierarchical updates
        efe_gamma: precision for EFE-based path selection
        pref_mode: preference mode ("uniform" or "data_empirical")

    Returns:
        Updated LorenzRGMParams.
    """
    for epoch in range(num_epochs):
        print(f"train_lorenz_rgm_with_tau(): Epoch {epoch+1}/{num_epochs}")

        params = update_dirichlet_from_sequence(
            hierarchy,
            params,
            lorenz_data_dict,
            num_iter_lowest=num_iter_lowest,
            num_iter_hier=num_iter_hier,
            efe_gamma=efe_gamma,
            pref_mode=pref_mode,
        )

        # Rebuild hierarchy from updated params to use learned A/B/E/path
        hierarchy = build_lorenz_hierarchy_from_params(
            lorenz_spatial_hierarchy={
                "levels": hierarchy.states_grids,  # states_grids reused
                "T": hierarchy.T,
                "H_blocks": hierarchy.H_blocks,
                "W_blocks": hierarchy.W_blocks,
            },
            params=params,
            num_paths_top=hierarchy.levels[-1].num_paths,
        )

    return params
