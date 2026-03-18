# lorenz_learning.py
"""
Learning routines for the Lorenz RGM.

This module implements:
- Accumulation of Dirichlet counts for A, E_states, and preferences from
  posterior beliefs over states and observations.
- Updates of Dirichlet concentration parameters stored in LorenzRGMParams:
  * A_alpha[l], E_states_alpha[l], pref_alpha
  * D_state_from_parent_alpha[l] (if desired),
  * (optionally) B_states_paths_alpha[l], C_paths_alpha[l], E_paths_alpha[l]
- A training loop scaffold that alternates between inference and parameter
  updates over one or more Lorenz trajectories.

Conventions (aligned with lorenz_model.py, lorenz_inference.py, lorenz_efe.py):

- For a level with S states and U paths:

  B_states_paths[s_next, s, u] = P(s_next | s, u)

  with shape (S, S, U).

- The corresponding Dirichlet parameters B_states_paths_alpha[l] have the same
  shape and are normalized over the s_next axis (axis=0) for each (s, u)
  when converting to categorical probabilities.
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
# 1. Sufficient statistics for lowest level (A0, E_states^0, preferences)
# -----------------------------------------------------------------------------

def accumulate_A_counts_level0(
    qs0_grid: jnp.ndarray,
    obs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for A0 (lowest-level emission matrix).

    Args:
      qs0_grid: (T0, H0, W0, S0)
      obs_grid: (T0, H0, W0, O0)

    Returns:
      dA0: (S0, O0)
    """
    T0, H0, W0, S0 = qs0_grid.shape
    _, _, _, O0 = obs_grid.shape

    qs_flat = qs0_grid.reshape(T0 * H0 * W0, S0)
    obs_flat = obs_grid.reshape(T0 * H0 * W0, O0)

    dA0 = qs_flat.T @ obs_flat
    return dA0


def accumulate_E_states_counts_level0(
    qs0_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for E_states^0 (prior over initial states).

    Args:
      qs0_grid: (T0, H0, W0, S0)

    Returns:
      dE0: (S0,)
    """
    q0 = qs0_grid[0]  # (H0, W0, S0)
    H0, W0, S0 = q0.shape
    q0_flat = q0.reshape(H0 * W0, S0)
    dE0 = q0_flat.sum(axis=0)
    return dE0


def accumulate_pref_counts(
    obs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate counts for lowest-level preferences C(o) from observations.

    Args:
      obs_grid: (T0, H0, W0, O0)

    Returns:
      dC0: (O0,)
    """
    dC0 = obs_grid.sum(axis=(0, 1, 2))
    return dC0


# -----------------------------------------------------------------------------
# 2. Sufficient statistics for higher-level E_states
# -----------------------------------------------------------------------------

def accumulate_E_states_counts_level(
    qs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for E_states at a higher level.

    Args:
      qs_grid: (T, H, W, S)

    Returns:
      dE: (S,)
    """
    q0 = qs_grid[0]  # (H, W, S)
    H, W, S = q0.shape
    q0_flat = q0.reshape(H * W, S)
    dE = q0_flat.sum(axis=0)
    return dE


# -----------------------------------------------------------------------------
# 3. Sufficient statistics for path-dependent transitions and paths
# -----------------------------------------------------------------------------

def accumulate_B_states_paths_counts_level(
    qs_grid: jnp.ndarray,
    qu: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for path-dependent state transitions
    B_states_paths[s_next, s, u] at a given level.

    Approximation:
      dB_states_paths[s, s', u]
      ≈ sum_t q(u_t = u) * sum_{h,w} q(s_t = s) q(s_{t+1} = s').

    Args:
      qs_grid: (T, H, W, S)
      qu: (T, U)

    Returns:
      dB_states_paths: (S, S, U)
    """
    T, H, W, S = qs_grid.shape
    Tq, U = qu.shape
    assert Tq == T, "qu and qs_grid must have same T."

    N_sites = H * W
    qs_flat = qs_grid.reshape(T, N_sites, S)  # (T, N_sites, S)

    dB_states_paths = jnp.zeros((S, S, U), dtype=jnp.float32)

    for t in range(T - 1):
        qs_t = qs_flat[t]      # (N_sites, S)
        qs_tp1 = qs_flat[t + 1]  # (N_sites, S)

        base_t = qs_t.T @ qs_tp1  # (S, S)

        qu_t = qu[t]  # (U,)
        for u in range(U):
            dB_states_paths = dB_states_paths.at[:, :, u].add(base_t * qu_t[u])

    return dB_states_paths  # (S, S, U)


def accumulate_C_E_paths(
    qu: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Accumulate Dirichlet counts for path transitions C_paths and E_paths.

    Args:
      qu: (T, U)

    Returns:
      dC_paths: (U, U)
      dE_paths: (U,)
    """
    T, U = qu.shape
    qu_t = qu[:-1]
    qu_t1 = qu[1:]

    dC_paths = qu_t.T @ qu_t1
    dE_paths = qu[0]
    return dC_paths, dE_paths


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
    concentration parameters in LorenzRGMParams.

    Args:
      hierarchy: current LorenzHierarchy
      params: current LorenzRGMParams
      lorenz_data_dict: data dict for this sequence
      num_iter_lowest: iterations for lowest-level VMP
      num_iter_hier: iterations for hierarchical VMP
      efe_gamma: precision over EFE
      pref_mode: preference mode

    Returns:
      Updated LorenzRGMParams
    """
    # 1. Observations as grid
    obs_grid = build_lowest_level_observations_grid(lorenz_data_dict)  # (T0, H0, W0, O0)

    # 2. Inference (preferences from pref_alpha via params)
    results = infer_lorenz_hierarchy(
        hierarchy,
        lorenz_data_dict,
        params=params,
        num_iter_lowest=num_iter_lowest,
        num_iter_hier=num_iter_hier,
        efe_gamma=efe_gamma,
        pref_mode=pref_mode,
    )

    qs_levels = results["qs_levels"]
    qu_levels = results["qu_levels"]

    qs0_grid = qs_levels[0]
    qs1_grid = qs_levels[1] if len(qs_levels) > 1 else None
    qs2_grid = qs_levels[2] if len(qs_levels) > 2 else None
    qu_top = qu_levels[-1] if len(qu_levels) > 1 else None

    # 3. Level 0 counts
    dA0 = accumulate_A_counts_level0(qs0_grid, obs_grid)
    dE0 = accumulate_E_states_counts_level0(qs0_grid)
    dC0 = accumulate_pref_counts(obs_grid)

    params.A_alpha[0] = params.A_alpha[0] + dA0
    params.E_states_alpha[0] = params.E_states_alpha[0] + dE0
    params.pref_alpha = params.pref_alpha + dC0

    # 4. Higher-level state priors (E_states) if present
    if qs1_grid is not None and len(params.E_states_alpha) > 1:
        dE1 = accumulate_E_states_counts_level(qs1_grid)
        params.E_states_alpha[1] = params.E_states_alpha[1] + dE1

    if qs2_grid is not None and len(params.E_states_alpha) > 2:
        dE2 = accumulate_E_states_counts_level(qs2_grid)
        params.E_states_alpha[2] = params.E_states_alpha[2] + dE2

    # 5. Path counts at the highest path-carrying level (top_idx)
    if qu_top is not None:
        # Find highest level with path factor, consistent with inference
        top_idx = None
        for idx in reversed(range(len(params.C_paths_alpha))):
            if (
                params.C_paths_alpha[idx] is not None
                and params.E_paths_alpha[idx] is not None
            ):
                top_idx = idx
                break

        if top_idx is not None:
            dC_paths, dE_paths = accumulate_C_E_paths(qu_top)
            params.C_paths_alpha[top_idx] = params.C_paths_alpha[top_idx] + dC_paths
            params.E_paths_alpha[top_idx] = params.E_paths_alpha[top_idx] + dE_paths

            # 6. Path-dependent state transitions at that same level
            if params.B_states_paths_alpha[top_idx] is not None:
                qs_top_grid = None
                if top_idx == 2 and qs2_grid is not None:
                    qs_top_grid = qs2_grid
                elif top_idx == 1 and qs1_grid is not None:
                    qs_top_grid = qs1_grid
                elif top_idx == 0:
                    qs_top_grid = qs0_grid

                if qs_top_grid is not None:
                    dB_states_paths = accumulate_B_states_paths_counts_level(
                        qs_top_grid, qu_top
                    )
                    params.B_states_paths_alpha[top_idx] = (
                        params.B_states_paths_alpha[top_idx] + dB_states_paths
                    )

    return params


# JIT-compiled version for faster sequence-level updates
update_dirichlet_from_sequence_jit = jax.jit(
    update_dirichlet_from_sequence,
    static_argnames=("num_iter_lowest", "num_iter_hier", "efe_gamma", "pref_mode"),
)


# -----------------------------------------------------------------------------
# 5. Training loop scaffold
# -----------------------------------------------------------------------------

def train_lorenz_rgm(
    initial_params: LorenzRGMParams,
    lorenz_spatial_hierarchy: Dict[str, Any],
    build_data_fn,
    K: int,
    L: int,
    T0: int,
    K0: int,
    K1: int,
    num_epochs: int = 1,
    num_sequences_per_epoch: int = 1,
    num_iter_lowest: int = 8,
    num_iter_hier: int = 4,
    efe_gamma: float = 16.0,
    pref_mode: str = "data_empirical",
) -> LorenzRGMParams:
    """
    High-level training loop scaffold for the Lorenz RGM.

    For each sequence:
      1. Build Lorenz data via build_data_fn.
      2. Build a hierarchy from current params and spatial hierarchy.
      3. Run inference and update Dirichlet parameters.

    Args:
      initial_params: initial Dirichlet parameters
      lorenz_spatial_hierarchy: spatial hierarchy (states_grids, D tensors)
      build_data_fn: callable returning a lorenz_data_dict
      K, L: lowest-level configuration
      T0, K0, K1: temporal structure
      num_epochs: number of epochs
      num_sequences_per_epoch: sequences per epoch
      num_iter_lowest: iterations at lowest level
      num_iter_hier: iterations at higher levels
      efe_gamma: precision over EFE
      pref_mode: preference mode

    Returns:
      Trained LorenzRGMParams
    """
    params = initial_params

    _ = K  # unused here, kept for API symmetry
    _ = L

    for epoch in range(num_epochs):
        for _ in range(num_sequences_per_epoch):
            lorenz_data_dict = build_data_fn()

            hierarchy = build_lorenz_hierarchy_from_params(
                lorenz_spatial_hierarchy,
                params,
                T0=T0,
                K0=K0,
                K1=K1,
            )

            params = update_dirichlet_from_sequence_jit(
                hierarchy,
                params,
                lorenz_data_dict,
                num_iter_lowest,
                num_iter_hier,
                efe_gamma,
                pref_mode,
            )

    return params


# -----------------------------------------------------------------------------
# 6. Convenience wrapper: single-hierarchy training (used by notebook)
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
    Minimal training loop used by the notebook:
    reuse a single hierarchy and dataset, iterating Dirichlet updates.

    This is a thin wrapper around update_dirichlet_from_sequence.
    """
    for epoch in range(num_epochs):
        print(f"train_lorenz_rgm_with_tau(): Epoch {epoch+1}/{num_epochs}")
        params = update_dirichlet_from_sequence_jit(
            hierarchy,
            params,
            lorenz_data_dict,
            num_iter_lowest,
            num_iter_hier,
            efe_gamma,
            pref_mode,
        )

    return params
