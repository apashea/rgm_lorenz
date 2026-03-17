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

Conventions (aligned with lorenz_model.py, lorenz_inference.py, lorenz_efe.py):

- For the top level with S1 states and U paths:

  B_states_paths[s_next, s, u] = P(s_next | s, u)

  with shape (S1, S1, U).

- The corresponding Dirichlet parameters B_states_paths_alpha have the same
  shape (S1, S1, U), and are normalized over the s_next axis (axis=0)
  for each (s, u) when converting to categorical probabilities.
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
      qs0_grid: (T, H0, W0, S0)
      obs_grid: (T, H0, W0, O0)

    Returns:
      dA0: (S0, O0)
    """
    T, H0, W0, S0 = qs0_grid.shape
    _, _, _, O0 = obs_grid.shape

    qs_flat = qs0_grid.reshape(T * H0 * W0, S0)
    obs_flat = obs_grid.reshape(T * H0 * W0, O0)

    dA0 = qs_flat.T @ obs_flat
    return dA0


def accumulate_B_counts_level0(
    qs0_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for B0 (lowest-level transitions).

    Approximation:
      dB[s, s'] = sum_{t,h,w} q(s_t = s) * q(s_{t+1} = s').

    Args:
      qs0_grid: (T, H0, W0, S0)

    Returns:
      dB0: (S0, S0)
    """
    T, H0, W0, S0 = qs0_grid.shape

    qt = qs0_grid[:-1].reshape((T - 1) * H0 * W0, S0)
    qt1 = qs0_grid[1:].reshape((T - 1) * H0 * W0, S0)

    dB0 = qt.T @ qt1
    return dB0


def accumulate_E_counts_level0(
    qs0_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for E0 (prior over initial states).

    Args:
      qs0_grid: (T, H0, W0, S0)

    Returns:
      dE0: (S0,)
    """
    q0 = qs0_grid[0]  # (H0, W0, S0)
    H0, W0, S0 = q0.shape
    q0_flat = q0.reshape(H0 * W0, S0)
    dE0 = q0_flat.sum(axis=0)
    return dE0


def accumulate_C_counts(
    obs_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate counts for lowest-level preferences C0(o) from observations.

    Args:
      obs_grid: (T, H0, W0, O0)

    Returns:
      dC0: (O0,)
    """
    dC0 = obs_grid.sum(axis=(0, 1, 2))
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
      dB1: (S1, S1)
    """
    T, H1, W1, S1 = qs1_grid.shape

    qt = qs1_grid[:-1].reshape((T - 1) * H1 * W1, S1)
    qt1 = qs1_grid[1:].reshape((T - 1) * H1 * W1, S1)

    dB1 = qt.T @ qt1
    return dB1


def accumulate_E_counts_level1(
    qs1_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for E1 (prior over initial states at level 1).

    Args:
      qs1_grid: (T, H1, W1, S1)

    Returns:
      dE1: (S1,)
    """
    q0 = qs1_grid[0]  # (H1, W1, S1)
    H1, W1, S1 = q0.shape
    q0_flat = q0.reshape(H1 * W1, S1)
    dE1 = q0_flat.sum(axis=0)
    return dE1


# -----------------------------------------------------------------------------
# 3. Sufficient statistics for path-dependent transitions and paths
# -----------------------------------------------------------------------------

def accumulate_B_states_paths_counts_level1(
    qs1_grid: jnp.ndarray,
    qu_top: jnp.ndarray,
) -> jnp.ndarray:
    """
    Accumulate expected counts for path-dependent state transitions
    B_states_paths[s_next, s, u] at the top level.

    Approximation:
      dB_states_paths[s, s', u]
        ≈ sum_t q(u_t = u) * sum_{h,w} q(s_t = s) q(s_{t+1} = s').

    Args:
      qs1_grid: (T, H1, W1, S1)
      qu_top: (T, U)

    Returns:
      dB_states_paths: (S1, S1, U)
    """
    T, H1, W1, S1 = qs1_grid.shape
    Tq, U = qu_top.shape
    assert Tq == T, "qu_top and qs1_grid must have same T."

    N_sites = H1 * W1
    qs1_flat = qs1_grid.reshape(T, N_sites, S1)  # (T, N_sites, S1)

    dB_states_paths = jnp.zeros((S1, S1, U), dtype=jnp.float32)

    # Loop in Python over time and paths; JAX arrays are updated with .at[]
    for t in range(T - 1):
        qs_t = qs1_flat[t]      # (N_sites, S1)
        qs_tp1 = qs1_flat[t + 1]  # (N_sites, S1)

        # base_t[s, s'] = sum_{h,w} q(s_t = s) q(s_{t+1} = s')
        base_t = qs_t.T @ qs_tp1  # (S1, S1)

        qu_t = qu_top[t]  # (U,)
        for u in range(U):
            dB_states_paths = dB_states_paths.at[:, :, u].add(base_t * qu_t[u])

    return dB_states_paths  # (S1, S1, U)


def accumulate_B_E_paths(
    qu_top: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Accumulate Dirichlet counts for path transitions B_paths and E_paths.

    Args:
      qu_top: (T, U)

    Returns:
      dB_paths: (U, U)
      dE_paths: (U,)
    """
    T, U = qu_top.shape
    qu_t = qu_top[:-1]
    qu_t1 = qu_top[1:]

    dB_paths = qu_t.T @ qu_t1
    dE_paths = qu_top[0]
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
    obs_grid = build_lowest_level_observations_grid(lorenz_data_dict)

    # 2. Inference (positional args to match infer_lorenz_hierarchy signature)
    results = infer_lorenz_hierarchy(
        hierarchy,
        lorenz_data_dict,
        num_iter_lowest,
        num_iter_hier,
        efe_gamma,
        pref_mode,
    )

    qs_levels = results["qs_levels"]
    qu_levels = results["qu_levels"]

    qs0_grid = qs_levels[0]
    qs1_grid = qs_levels[1] if len(qs_levels) > 1 else None
    qu_top = qu_levels[-1] if len(qu_levels) > 1 else None

    # 3. Level 0 counts
    dA0 = accumulate_A_counts_level0(qs0_grid, obs_grid)
    dB0 = accumulate_B_counts_level0(qs0_grid)
    dE0 = accumulate_E_counts_level0(qs0_grid)
    dC0 = accumulate_C_counts(obs_grid)

    params.A_alpha[0] = params.A_alpha[0] + dA0
    params.B_alpha[0] = params.B_alpha[0] + dB0
    params.E_alpha[0] = params.E_alpha[0] + dE0
    params.C_alpha = params.C_alpha + dC0

    # 4. Level 1 counts (if present)
    if qs1_grid is not None and len(params.B_alpha) > 1:
        dB1 = accumulate_B_counts_level1(qs1_grid)
        dE1 = accumulate_E_counts_level1(qs1_grid)
        params.B_alpha[1] = params.B_alpha[1] + dB1
        params.E_alpha[1] = params.E_alpha[1] + dE1

    # 5. Path counts at top level
    if (
        qu_top is not None
        and params.B_paths_alpha is not None
        and params.E_paths_alpha is not None
    ):
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
# 5. Normalize Dirichlet parameters to categorical A/B/E/C and paths
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
    for inference: A, B, E, C, and (if present) B_states_paths, B_paths, E_paths.

    Args:
      params: LorenzRGMParams
      K, L: lowest-level configuration (O0 = K * L)

    Returns:
      A_list, B_list, E_list, C0, B_states_paths, B_paths, E_paths
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

    if params.B_paths_alpha is not None and params.E_paths_alpha is not None:
        B_paths = params.B_paths_alpha / (
            params.B_paths_alpha.sum(axis=1, keepdims=True) + 1e-8
        )
        E_paths = params.E_paths_alpha / (params.E_paths_alpha.sum() + 1e-8)
    else:
        B_paths, E_paths = None, None

    if params.B_states_paths_alpha is not None:
        # params.B_states_paths_alpha is (S1, S1, U);
        # normalize over s_next axis (axis=0) for each (s, u)
        B_states_paths = params.B_states_paths_alpha / (
            params.B_states_paths_alpha.sum(axis=0, keepdims=True) + 1e-8
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

    For each sequence:
      1. Build Lorenz data via build_data_fn.
      2. Build a hierarchy from current params and spatial hierarchy.
      3. Run inference and update Dirichlet parameters.

    Returns:
      Trained LorenzRGMParams
    """
    params = initial_params

    for epoch in range(num_epochs):
        for seq_idx in range(num_sequences_per_epoch):
            lorenz_data_dict = build_data_fn()

            hierarchy = build_lorenz_hierarchy_from_params(
                lorenz_spatial_hierarchy,
                params,
                num_paths_top=num_paths_top,
            )

            params = update_dirichlet_from_sequence(
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
# 7. Convenience wrapper: single-hierarchy training (used by notebook)
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
        params = update_dirichlet_from_sequence(
            hierarchy,
            params,
            lorenz_data_dict,
            num_iter_lowest,
            num_iter_hier,
            efe_gamma,
            pref_mode,
        )

    return params
