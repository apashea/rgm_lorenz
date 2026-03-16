# lorenz_model.py
"""
Model construction for the Lorenz RGM.

This module defines:
- LorenzLevel: a single spatial / temporal level in the RGM.
- LorenzHierarchy: a stack of LorenzLevels with associated state grids.
- Functions to build an initial Lorenz hierarchy given a spatial
  RG hierarchy (states_grids, D mappings).
- Functions to initialize and convert Dirichlet parameters for A, B, E, C,
  and path-related quantities.

IMPORTANT CONVENTIONS (new, unified B layout):

- For each level ℓ with S_l hidden states:

    B_states_full_l: (S_l_next, S_l_current, U_l)

  where U_l is the number of "control/path" values at that level.
  By default, U_l = 1 for ordinary levels (no explicit paths) and
  U_top = num_paths_top for the top level where a path factor exists.

  Semantics:
      B_states_full_l[s_next, s_current, u] = P(s_next | s_current, u_l = u)

  For levels without multiple paths, we fix U_l = 1 and always use u=0,
  so the effective transition matrix is:

      B_states_l = B_states_full_l[:, :, 0]  # (S_l, S_l)

- At the top level L-1, we also define:

    B_paths: (U, U)       with B_paths[u_next, u] = P(u_next | u)
    E_paths: (U,)         with E_paths[u] = P(u_0 = u)

- For compatibility with existing code:

    LorenzLevel.B_states   will hold the effective (S, S) matrix for that level,
                           i.e., B_states_full[:, :, 0] for levels with U_l = 1,
                           or a default mixture over paths for the top level if
                           needed for certain operations.

    LorenzLevel.B_states_paths will now store the full (S, S, U) tensor at the
                               top level only, and None at other levels.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class LorenzLevel:
    """
    A single level in the Lorenz hierarchy.

    Attributes:
        S: number of hidden states at this level
        A: emission matrix (S, O) or empty if no direct observations
        B_states: effective transition matrix (S, S) used in generic code
        E_states: prior over initial states (S,)
        D: deterministic RG mapping from parent to child states, or None
        num_paths: number of path states at this level (U_l)
        B_paths: path transition matrix (U_l, U_l) or None
        E_paths: prior over initial path state (U_l,) or None
        B_states_paths: full path-dependent transitions (S, S, U_l) or None
    """
    S: int
    A: jnp.ndarray
    B_states: jnp.ndarray
    E_states: jnp.ndarray
    D: Optional[jnp.ndarray]

    num_paths: int
    B_paths: Optional[jnp.ndarray]
    E_paths: Optional[jnp.ndarray]
    B_states_paths: Optional[jnp.ndarray]


@dataclass
class LorenzHierarchy:
    """
    A hierarchical Lorenz model with multiple spatial levels.

    Attributes:
        levels: list of LorenzLevel objects, from finest (level 0) to coarsest (top)
        states_grids: list of state grids per level (T, H_l, W_l)
        T: number of time steps
        H_blocks: number of lowest-level patches in vertical dimension
        W_blocks: number of lowest-level patches in horizontal dimension
    """
    levels: List[LorenzLevel]
    states_grids: List[jnp.ndarray]
    T: int
    H_blocks: int
    W_blocks: int


@dataclass
class LorenzRGMParams:
    """
    Dirichlet concentration parameters for the Lorenz RGM.

    Attributes:
        A_alpha: list of A^l alphas, each (S_l, O_l)
        B_alpha: list of B^l alphas, each (S_l, S_l)
        E_alpha: list of E^l alphas, each (S_l,)
        C_alpha: preferences over lowest-level outcomes (O0,)
        B_states_paths_alpha: (S_top, S_top, U_top) or None
        B_paths_alpha: (U_top, U_top) or None
        E_paths_alpha: (U_top,) or None
    """
    A_alpha: List[jnp.ndarray]
    B_alpha: List[jnp.ndarray]
    E_alpha: List[jnp.ndarray]
    C_alpha: jnp.ndarray

    B_states_paths_alpha: Optional[jnp.ndarray]
    B_paths_alpha: Optional[jnp.ndarray]
    E_paths_alpha: Optional[jnp.ndarray]


# -----------------------------------------------------------------------------
# Utilities for building B matrices
# -----------------------------------------------------------------------------

def build_uniform_B(S: int, self_bias: float = 1.0) -> jnp.ndarray:
    """
    Build a simple transition matrix B(s'|s) with a self-transition bias.

    Returns:
        B: (S, S) with B[s_next, s] = P(s_next | s)
    """
    B = jnp.ones((S, S), dtype=jnp.float32)
    B = B + self_bias * jnp.eye(S, dtype=jnp.float32)
    B = B / (B.sum(axis=0, keepdims=True) + 1e-8)
    return B


def build_path_dependent_B_states(
    S: int,
    num_paths: int,
    self_bias: float = 1.0,
) -> jnp.ndarray:
    """
    Build a path-dependent transition tensor for a level with S states and
    num_paths path states.

    Returns:
        B_states_paths: (S, S, U) where
            B_states_paths[s_next, s, u] = P(s_next | s, u)
    """
    B_base = build_uniform_B(S, self_bias=self_bias)  # (S_next, S_current)
    B_states_paths = jnp.stack(
        [B_base for _ in range(num_paths)],
        axis=2,
    )  # (S, S, U)
    return B_states_paths


def build_path_dynamics(num_paths: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build simple path transition dynamics B_paths and initial prior E_paths.

    Returns:
        B_paths: (U, U) with B_paths[u_next, u] = P(u_next | u)
        E_paths: (U,) initial prior over u_0
    """
    U = num_paths
    B_paths = jnp.ones((U, U), dtype=jnp.float32)
    B_paths = B_paths + 2.0 * jnp.eye(U, dtype=jnp.float32)
    B_paths = B_paths / (B_paths.sum(axis=0, keepdims=True) + 1e-8)

    E_paths = jnp.ones((U,), dtype=jnp.float32)
    E_paths = E_paths / (E_paths.sum() + 1e-8)

    return B_paths, E_paths


# -----------------------------------------------------------------------------
# Build hierarchy from spatial RG structure (2 levels for now)
# -----------------------------------------------------------------------------

def build_lorenz_hierarchy(
    T: int,
    dt: float,
    img_size: int,
    patch_size: int,
    K: int,
    L: int,
    thickness: int,
    num_spatial_levels: int,
    num_paths_top: int,
    lorenz_spatial_hierarchy: Dict[str, Any],
) -> LorenzHierarchy:
    """
    Build an initial LorenzHierarchy with fixed A/B/E/D/path parameters,
    given a spatial RG hierarchy (states_grids, D matrices) from
    lorenz_renorm.build_lorenz_spatial_hierarchy.

    Currently supports:
        - level 0: patch level
        - level 1: one parent spatial level
    but is designed to be extended to more levels.

    Args:
        T, dt, img_size, patch_size, K, L, thickness: config parameters
        num_spatial_levels: number of RG steps used to build spatial_h
        num_paths_top: number of path states at top level
        lorenz_spatial_hierarchy: dict with keys:
            - "levels": list of dicts with "states_grid" and "D" (for l>0)
            - "H_blocks", "W_blocks"

    Returns:
        LorenzHierarchy
    """
    levels: List[LorenzLevel] = []
    states_grids: List[jnp.ndarray] = []

    # ---------------------
    # Level 0 (patch level)
    # ---------------------
    level0_states_grid = lorenz_spatial_hierarchy["levels"][0]["states_grid"]
    states_grids.append(level0_states_grid)

    H0 = int(lorenz_spatial_hierarchy["H_blocks"])
    W0 = int(lorenz_spatial_hierarchy["W_blocks"])
    T_h = int(lorenz_spatial_hierarchy.get("T", level0_states_grid.shape[0]))

    S0 = int(jnp.max(level0_states_grid) + 1)

    # Lowest-level emission A0: (S0, O0) with O0 = K * L
    O0 = K * L
    A0 = jnp.ones((S0, O0), dtype=jnp.float32)
    A0 = A0 / (A0.sum(axis=1, keepdims=True) + 1e-8)

    # Transition B0: (S0, S0, U0=1)
    B0_base = build_uniform_B(S0, self_bias=1.0)  # (S0, S0)
    B0_full = B0_base[:, :, None]                 # (S0, S0, 1)

    # Prior E0: (S0,)
    E0 = jnp.ones((S0,), dtype=jnp.float32)
    E0 = E0 / (E0.sum() + 1e-8)

    level0 = LorenzLevel(
        S=S0,
        A=A0,
        B_states=B0_base,          # effective (S0, S0)
        E_states=E0,
        D=None,
        num_paths=1,               # trivial path dimension
        B_paths=None,
        E_paths=None,
        B_states_paths=B0_full,    # (S0, S0, 1)
    )
    levels.append(level0)

    # ---------------------
    # Level 1 (first parent)
    # ---------------------
    if num_spatial_levels >= 1 and len(lorenz_spatial_hierarchy["levels"]) > 1:
        spatial_level1 = lorenz_spatial_hierarchy["levels"][1]
        states_grid1 = spatial_level1["states_grid"]
        D1 = spatial_level1["D"]  # (S1, 4)
        states_grids.append(states_grid1)

        S1 = int(jnp.max(states_grid1) + 1)

        # For now, no direct observations at level 1: A1 is empty
        A1 = jnp.zeros((S1, 0), dtype=jnp.float32)

        # Level-1 transitions, again with a trivial path dimension U1=1
        B1_base = build_uniform_B(S1, self_bias=1.0)  # (S1, S1)
        B1_full = B1_base[:, :, None]                 # (S1, S1, 1)

        E1 = jnp.ones((S1,), dtype=jnp.float32)
        E1 = E1 / (E1.sum() + 1e-8)

        # Path factor will live at the top level (not here) in the 3-level design.
        # For now (2-level hierarchy), we can either attach it here or later.
        # In this version, we DO NOT attach paths to level 1; they will be added
        # at a higher level once the 3-level hierarchy is implemented.
        level1 = LorenzLevel(
            S=S1,
            A=A1,
            B_states=B1_base,
            E_states=E1,
            D=D1,
            num_paths=1,
            B_paths=None,
            E_paths=None,
            B_states_paths=B1_full,
        )
        levels.append(level1)

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T=T_h,
        H_blocks=H0,
        W_blocks=W0,
    )

    return hierarchy


# -----------------------------------------------------------------------------
# Dirichlet parameter initialization and conversion
# -----------------------------------------------------------------------------

def init_dirichlet_params_from_hierarchy(
    hierarchy: LorenzHierarchy,
    K: int,
    L: int,
    alpha_A: float = 1.0,
    alpha_B: float = 1.0,
    alpha_E: float = 1.0,
    alpha_C: float = 1.0,
    num_paths_top: int = 0,
) -> LorenzRGMParams:
    """
    Initialize Dirichlet concentration parameters from a given hierarchy.

    Args:
        hierarchy: LorenzHierarchy with levels already built
        K, L: for lowest level, O0 = K*L
        alpha_A, alpha_B, alpha_E, alpha_C: scalar initializations
        num_paths_top: number of path states at the top level

    Returns:
        LorenzRGMParams with A_alpha, B_alpha, E_alpha, C_alpha,
        and (if num_paths_top>0) B_states_paths_alpha, B_paths_alpha, E_paths_alpha.
    """
    A_alpha_list: List[jnp.ndarray] = []
    B_alpha_list: List[jnp.ndarray] = []
    E_alpha_list: List[jnp.ndarray] = []

    for level in hierarchy.levels:
        S = level.S
        A = level.A
        B = level.B_states
        E = level.E_states

        if A.size > 0:
            A_alpha = jnp.full_like(A, alpha_A, dtype=jnp.float32)
        else:
            A_alpha = A

        B_alpha = jnp.full_like(B, alpha_B, dtype=jnp.float32)
        E_alpha = jnp.full_like(E, alpha_E, dtype=jnp.float32)

        A_alpha_list.append(A_alpha)
        B_alpha_list.append(B_alpha)
        E_alpha_list.append(E_alpha)

    O0 = K * L
    C_alpha = jnp.full((O0,), alpha_C, dtype=jnp.float32)

    # Path-related Dirichlet parameters at the top level
    if num_paths_top > 0:
        top_level = hierarchy.levels[-1]
        S_top = top_level.S
        U = num_paths_top

        B_states_paths_alpha = jnp.full(
            (S_top, S_top, U), alpha_B, dtype=jnp.float32
        )
        B_paths_alpha = jnp.full((U, U), alpha_B, dtype=jnp.float32)
        E_paths_alpha = jnp.full((U,), alpha_E, dtype=jnp.float32)
    else:
        B_states_paths_alpha = None
        B_paths_alpha = None
        E_paths_alpha = None

    params = LorenzRGMParams(
        A_alpha=A_alpha_list,
        B_alpha=B_alpha_list,
        E_alpha=E_alpha_list,
        C_alpha=C_alpha,
        B_states_paths_alpha=B_states_paths_alpha,
        B_paths_alpha=B_paths_alpha,
        E_paths_alpha=E_paths_alpha,
    )

    return params


def build_lorenz_hierarchy_from_params(
    lorenz_spatial_hierarchy: Dict[str, Any],
    params: LorenzRGMParams,
    num_paths_top: int,
) -> LorenzHierarchy:
    """
    Build a LorenzHierarchy from spatial structure and Dirichlet parameters.

    This converts Dirichlet parameters into categorical A, B, E, C and (if
    available) path-related B_states_paths, B_paths, E_paths, then constructs
    LorenzLevel objects consistent with the spatial RG hierarchy.

    Args:
        lorenz_spatial_hierarchy: output of build_lorenz_spatial_hierarchy
        params: LorenzRGMParams with current alphas
        num_paths_top: number of path states at top level

    Returns:
        LorenzHierarchy
    """
    levels: List[LorenzLevel] = []
    states_grids: List[jnp.ndarray] = []

    T = int(lorenz_spatial_hierarchy.get("T"))
    H0 = int(lorenz_spatial_hierarchy["H_blocks"])
    W0 = int(lorenz_spatial_hierarchy["W_blocks"])

    A_alpha_list = params.A_alpha
    B_alpha_list = params.B_alpha
    E_alpha_list = params.E_alpha

    # Normalize path-related alphas if present
    if params.B_states_paths_alpha is not None and num_paths_top > 0:
        B_states_paths = params.B_states_paths_alpha / (
            params.B_states_paths_alpha.sum(axis=0, keepdims=True) + 1e-8
        )  # (S_top, S_top, U)
    else:
        B_states_paths = None

    if params.B_paths_alpha is not None and params.E_paths_alpha is not None and num_paths_top > 0:
        B_paths = params.B_paths_alpha / (
            params.B_paths_alpha.sum(axis=0, keepdims=True) + 1e-8
        )  # (U, U) column-normalized
        E_paths = params.E_paths_alpha / (params.E_paths_alpha.sum() + 1e-8)
    else:
        B_paths, E_paths = None, None

    # ----- Level 0 -----
    level0_states_grid = lorenz_spatial_hierarchy["levels"][0]["states_grid"]
    states_grids.append(level0_states_grid)

    A0_alpha = A_alpha_list[0]
    B0_alpha = B_alpha_list[0]
    E0_alpha = E_alpha_list[0]

    if A0_alpha.size > 0:
        A0 = A0_alpha / (A0_alpha.sum(axis=1, keepdims=True) + 1e-8)
    else:
        A0 = A0_alpha

    B0 = B0_alpha / (B0_alpha.sum(axis=0, keepdims=True) + 1e-8)  # (S0, S0)
    E0 = E0_alpha / (E0_alpha.sum() + 1e-8)

    B0_full = B0[:, :, None]  # (S0, S0, 1)

    level0 = LorenzLevel(
        S=B0.shape[0],
        A=A0,
        B_states=B0,
        E_states=E0,
        D=None,
        num_paths=1,
        B_paths=None,
        E_paths=None,
        B_states_paths=B0_full,
    )
    levels.append(level0)

    # ----- Higher levels (currently just level 1, but extendable) -----
    if len(lorenz_spatial_hierarchy["levels"]) > 1 and len(A_alpha_list) > 1:
        # Level 1
        spatial_level1 = lorenz_spatial_hierarchy["levels"][1]
        states_grid1 = spatial_level1["states_grid"]
        D1 = spatial_level1["D"]  # (S1, 4)
        states_grids.append(states_grid1)

        A1_alpha = A_alpha_list[1]
        B1_alpha = B_alpha_list[1]
        E1_alpha = E_alpha_list[1]

        if A1_alpha.size > 0:
            A1 = A1_alpha / (A1_alpha.sum(axis=1, keepdims=True) + 1e-8)
        else:
            A1 = A1_alpha

        B1 = B1_alpha / (B1_alpha.sum(axis=0, keepdims=True) + 1e-8)
        E1 = E1_alpha / (E1_alpha.sum() + 1e-8)

        B1_full = B1[:, :, None]  # (S1, S1, 1)

        # At this stage (2-level spatial hierarchy), we do NOT attach paths here.
        level1 = LorenzLevel(
            S=B1.shape[0],
            A=A1,
            B_states=B1,
            E_states=E1,
            D=D1,
            num_paths=1,
            B_paths=None,
            E_paths=None,
            B_states_paths=B1_full,
        )
        levels.append(level1)

    # NOTE: In the eventual 3-level design, a third level (top) will be added
    # here, with num_paths = num_paths_top, B_states_paths from the normalized
    # B_states_paths above, and B_paths/E_paths for path dynamics.

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T=T,
        H_blocks=H0,
        W_blocks=W0,
    )

    return hierarchy
