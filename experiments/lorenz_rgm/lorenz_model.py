# lorenz_model.py
"""
Model construction for the Lorenz RGM.

This module defines:
- LorenzLevel: a single spatial / temporal level in the RGM.
- LorenzHierarchy: a stack of LorenzLevel objects with associated state grids.
- LorenzRGMParams: Dirichlet concentration parameters for A/B/E/C and
  path-related quantities.
- Functions to build an initial Lorenz hierarchy given a spatial
  RG hierarchy (states_grids, D mappings).
- Functions to initialize and convert Dirichlet parameters for A, B, E, C,
  and path-related quantities.

IMPORTANT CONVENTIONS (unified B layout):

- For each level ℓ with S_l hidden states and U_l path/control values:

    B_states_full_l[s_next, s, u_l] = P(s_next | s, u_l)

  with shape (S_l, S_l, U_l). For levels without multiple paths we set
  U_l = 1 and fix u_l = 0, so the effective transition matrix is:

      B_states_l = B_states_full_l[:, :, 0]  # (S_l, S_l)

- At the top level L-1, if num_paths_top > 1, we also define:

    B_paths: (U, U)       with B_paths[u_next, u] = P(u_next | u)
    E_paths: (U,)         with E_paths[u] = P(u_0 = u)
    B_states_paths: (S_top, S_top, U) full path-dependent transitions

All functions below are written in a level-agnostic way so the hierarchy
can have 2, 3, or more spatial levels; the Lorenz example uses 3.
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
        H_blocks: number of lowest-level patches vertically
        W_blocks: number of lowest-level patches horizontally
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
# Utilities for constructing A, B, E, and path factors
# -----------------------------------------------------------------------------

def build_lorenz_A0(S0: int, K: int, L: int) -> jnp.ndarray:
    """
    Build the lowest-level emission matrix A0 for the Lorenz example.

    Hidden states encode K discrete coefficients, each taking L values.
    Outcomes are one-hot over K*L possible coefficient-value slots.

    Construction:
        - each hidden state s corresponds to a unique K-tuple (c1,...,cK)
        - A0[s, o] = 1 if the one-hot outcome o is consistent with that tuple.

    This is a fixed initialization; learning will update A_alpha and then
    re-normalize to get a learned A0.
    """
    O = K * L
    assert S0 == L ** K, "S0 must be L^K for this construction."

    coeffs = np.zeros((S0, K), dtype=np.int32)
    for s in range(S0):
        x = s
        for k in reversed(range(K)):
            coeffs[s, k] = x % L
            x //= L

    A0 = np.zeros((S0, O), dtype=np.float32)
    for s in range(S0):
        for k in range(K):
            idx = k * L + coeffs[s, k]
            A0[s, idx] = 1.0

    A0 = A0 / (A0.sum(axis=1, keepdims=True) + 1e-8)
    return jnp.array(A0)


def build_uniform_B(S: int, self_bias: float = 1.0) -> jnp.ndarray:
    """
    Build a simple transition matrix B(s_next|s) with a self-transition bias.

    Returns:
        B: (S, S) with B[s_next, s] = P(s_next | s)
    """
    B = np.ones((S, S), dtype=np.float32)
    np.fill_diagonal(B, 1.0 + self_bias)
    B = B / (B.sum(axis=0, keepdims=True) + 1e-8)
    return jnp.array(B)


def build_uniform_E(S: int) -> jnp.ndarray:
    """
    Build a uniform prior over initial states.

    Returns:
        E: (S,) vector with uniform probabilities.
    """
    E = np.ones((S,), dtype=np.float32)
    E = E / E.sum()
    return jnp.array(E)


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
# Build hierarchy from spatial RG structure (level-agnostic)
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

    Args:
        T, dt, img_size, patch_size, K, L, thickness: config parameters
        num_spatial_levels: number of RG steps used to build spatial_h;
            total spatial levels = num_spatial_levels + 1
        num_paths_top: number of path states at the top level
        lorenz_spatial_hierarchy: dict with keys:
            - "levels": list of dicts:
                * level[0]: {"states_grid": (T,H0,W0)}
                * level[l>0]: {"states_grid": (T,Hl,Wl), "D": (Sl,4)}
            - "H_blocks", "W_blocks"

    Returns:
        LorenzHierarchy with len(levels) = num_spatial_levels + 1.
        Only the top level has num_paths > 1 (path factor) if num_paths_top>1.
    """
    levels: List[LorenzLevel] = []
    states_grids: List[jnp.ndarray] = []

    # Level metadata
    spatial_levels = lorenz_spatial_hierarchy["levels"]
    L_total = len(spatial_levels)  # should be num_spatial_levels + 1
    assert L_total >= 1, "Spatial hierarchy must have at least one level."

    H0 = int(lorenz_spatial_hierarchy["H_blocks"])
    W0 = int(lorenz_spatial_hierarchy["W_blocks"])
    T_h = int(lorenz_spatial_hierarchy.get("T", T))

    # ---------------------
    # Level 0 (patch level)
    # ---------------------
    level0_states_grid = spatial_levels[0]["states_grid"]
    states_grids.append(level0_states_grid)

    S0 = int(jnp.max(level0_states_grid) + 1)
    O0 = K * L

    # Emission A0: (S0, O0)
    A0 = jnp.ones((S0, O0), dtype=jnp.float32)
    A0 = A0 / (A0.sum(axis=1, keepdims=True) + 1e-8)

    # Transition B0_full: (S0, S0, U0=1)
    B0_base = build_uniform_B(S0, self_bias=1.0)  # (S0, S0)
    B0_full = B0_base[:, :, None]                 # (S0, S0, 1)

    # Prior E0: (S0,)
    E0 = jnp.ones((S0,), dtype=jnp.float32)
    E0 = E0 / (E0.sum() + 1e-8)

    level0 = LorenzLevel(
        S=S0,
        A=A0,
        B_states=B0_base,
        E_states=E0,
        D=None,
        num_paths=1,
        B_paths=None,
        E_paths=None,
        B_states_paths=B0_full,
    )
    levels.append(level0)

    # ---------------------
    # Higher spatial levels (1 .. L_total-1)
    # ---------------------
    for l in range(1, L_total):
        spatial_level = spatial_levels[l]
        states_grid_l = spatial_level["states_grid"]
        D_l = spatial_level["D"]  # (S_l, 4)
        states_grids.append(states_grid_l)

        S_l = int(jnp.max(states_grid_l) + 1)

        # No direct observations at higher levels (for now)
        A_l = jnp.zeros((S_l, 0), dtype=jnp.float32)

        # Path-free transitions and priors at this level
        B_l_base = build_uniform_B(S_l, self_bias=1.0)  # (S_l, S_l)
        B_l_full = B_l_base[:, :, None]                 # (S_l, S_l, 1)

        E_l = jnp.ones((S_l,), dtype=jnp.float32)
        E_l = E_l / (E_l.sum() + 1e-8)

        level_l = LorenzLevel(
            S=S_l,
            A=A_l,
            B_states=B_l_base,
            E_states=E_l,
            D=D_l,
            num_paths=1,          # will be overridden at top level if needed
            B_paths=None,
            E_paths=None,
            B_states_paths=B_l_full,
        )
        levels.append(level_l)

    # ---------------------
    # Attach path factor at top level (if num_paths_top > 1)
    # ---------------------
    if num_paths_top > 1:
        top_idx = len(levels) - 1
        top_level = levels[top_idx]
        S_top = top_level.S

        B_paths, E_paths = build_path_dynamics(num_paths_top)
        B_states_paths = build_path_dependent_B_states(S_top, num_paths_top, self_bias=1.0)

        # Replace top level with path-enabled version
        levels[top_idx] = LorenzLevel(
            S=S_top,
            A=top_level.A,
            B_states=top_level.B_states,
            E_states=top_level.E_states,
            D=top_level.D,
            num_paths=num_paths_top,
            B_paths=B_paths,
            E_paths=E_paths,
            B_states_paths=B_states_paths,
        )

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
    if num_paths_top > 1:
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

    spatial_levels = lorenz_spatial_hierarchy["levels"]
    L_total = len(spatial_levels)

    A_alpha_list = params.A_alpha
    B_alpha_list = params.B_alpha
    E_alpha_list = params.E_alpha

    # Normalize path-related alphas if present
    if params.B_states_paths_alpha is not None and num_paths_top > 1:
        B_states_paths = params.B_states_paths_alpha / (
            params.B_states_paths_alpha.sum(axis=0, keepdims=True) + 1e-8
        )  # (S_top, S_top, U)
    else:
        B_states_paths = None

    if params.B_paths_alpha is not None and params.E_paths_alpha is not None and num_paths_top > 1:
        B_paths = params.B_paths_alpha / (
            params.B_paths_alpha.sum(axis=0, keepdims=True) + 1e-8
        )  # (U, U) column-normalized
        E_paths = params.E_paths_alpha / (params.E_paths_alpha.sum() + 1e-8)
    else:
        B_paths, E_paths = None, None

    # ----- Level 0 -----
    level0_states_grid = spatial_levels[0]["states_grid"]
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

    # ----- Higher levels -----
    for l in range(1, L_total):
        spatial_level = spatial_levels[l]
        states_grid_l = spatial_level["states_grid"]
        D_l = spatial_level["D"]  # (S_l, 4)
        states_grids.append(states_grid_l)

        A_alpha_l = A_alpha_list[l]
        B_alpha_l = B_alpha_list[l]
        E_alpha_l = E_alpha_list[l]

        if A_alpha_l.size > 0:
            A_l = A_alpha_l / (A_alpha_l.sum(axis=1, keepdims=True) + 1e-8)
        else:
            A_l = A_alpha_l

        B_l = B_alpha_l / (B_alpha_l.sum(axis=0, keepdims=True) + 1e-8)
        E_l = E_alpha_l / (E_alpha_l.sum() + 1e-8)

        B_l_full = B_l[:, :, None]  # (S_l, S_l, 1)

        level_l = LorenzLevel(
            S=B_l.shape[0],
            A=A_l,
            B_states=B_l,
            E_states=E_l,
            D=D_l,
            num_paths=1,
            B_paths=None,
            E_paths=None,
            B_states_paths=B_l_full,
        )
        levels.append(level_l)

    # ----- Attach path factor at top (if any) -----
    if num_paths_top > 1 and B_states_paths is not None and B_paths is not None and E_paths is not None:
        top_idx = len(levels) - 1
        top_level = levels[top_idx]
        S_top = top_level.S

        assert B_states_paths.shape[0] == S_top, "B_states_paths S_top mismatch."
        assert B_states_paths.shape[1] == S_top, "B_states_paths S_top mismatch."
        assert B_states_paths.shape[2] == num_paths_top, "B_states_paths U mismatch."

        levels[top_idx] = LorenzLevel(
            S=S_top,
            A=top_level.A,
            B_states=top_level.B_states,
            E_states=top_level.E_states,
            D=top_level.D,
            num_paths=num_paths_top,
            B_paths=B_paths,
            E_paths=E_paths,
            B_states_paths=B_states_paths,
        )

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T=T,
        H_blocks=H0,
        W_blocks=W0,
    )

    return hierarchy
