# lorenz_model.py
"""
Lorenz-specific renormalizing generative model (RGM) hierarchy.

This module defines:
- LorenzLevel: a single level in the hierarchy, with:
  * Hidden states s^l_t over a finite state space of size S_l
  * Emission model A^l(o | s) for observations at that level (if any)
  * Transition model B^l(s' | s) for temporal dynamics
  * Initial prior E^l(s_1)
  * Optional spatial mapping D^l from parent states to child configurations
  * Optional path factor (u_t) with its own B_paths, E_paths
  * Optional path-dependent state transitions B_states_paths(u, s' | s)
- LorenzHierarchy: a stack of LorenzLevel objects plus structural info
  (T, H_blocks, W_blocks, etc.)
- LorenzRGMParams: Dirichlet concentration parameters for A/B/E/C
  at each level, plus B_paths/E_paths/B_states_paths at the top level.
- Builders:
  * build_lorenz_hierarchy: construct the hierarchy with fixed
    initial parameters
  * init_dirichlet_params_from_hierarchy: initialize Dirichlet alphas
    from an existing hierarchy
  * build_lorenz_hierarchy_from_params: rebuild a hierarchy from
    learned Dirichlet parameters.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

# -----------------------------------------------------------------------------
# Core level and hierarchy dataclasses
# -----------------------------------------------------------------------------

@dataclass
class LorenzLevel:
    """
    A single level in the Lorenz RGM.

    Attributes:
        S: number of hidden states at this level
        A: (S, O) emission matrix P(o | s); O=0 if no explicit emissions
        B_states: (S, S) transition matrix P(s' | s) (path-averaged baseline)
        E_states: (S,) prior over initial state s_1
        D: for spatial levels > 0:
           - (S, child_config_dim) deterministic mapping from parent state
             to child state pattern (e.g., 4 child indices for 2x2 groups)
           for level 0: D is None
        num_paths: number of paths (policies) at this level
        B_paths: (num_paths, num_paths) transition over path states u_t
        E_paths: (num_paths,) prior over u_1
        B_states_paths: path-dependent state transitions.

        NOTE: In the initial construction we use shape (num_paths, S, S)
        (B_states_paths[u, s', s]); in the learned hierarchy we re-encode
        it as (S, S, num_paths) when needed by inference.
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
    Full Lorenz RGM hierarchy.

    Attributes:
        levels: list of LorenzLevel objects, lowest level at index 0
        states_grids: list of (T, H_l, W_l) integer grids encoding
            the layout / identity of states per level
        T: number of time steps
        H_blocks: number of patch rows at lowest level
        W_blocks: number of patch columns at lowest level
    """
    levels: List[LorenzLevel]
    states_grids: List[jnp.ndarray]
    T: int
    H_blocks: int
    W_blocks: int


# -----------------------------------------------------------------------------
# Dirichlet parameter container (for learning)
# -----------------------------------------------------------------------------

@dataclass
class LorenzRGMParams:
    """
    Dirichlet concentration parameters for a Lorenz RGM.

    Attributes:
        A_alpha: list of Dirichlet concentrations over A^l:
          A_alpha[l] has shape (S_l, O_l)
        B_alpha: list of Dirichlet concentrations over B_states^l:
          B_alpha[l] has shape (S_l, S_l)
        E_alpha: list of Dirichlet concentrations over E_states^l:
          E_alpha[l] has shape (S_l,)
        C_alpha: Dirichlet concentration over lowest-level outcomes C^0(o):
          shape (O_0,)

        B_states_paths_alpha: Dirichlet concentrations for path-dependent
          state transitions at the top level:
          shape (U, S_top, S_top) or None if not used
        B_paths_alpha: Dirichlet concentrations over path transitions:
          shape (U, U) or None if not used
        E_paths_alpha: Dirichlet concentrations over initial path states:
          shape (U,) or None if not used
    """
    A_alpha: List[jnp.ndarray]
    B_alpha: List[jnp.ndarray]
    E_alpha: List[jnp.ndarray]
    C_alpha: jnp.ndarray
    B_states_paths_alpha: Optional[jnp.ndarray]
    B_paths_alpha: Optional[jnp.ndarray]
    E_paths_alpha: Optional[jnp.ndarray]


# -----------------------------------------------------------------------------
# Helper functions: constructing A, B, E, path factors
# -----------------------------------------------------------------------------

def build_lorenz_A0(S0: int, K: int, L: int) -> jnp.ndarray:
    """
    Build the lowest-level emission matrix A0 for the Lorenz example.

    Hidden states encode K discrete coefficients, each taking L values.
    Outcomes are one-hot over K*L possible coefficient-value slots.
    """
    O = K * L
    assert S0 == L ** K, "S0 must be L^K for this construction."

    # Precompute state -> coefficients via base-L representation
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
    Build a simple transition matrix B(s'|s) with optional self-transition bias.
    """
    B = np.ones((S, S), dtype=np.float32)
    np.fill_diagonal(B, 1.0 + self_bias)
    B = B / (B.sum(axis=1, keepdims=True) + 1e-8)
    return jnp.array(B)


def build_uniform_E(S: int) -> jnp.ndarray:
    """
    Build a uniform prior over initial states.
    """
    E = np.ones((S,), dtype=np.float32)
    E = E / E.sum()
    return jnp.array(E)


def build_path_dynamics(num_paths: int) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Build a simple path factor dynamics (B_paths, E_paths) for the top level.

    Args:
        num_paths: number of path states u_t

    Returns:
        B_paths: (num_paths, num_paths) or None
        E_paths: (num_paths,) or None
    """
    if num_paths <= 0:
        return None, None

    B_paths = np.ones((num_paths, num_paths), dtype=np.float32)
    np.fill_diagonal(B_paths, 2.0)  # mild self-bias
    B_paths = B_paths / (B_paths.sum(axis=1, keepdims=True) + 1e-8)

    E_paths = np.ones((num_paths,), dtype=np.float32)
    E_paths = E_paths / E_paths.sum()

    return jnp.array(B_paths), jnp.array(E_paths)


def build_path_dependent_B_states(
    S: int,
    num_paths: int,
    self_bias: float = 1.0,
) -> jnp.ndarray:
    """
    Build initial path-dependent state transitions for the top level.

    Returns:
        B_states_paths: (num_paths, S, S)
    """
    B_base = build_uniform_B(S, self_bias=self_bias)
    B_stack = jnp.stack([B_base for _ in range(num_paths)], axis=0)
    return B_stack


# -----------------------------------------------------------------------------
# Building the Lorenz hierarchy (fixed parameters, no learning yet)
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
    lorenz_spatial_hierarchy: Optional[Dict[str, Any]] = None,
) -> LorenzHierarchy:
    """
    Build the Lorenz RGM hierarchy with fixed initial parameters.

    Assumes a spatial hierarchy from lorenz_renorm.build_lorenz_spatial_hierarchy.
    """
    if lorenz_spatial_hierarchy is None:
        raise ValueError(
            "lorenz_spatial_hierarchy must be provided (from lorenz_renorm)."
        )

    levels: List[LorenzLevel] = []
    states_grids: List[jnp.ndarray] = []

    # Lowest level: patches
    H0 = int(lorenz_spatial_hierarchy["H_blocks"])
    W0 = int(lorenz_spatial_hierarchy["W_blocks"])
    level0_states_grid = lorenz_spatial_hierarchy["levels"][0]["states_grid"]
    states_grids.append(level0_states_grid)

    S0 = L ** K
    O0 = K * L

    A0 = build_lorenz_A0(S0, K, L)
    B0 = build_uniform_B(S0, self_bias=1.0)
    E0 = build_uniform_E(S0)

    level0 = LorenzLevel(
        S=S0,
        A=A0,
        B_states=B0,
        E_states=E0,
        D=None,
        num_paths=0,
        B_paths=None,
        E_paths=None,
        B_states_paths=None,
    )
    levels.append(level0)

    # Spatial parent level
    current_num_levels = len(lorenz_spatial_hierarchy["levels"])
    if current_num_levels < 2:
        raise ValueError(
            "Spatial hierarchy must have at least 2 levels (patch and parent)."
        )

    spatial_level1 = lorenz_spatial_hierarchy["levels"][1]
    states_grid1 = spatial_level1["states_grid"]
    D1 = spatial_level1["D"]  # (S1, 4)

    states_grids.append(states_grid1)

    S1 = int(D1.shape[0])

    B1 = build_uniform_B(S1, self_bias=1.0)
    E1 = build_uniform_E(S1)
    A1 = jnp.zeros((S1, 0), dtype=jnp.float32)  # no explicit emissions at level 1

    # Path factor at top level
    B_paths, E_paths = build_path_dynamics(num_paths_top)
    B_states_paths = (
        build_path_dependent_B_states(S1, num_paths_top, self_bias=1.0)
        if num_paths_top > 0
        else None
    )

    level1 = LorenzLevel(
        S=S1,
        A=A1,
        B_states=B1,
        E_states=E1,
        D=D1,
        num_paths=num_paths_top,
        B_paths=B_paths,
        E_paths=E_paths,
        B_states_paths=B_states_paths,
    )
    levels.append(level1)

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T=T,
        H_blocks=H0,
        W_blocks=W0,
    )
    return hierarchy


# -----------------------------------------------------------------------------
# Helper: initialize Dirichlet parameters from an existing hierarchy
# -----------------------------------------------------------------------------

def init_dirichlet_params_from_hierarchy(
    hierarchy: LorenzHierarchy,
    K: int,
    L: int,
    alpha_A: float = 1.0,
    alpha_B: float = 1.0,
    alpha_E: float = 1.0,
    alpha_C: float = 1.0,
    alpha_B_paths: float = 1.0,
    alpha_E_paths: float = 1.0,
    alpha_B_states_paths: float = 1.0,
) -> LorenzRGMParams:
    """
    Initialize Dirichlet concentrations from an existing LorenzHierarchy.
    """
    A_alpha: List[jnp.ndarray] = []
    B_alpha: List[jnp.ndarray] = []
    E_alpha: List[jnp.ndarray] = []

    for level in hierarchy.levels:
        S = level.S
        O = level.A.shape[1]

        if O > 0:
            A_alpha_l = jnp.full((S, O), alpha_A, dtype=jnp.float32)
        else:
            A_alpha_l = jnp.zeros((S, 0), dtype=jnp.float32)

        B_alpha_l = jnp.full((S, S), alpha_B, dtype=jnp.float32)
        E_alpha_l = jnp.full((S,), alpha_E, dtype=jnp.float32)

        A_alpha.append(A_alpha_l)
        B_alpha.append(B_alpha_l)
        E_alpha.append(E_alpha_l)

    O0 = K * L
    C_alpha = jnp.full((O0,), alpha_C, dtype=jnp.float32)

    # Path-related alphas (assume only at top level)
    top_level = hierarchy.levels[-1]
    U = top_level.num_paths

    if U > 0:
        B_paths_alpha = jnp.full((U, U), alpha_B_paths, dtype=jnp.float32)
        E_paths_alpha = jnp.full((U,), alpha_E_paths, dtype=jnp.float32)

        if top_level.B_states_paths is not None:
            # top_level.B_states_paths is (U, S_top, S_top)
            S_top = top_level.S
            B_states_paths_alpha = jnp.full(
                (U, S_top, S_top),
                alpha_B_states_paths,
                dtype=jnp.float32,
            )
        else:
            B_states_paths_alpha = None
    else:
        B_paths_alpha = None
        E_paths_alpha = None
        B_states_paths_alpha = None

    return LorenzRGMParams(
        A_alpha=A_alpha,
        B_alpha=B_alpha,
        E_alpha=E_alpha,
        C_alpha=C_alpha,
        B_states_paths_alpha=B_states_paths_alpha,
        B_paths_alpha=B_paths_alpha,
        E_paths_alpha=E_paths_alpha,
    )


# -----------------------------------------------------------------------------
# Helper: build hierarchy from learned Dirichlet parameters
# -----------------------------------------------------------------------------

def build_lorenz_hierarchy_from_params(
    lorenz_spatial_hierarchy: Dict[str, Any],
    params: LorenzRGMParams,
    num_paths_top: int,
) -> LorenzHierarchy:
    """
    Rebuild a LorenzHierarchy using learned Dirichlet parameters.
    """
    # Normalize A, B, E from alphas
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

    levels: List[LorenzLevel] = []
    states_grids: List[jnp.ndarray] = []

    # Level 0
    level0_states_grid = lorenz_spatial_hierarchy["levels"][0]["states_grid"]
    states_grids.append(level0_states_grid)
    H0 = int(lorenz_spatial_hierarchy["H_blocks"])
    W0 = int(lorenz_spatial_hierarchy["W_blocks"])
    T = int(lorenz_spatial_hierarchy.get("T", level0_states_grid.shape[0]))

    A0 = A_list[0]
    B0 = B_list[0]
    E0 = E_list[0]
    S0 = A0.shape[0]

    level0 = LorenzLevel(
        S=S0,
        A=A0,
        B_states=B0,
        E_states=E0,
        D=None,
        num_paths=0,
        B_paths=None,
        E_paths=None,
        B_states_paths=None,
    )
    levels.append(level0)

    # Level 1 (spatial parent, top in this configuration)
    spatial_level1 = lorenz_spatial_hierarchy["levels"][1]
    states_grid1 = spatial_level1["states_grid"]
    D1 = spatial_level1["D"]  # (S1, 4)
    states_grids.append(states_grid1)

    A1 = A_list[1]
    B1 = B_list[1]
    E1 = E_list[1]
    S1 = B1.shape[0]

    # Path-related learned parameters (if any)
    if (
        num_paths_top > 0
        and params.B_paths_alpha is not None
        and params.E_paths_alpha is not None
    ):
        B_paths = params.B_paths_alpha / (
            params.B_paths_alpha.sum(axis=1, keepdims=True) + 1e-8
        )
        E_paths = params.E_paths_alpha / (params.E_paths_alpha.sum() + 1e-8)
    else:
        B_paths, E_paths = build_path_dynamics(num_paths_top)

    if num_paths_top > 0 and params.B_states_paths_alpha is not None:
        # params.B_states_paths_alpha is (U, S1, S1); normalize over s' axis (axis=2)
        B_states_paths = params.B_states_paths_alpha / (
            params.B_states_paths_alpha.sum(axis=2, keepdims=True) + 1e-8
        )
    else:
        B_states_paths = (
            build_path_dependent_B_states(S1, num_paths_top, self_bias=1.0)
            if num_paths_top > 0
            else None
        )

    level1 = LorenzLevel(
        S=S1,
        A=A1,
        B_states=B1,
        E_states=E1,
        D=D1,
        num_paths=num_paths_top,
        B_paths=B_paths,
        E_paths=E_paths,
        B_states_paths=B_states_paths,
    )
    levels.append(level1)

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T=T,
        H_blocks=H0,
        W_blocks=W0,
    )
    return hierarchy
