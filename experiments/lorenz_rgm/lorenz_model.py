# lorenz_model.py
"""
Lorenz-specific renormalizing generative model (RGM) hierarchy.

This module defines:
- LorenzLevel: a single level in the hierarchy, with:
  * Hidden states s^l_t over a finite state space of size S_l
  * Emission model A^l(o | s) for observations at that level (if any)
  * Initial prior over states E_states^l(s_1)
  * Optional spatial mapping D_state_from_parent^l from parent states to
    child state configurations (D tensors)
  * Path factor u^l_t with:
    - C_paths^l(u_{t+1} | u_t) (path transitions, "C" in the paper)
    - E_paths^l(u_1)
    - B_states_paths^l(s' | s, u) (state transitions conditioned on paths)
    - E_paths_from_parent^l(u^l | s^{l+1}) (parent state -> child paths, E tensors)
- LorenzHierarchy: a stack of LorenzLevel objects plus structural info,
  including explicit temporal block structure across levels.
- LorenzRGMParams: Dirichlet concentration parameters for A/D/E/B/C
  at each level, and preferences over lowest-level outcomes.

Builders:
- build_lorenz_hierarchy: construct the hierarchy with fixed initial parameters
- init_dirichlet_params_from_hierarchy: initialize Dirichlet alphas
- build_lorenz_hierarchy_from_params: rebuild a hierarchy from learned
  Dirichlet parameters.
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

      E_states: (S,) prior over initial state s_1 at this level

      D_state_from_parent:
        For spatial levels > 0:
          (S, child_config_dim) deterministic mapping from parent state
          to child state pattern (e.g., 4 child indices for 2x2 groups)
        For level 0 (lowest), this is None.

      num_paths: number of paths (policies) at this level, U >= 1

      C_paths: (U, U) path transition matrix P(u_{t+1} | u_t)
        This is the C tensor for paths at this level.

      E_paths: (U,) prior over initial path state u_1 at this level

      B_states_paths:
        Path-dependent state transitions at this level:
        B_states_paths[s_next, s, u] = P(s_next | s, u_t = u)
        Shape: (S, S, U) or None if num_paths <= 0.

      E_paths_from_parent:
        Mapping from parent state to initial paths at this level:
        E_paths_from_parent[u, s_parent] ∝ P(u_t = u | s_parent_t = s_parent)
        Shape: (U, S_parent) or None for the top level (no parent).
    """
    S: int
    A: jnp.ndarray
    E_states: jnp.ndarray
    D_state_from_parent: Optional[jnp.ndarray]

    num_paths: int
    C_paths: Optional[jnp.ndarray]
    E_paths: Optional[jnp.ndarray]

    B_states_paths: Optional[jnp.ndarray]
    E_paths_from_parent: Optional[jnp.ndarray]


@dataclass
class LorenzHierarchy:
    """
    Full Lorenz RGM hierarchy with explicit temporal block structure.

    Attributes:
      levels: list of LorenzLevel objects, lowest level at index 0
      states_grids: list of (T0, H_l, W_l) integer grids encoding
        the layout / identity of states per level in space
      # Temporal structure:
      T0: number of fine (level-0) time steps
      T1: number of level-1 time steps
      T2: number of level-2 time steps
      K0: number of level-0 steps per level-1 step (T0 = K0 * T1)
      K1: number of level-1 steps per level-2 step (T1 = K1 * T2)
      H_blocks: number of patch rows at lowest level
      W_blocks: number of patch columns at lowest level
    """
    levels: List[LorenzLevel]
    states_grids: List[jnp.ndarray]
    T0: int
    T1: int
    T2: int
    K0: int
    K1: int
    H_blocks: int
    W_blocks: int

# -----------------------------------------------------------------------------
# Dirichlet parameter container (for learning)
# -----------------------------------------------------------------------------

@dataclass
class LorenzRGMParams:
    """
    Dirichlet concentration parameters for a Lorenz RGM.

    A-family parameters:
      A_alpha[l]: (S_l, O_l) over emission matrices A^l
      E_states_alpha[l]: (S_l,) over initial state priors at level l
      D_state_from_parent_alpha[l]: (S_l, child_config_dim) or None
        over state-from-parent mappings D^l
      E_paths_from_parent_alpha[l]: (U_l, S_parent_l) or None
        over path-from-parent mappings E^l
      E_paths_alpha[l]: (U_l,) or None over initial path priors at level l

    B-family parameters (state transitions given paths):
      B_states_paths_alpha[l]: (S_l, S_l, U_l) or None
        over B_states_paths^l(s' | s, u)

    C-family parameters (path transitions):
      C_paths_alpha[l]: (U_l, U_l) or None
        over C_paths^l(u_{t+1} | u_t)

    Preferences over outcomes at lowest level:
      pref_alpha: (O_0,) Dirichlet concentration over outcome preferences
      C^0(o) in active inference notation.
    """
    A_alpha: List[jnp.ndarray]
    E_states_alpha: List[jnp.ndarray]
    D_state_from_parent_alpha: List[Optional[jnp.ndarray]]
    E_paths_from_parent_alpha: List[Optional[jnp.ndarray]]
    E_paths_alpha: List[Optional[jnp.ndarray]]

    B_states_paths_alpha: List[Optional[jnp.ndarray]]

    C_paths_alpha: List[Optional[jnp.ndarray]]

    pref_alpha: jnp.ndarray

# -----------------------------------------------------------------------------
# Helper functions: constructing A, E, path factors
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


def build_uniform_E_states(S: int) -> jnp.ndarray:
    """
    Build a uniform prior over initial states at a given level.
    """
    E = np.ones((S,), dtype=np.float32)
    E = E / E.sum()
    return jnp.array(E)


def build_path_dynamics(num_paths: int) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    """
    Build simple path dynamics (C_paths, E_paths) for a level.

    Args:
      num_paths: number of path states u_t

    Returns:
      C_paths: (U, U) with C_paths[u_next, u] = P(u_{t+1} | u_t), or None
      E_paths: (U,) with E_paths[u] = P(u_1 = u), or None
    """
    if num_paths <= 0:
        return None, None

    C_paths = np.ones((num_paths, num_paths), dtype=np.float32)
    np.fill_diagonal(C_paths, 2.0)  # mild self-bias
    C_paths = C_paths / (C_paths.sum(axis=1, keepdims=True) + 1e-8)

    E_paths = np.ones((num_paths,), dtype=np.float32)
    E_paths = E_paths / E_paths.sum()

    return jnp.array(C_paths), jnp.array(E_paths)


def build_path_dependent_B_states(
    S: int,
    num_paths: int,
    self_bias: float = 1.0,
) -> Optional[jnp.ndarray]:
    """
    Build initial path-dependent state transitions for a level.

    Convention:
      B_states_paths[s_next, s, u] = P(s_next | s, u)

    Returns:
      B_states_paths: (S, S, U) or None if num_paths <= 0
    """
    if num_paths <= 0:
        return None

    # Baseline transition with self-bias
    B_base = np.ones((S, S), dtype=np.float32)
    np.fill_diagonal(B_base, 1.0 + self_bias)
    B_base = B_base / (B_base.sum(axis=1, keepdims=True) + 1e-8)

    # Stack along the last axis to encode path index u
    B_stack = jnp.stack([jnp.array(B_base) for _ in range(num_paths)], axis=2)  # (S, S, U)
    return B_stack


def build_uniform_E_paths_from_parent(
    num_paths: int,
    S_parent: int,
) -> Optional[jnp.ndarray]:
    """
    Build a simple uniform mapping from parent states to child paths:

      E_paths_from_parent[u, s_parent] ∝ P(u_t = u | s_parent_t = s_parent)

    Returns:
      (U, S_parent) or None if num_paths <= 0 or this is the top level.
    """
    if num_paths <= 0:
        return None

    E_map = np.ones((num_paths, S_parent), dtype=np.float32)
    E_map = E_map / (E_map.sum(axis=0, keepdims=True) + 1e-8)
    return jnp.array(E_map)

# -----------------------------------------------------------------------------
# Building the Lorenz hierarchy (fixed parameters, with temporal blocks)
# -----------------------------------------------------------------------------

def build_lorenz_hierarchy(
    T0: int,
    img_size: int,
    patch_size: int,
    K: int,
    L: int,
    thickness: int,
    num_spatial_levels: int,
    num_paths_levels: List[int],
    K0: int,
    K1: int,
    lorenz_spatial_hierarchy: Optional[Dict[str, Any]] = None,
) -> LorenzHierarchy:
    """
    Build the Lorenz RGM hierarchy with fixed initial parameters and
    explicit temporal block structure.

    Args:
      T0: number of lowest-level time steps (fine scale)
      img_size, patch_size, K, L, thickness: data/encoding params for Lorenz
      num_spatial_levels: number of spatial parent levels above patches
      num_paths_levels: list length = 1 + num_spatial_levels with U_l per level
      K0: number of level-0 steps per level-1 step
      K1: number of level-1 steps per level-2 step
      lorenz_spatial_hierarchy: output from lorenz_renorm.build_lorenz_spatial_hierarchy

    Temporal structure:
      T0 = K0 * T1
      T1 = K1 * T2

    Spatial convention:
      - lorenz_spatial_hierarchy["levels"][0] is the patch level.
      - lorenz_spatial_hierarchy["levels"][l] for l > 0 are successive
        spatial RG levels (parents).
    """
    if lorenz_spatial_hierarchy is None:
        raise ValueError(
            "lorenz_spatial_hierarchy must be provided (from lorenz_renorm)."
        )

    # Derive T1, T2 from T0 and K0, K1
    if T0 % K0 != 0:
        raise ValueError(f"T0={T0} must be divisible by K0={K0}.")
    T1 = T0 // K0
    if T1 % K1 != 0:
        raise ValueError(f"T1={T1} must be divisible by K1={K1}.")
    T2 = T1 // K1

    spatial_levels = lorenz_spatial_hierarchy["levels"]
    num_spatial_levels_available = len(spatial_levels) - 1  # excluding patch level

    if num_spatial_levels < 0:
        raise ValueError("num_spatial_levels must be >= 0.")

    if num_spatial_levels > num_spatial_levels_available:
        raise ValueError(
            f"Requested num_spatial_levels={num_spatial_levels}, "
            f"but spatial hierarchy only has {num_spatial_levels_available} "
            "parent levels."
        )

    # We always include the patch level (level 0) plus num_spatial_levels parents.
    total_lorenz_levels = 1 + num_spatial_levels
    if len(num_paths_levels) != total_lorenz_levels:
        raise ValueError(
            f"num_paths_levels must have length {total_lorenz_levels}, "
            f"got {len(num_paths_levels)}."
        )

    levels: List[LorenzLevel] = []
    states_grids: List[jnp.ndarray] = []

    # Lowest level: patches (LorenzLevel 0)
    H0 = int(lorenz_spatial_hierarchy["H_blocks"])
    W0 = int(lorenz_spatial_hierarchy["W_blocks"])
    level0_states_grid = spatial_levels[0]["states_grid"]
    states_grids.append(level0_states_grid)

    S0 = L ** K
    O0 = K * L

    A0 = build_lorenz_A0(S0, K, L)
    E0 = build_uniform_E_states(S0)

    U0 = int(num_paths_levels[0])
    C_paths0, E_paths0 = build_path_dynamics(U0)
    B_states_paths0 = build_path_dependent_B_states(S0, U0, self_bias=1.0)

    # No parent above level 0 for states or paths
    E_paths_from_parent0 = None

    level0 = LorenzLevel(
        S=S0,
        A=A0,
        E_states=E0,
        D_state_from_parent=None,
        num_paths=U0,
        C_paths=C_paths0,
        E_paths=E_paths0,
        B_states_paths=B_states_paths0,
        E_paths_from_parent=E_paths_from_parent0,
    )

    levels.append(level0)

    # Higher spatial levels 1..num_spatial_levels (LorenzLevels 1..)
    for l in range(1, total_lorenz_levels):
        spatial_level = spatial_levels[l]
        states_grid_l = spatial_level["states_grid"]
        D_l = spatial_level["D"]  # (S_l, child_config_dim)
        states_grids.append(states_grid_l)

        S_l = int(D_l.shape[0])

        # No explicit emissions at higher levels in this Lorenz example
        A_l = jnp.zeros((S_l, 0), dtype=jnp.float32)
        E_l = build_uniform_E_states(S_l)

        U_l = int(num_paths_levels[l])
        C_paths_l, E_paths_l = build_path_dynamics(U_l)
        B_states_paths_l = build_path_dependent_B_states(S_l, U_l, self_bias=1.0)

        # Parent for this level is the next spatial level if it exists
        if l < total_lorenz_levels - 1:
            S_parent = int(spatial_levels[l + 1]["D"].shape[0])
            E_paths_from_parent_l = build_uniform_E_paths_from_parent(U_l, S_parent)
        else:
            # Top level: no parent
            E_paths_from_parent_l = None

        level_l = LorenzLevel(
            S=S_l,
            A=A_l,
            E_states=E_l,
            D_state_from_parent=D_l,
            num_paths=U_l,
            C_paths=C_paths_l,
            E_paths=E_paths_l,
            B_states_paths=B_states_paths_l,
            E_paths_from_parent=E_paths_from_parent_l,
        )

        levels.append(level_l)

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T0=T0,
        T1=T1,
        T2=T2,
        K0=K0,
        K1=K1,
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
    alpha_E_states: float = 1.0,
    alpha_D_state_from_parent: float = 1.0,
    alpha_E_paths_from_parent: float = 1.0,
    alpha_E_paths: float = 1.0,
    alpha_B_states_paths: float = 1.0,
    alpha_C_paths: float = 1.0,
    alpha_pref: float = 1.0,
) -> LorenzRGMParams:
    """
    Initialize Dirichlet concentrations from an existing LorenzHierarchy.
    """
    A_alpha: List[jnp.ndarray] = []
    E_states_alpha: List[jnp.ndarray] = []
    D_state_from_parent_alpha: List[Optional[jnp.ndarray]] = []
    E_paths_from_parent_alpha: List[Optional[jnp.ndarray]] = []
    E_paths_alpha: List[Optional[jnp.ndarray]] = []
    B_states_paths_alpha: List[Optional[jnp.ndarray]] = []
    C_paths_alpha: List[Optional[jnp.ndarray]] = []

    for level in hierarchy.levels:
        S = level.S
        O = level.A.shape[1]

        if O > 0:
            A_alpha_l = jnp.full((S, O), alpha_A, dtype=jnp.float32)
        else:
            A_alpha_l = jnp.zeros((S, 0), dtype=jnp.float32)

        E_states_alpha_l = jnp.full((S,), alpha_E_states, dtype=jnp.float32)

        if level.D_state_from_parent is not None:
            D_alpha_l = jnp.full(
                level.D_state_from_parent.shape,
                alpha_D_state_from_parent,
                dtype=jnp.float32,
            )
        else:
            D_alpha_l = None

        if level.E_paths_from_parent is not None:
            E_paths_from_parent_alpha_l = jnp.full(
                level.E_paths_from_parent.shape,
                alpha_E_paths_from_parent,
                dtype=jnp.float32,
            )
        else:
            E_paths_from_parent_alpha_l = None

        if level.E_paths is not None and level.num_paths > 0:
            E_paths_alpha_l = jnp.full(
                (level.num_paths,), alpha_E_paths, dtype=jnp.float32
            )
        else:
            E_paths_alpha_l = None

        if level.B_states_paths is not None and level.num_paths > 0:
            B_states_paths_alpha_l = jnp.full(
                level.B_states_paths.shape,
                alpha_B_states_paths,
                dtype=jnp.float32,
            )
        else:
            B_states_paths_alpha_l = None

        if level.C_paths is not None and level.num_paths > 0:
            C_paths_alpha_l = jnp.full(
                level.C_paths.shape,
                alpha_C_paths,
                dtype=jnp.float32,
            )
        else:
            C_paths_alpha_l = None

        A_alpha.append(A_alpha_l)
        E_states_alpha.append(E_states_alpha_l)
        D_state_from_parent_alpha.append(D_alpha_l)
        E_paths_from_parent_alpha.append(E_paths_from_parent_alpha_l)
        E_paths_alpha.append(E_paths_alpha_l)
        B_states_paths_alpha.append(B_states_paths_alpha_l)
        C_paths_alpha.append(C_paths_alpha_l)

    O0 = K * L
    pref_alpha = jnp.full((O0,), alpha_pref, dtype=jnp.float32)

    return LorenzRGMParams(
        A_alpha=A_alpha,
        E_states_alpha=E_states_alpha,
        D_state_from_parent_alpha=D_state_from_parent_alpha,
        E_paths_from_parent_alpha=E_paths_from_parent_alpha,
        E_paths_alpha=E_paths_alpha,
        B_states_paths_alpha=B_states_paths_alpha,
        C_paths_alpha=C_paths_alpha,
        pref_alpha=pref_alpha,
    )

# -----------------------------------------------------------------------------
# Helper: build hierarchy from learned Dirichlet parameters
# -----------------------------------------------------------------------------

def build_lorenz_hierarchy_from_params(
    lorenz_spatial_hierarchy: Dict[str, Any],
    params: LorenzRGMParams,
    T0: int,
    K0: int,
    K1: int,
) -> LorenzHierarchy:
    """
    Rebuild a LorenzHierarchy using learned Dirichlet parameters.

    The number of Lorenz levels is determined by the length of params.A_alpha,
    params.E_states_alpha, etc. It is assumed to match the spatial hierarchy.

    Args:
      lorenz_spatial_hierarchy: spatial structure (states_grids, D tensors)
      params: learned Dirichlet parameters
      T0, K0, K1: temporal structure (same conventions as build_lorenz_hierarchy)
    """
    # Derive T1, T2 from T0 and K0, K1
    if T0 % K0 != 0:
        raise ValueError(f"T0={T0} must be divisible by K0={K0}.")
    T1 = T0 // K0
    if T1 % K1 != 0:
        raise ValueError(f"T1={T1} must be divisible by K1={K1}.")
    T2 = T1 // K1

    spatial_levels = lorenz_spatial_hierarchy["levels"]

    A_list: List[jnp.ndarray] = []
    E_states_list: List[jnp.ndarray] = []
    D_state_from_parent_list: List[Optional[jnp.ndarray]] = []
    E_paths_from_parent_list: List[Optional[jnp.ndarray]] = []
    E_paths_list: List[Optional[jnp.ndarray]] = []
    B_states_paths_list: List[Optional[jnp.ndarray]] = []
    C_paths_list: List[Optional[jnp.ndarray]] = []

    for (
        A_alpha_l,
        E_states_alpha_l,
        D_alpha_l,
        E_paths_from_parent_alpha_l,
        E_paths_alpha_l,
        B_states_paths_alpha_l,
        C_paths_alpha_l,
    ) in zip(
        params.A_alpha,
        params.E_states_alpha,
        params.D_state_from_parent_alpha,
        params.E_paths_from_parent_alpha,
        params.E_paths_alpha,
        params.B_states_paths_alpha,
        params.C_paths_alpha,
    ):
        # A
        if A_alpha_l.size > 0:
            A_l = A_alpha_l / (A_alpha_l.sum(axis=1, keepdims=True) + 1e-8)
        else:
            A_l = A_alpha_l

        # E_states
        E_states_l = E_states_alpha_l / (E_states_alpha_l.sum() + 1e-8)

        # D_state_from_parent
        if D_alpha_l is not None:
            D_l = D_alpha_l / (D_alpha_l.sum(axis=1, keepdims=True) + 1e-8)
        else:
            D_l = None

        # E_paths_from_parent
        if E_paths_from_parent_alpha_l is not None:
            E_paths_from_parent_l = E_paths_from_parent_alpha_l / (
                E_paths_from_parent_alpha_l.sum(axis=0, keepdims=True) + 1e-8
            )
        else:
            E_paths_from_parent_l = None

        # E_paths
        if E_paths_alpha_l is not None:
            E_paths_l = E_paths_alpha_l / (E_paths_alpha_l.sum() + 1e-8)
        else:
            E_paths_l = None

        # B_states_paths
        if B_states_paths_alpha_l is not None:
            B_states_paths_l = B_states_paths_alpha_l / (
                B_states_paths_alpha_l.sum(axis=0, keepdims=True) + 1e-8
            )
        else:
            B_states_paths_l = None

        # C_paths
        if C_paths_alpha_l is not None:
            C_paths_l = C_paths_alpha_l / (
                C_paths_alpha_l.sum(axis=1, keepdims=True) + 1e-8
            )
        else:
            C_paths_l = None

        A_list.append(A_l)
        E_states_list.append(E_states_l)
        D_state_from_parent_list.append(D_l)
        E_paths_from_parent_list.append(E_paths_from_parent_l)
        E_paths_list.append(E_paths_l)
        B_states_paths_list.append(B_states_paths_l)
        C_paths_list.append(C_paths_l)

    levels: List[LorenzLevel] = []
    states_grids: List[jnp.ndarray] = []

    # Level 0 (patch level)
    level0_states_grid = spatial_levels[0]["states_grid"]
    states_grids.append(level0_states_grid)
    H0 = int(lorenz_spatial_hierarchy["H_blocks"])
    W0 = int(lorenz_spatial_hierarchy["W_blocks"])

    for l in range(len(A_list)):
        spatial_level = spatial_levels[l]
        states_grid_l = spatial_level["states_grid"]
        if l < len(states_grids):
            states_grids[l] = states_grid_l
        else:
            states_grids.append(states_grid_l)

        A_l = A_list[l]
        E_states_l = E_states_list[l]
        D_l = D_state_from_parent_list[l]
        E_paths_from_parent_l = E_paths_from_parent_list[l]
        E_paths_l = E_paths_list[l]
        B_states_paths_l = B_states_paths_list[l]
        C_paths_l = C_paths_list[l]

        S_l = A_l.shape[0] if A_l.size > 0 else E_states_l.shape[0]
        num_paths_l = E_paths_l.shape[0] if E_paths_l is not None else 0

        level_l = LorenzLevel(
            S=S_l,
            A=A_l,
            E_states=E_states_l,
            D_state_from_parent=D_l,
            num_paths=num_paths_l,
            C_paths=C_paths_l,
            E_paths=E_paths_l,
            B_states_paths=B_states_paths_l,
            E_paths_from_parent=E_paths_from_parent_l,
        )

        if l < len(levels):
            levels[l] = level_l
        else:
            levels.append(level_l)

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T0=T0,
        T1=T1,
        T2=T2,
        K0=K0,
        K1=K1,
        H_blocks=H0,
        W_blocks=W0,
    )

    return hierarchy
