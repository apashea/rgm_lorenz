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
      * Optional path factor (u_t) with its own B^u, E^u
  - LorenzHierarchy: a stack of LorenzLevel objects plus structural info
    (T, H_blocks, W_blocks, etc.)
  - Builders:
      * build_lorenz_hierarchy: construct the hierarchy given:
          - spatial hierarchy (from lorenz_renorm)
          - configuration parameters
      * Dirichlet parameter container LorenzRGMParams, which will be used
        in learning code to represent priors over A/B/E/C.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
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
        B_states: (S, S) transition matrix P(s' | s)
        E_states: (S,) prior over initial state s_1
        D: for spatial levels > 0:
           - (S, child_config_dim) deterministic mapping from parent state
             to child state pattern (e.g., 4 child indices for 2x2 groups)
           for level 0: D is None
        num_paths: number of paths (policies) at this level
        B_paths: (num_paths, num_paths) transition over path states u_t
        E_paths: (num_paths,) prior over u_1
    """
    S: int
    A: jnp.ndarray
    B_states: jnp.ndarray
    E_states: jnp.ndarray
    D: Optional[jnp.ndarray]
    num_paths: int
    B_paths: Optional[jnp.ndarray]
    E_paths: Optional[jnp.ndarray]


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

    This structure will be used by learning code (lorenz_learning.py) to
    represent and update priors over A/B/E/C at each level of the hierarchy.

    Attributes:
        A_alpha: list of Dirichlet concentrations over A^l:
                 A_alpha[l] has shape (S_l, O_l)
        B_alpha: list of Dirichlet concentrations over B_states^l:
                 B_alpha[l] has shape (S_l, S_l)
        E_alpha: list of Dirichlet concentrations over E_states^l:
                 E_alpha[l] has shape (S_l,)
        C_alpha: Dirichlet concentration over lowest-level outcomes C^0(o):
                 shape (O_0,)
    """
    A_alpha: List[jnp.ndarray]
    B_alpha: List[jnp.ndarray]
    E_alpha: List[jnp.ndarray]
    C_alpha: jnp.ndarray


# -----------------------------------------------------------------------------
# Helper functions: constructing A, B, E for the Lorenz example
# -----------------------------------------------------------------------------

def build_lorenz_A0(S0: int, K: int, L: int) -> jnp.ndarray:
    """
    Build the lowest-level emission matrix A0 for the Lorenz example.

    Here, hidden states encode K discrete coefficients, each taking L values.
    Outcomes are one-hot over K*L possible coefficient-value slots.

    For now we use a simple deterministic mapping:
      - each hidden state s corresponds to a unique K-tuple (c1,...,cK)
      - A0(s, o) = 1 if the one-hot outcome o is consistent with that tuple.

    This is a placeholder for a more general learned A0; learning code will
    update A_alpha and re-normalize to get A0.
    """
    O = K * L
    # Enumerate all K-tuples of {0,...,L-1}
    # s index is from 0 to S0-1, corresponding to base-L representation
    assert S0 == L ** K, "S0 must be L^K for this construction."

    # Precompute state -> coefficients
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
    # Normalize over outcomes (still deterministic, but ensures valid probs)
    A0 = A0 / (A0.sum(axis=1, keepdims=True) + 1e-8)
    return jnp.array(A0)


def build_uniform_B(S: int, self_bias: float = 1.0) -> jnp.ndarray:
    """
    Build a simple transition matrix B(s'|s) with optional self-transition bias.

    Args:
        S: number of states
        self_bias: additional weight on diagonal entries

    Returns:
        B: (S, S) matrix, rows normalized to 1.
    """
    B = np.ones((S, S), dtype=np.float32)
    np.fill_diagonal(B, 1.0 + self_bias)
    B = B / (B.sum(axis=1, keepdims=True) + 1e-8)
    return jnp.array(B)


def build_uniform_E(S: int) -> jnp.ndarray:
    """
    Build a uniform prior over initial states.

    Args:
        S: number of states

    Returns:
        E: (S,) vector with uniform probabilities.
    """
    E = np.ones((S,), dtype=np.float32)
    E = E / E.sum()
    return jnp.array(E)


def build_path_dynamics(num_paths: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a simple path factor dynamics (B_paths, E_paths) for the top level.

    Args:
        num_paths: number of path states u_t

    Returns:
        B_paths: (num_paths, num_paths) transition matrix over paths
        E_paths: (num_paths,) prior over initial path state
    """
    if num_paths <= 0:
        return None, None

    B_paths = np.ones((num_paths, num_paths), dtype=np.float32)
    np.fill_diagonal(B_paths, 2.0)
    B_paths = B_paths / (B_paths.sum(axis=1, keepdims=True) + 1e-8)

    E_paths = np.ones((num_paths,), dtype=np.float32)
    E_paths = E_paths / E_paths.sum()

    return jnp.array(B_paths), jnp.array(E_paths)


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
    Build the Lorenz RGM hierarchy.

    This function assumes that a spatial hierarchy has already been built
    using lorenz_renorm.build_lorenz_spatial_hierarchy, providing:
      - level-0 states_grid (T, H0, W0) for patches
      - level-1 states_grid (T, H1, W1) for spatially grouped parent states
      - D1 mapping for spatial level

    Args:
        T: number of time steps
        dt: time step (not used directly here, but useful for consistency)
        img_size: size of images (img_size x img_size)
        patch_size: size of patches (patch_size x patch_size)
        K: number of singular vectors / coefficients per patch
        L: number of discrete levels per coefficient
        thickness: not used here yet; placeholder for richer models
        num_spatial_levels: number of spatial RG levels (1 => level 0 and 1)
        num_paths_top: number of path states at the top level
        lorenz_spatial_hierarchy: dict returned by build_lorenz_spatial_hierarchy

    Returns:
        LorenzHierarchy instance with two levels (0: patches, 1: parents)
        and a path factor at the highest level.
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

    # Hidden state space at lowest level
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
    )
    levels.append(level0)

    # Spatial level(s)
    current_num_levels = len(lorenz_spatial_hierarchy["levels"])
    if current_num_levels < 2:
        raise ValueError(
            "Spatial hierarchy must have at least 2 levels (patch and parent)."
        )

    # For Lorenz example, we construct only one spatial parent level
    spatial_level1 = lorenz_spatial_hierarchy["levels"][1]
    states_grid1 = spatial_level1["states_grid"]
    D1 = spatial_level1["D"]  # (S1_total, 4)

    states_grids.append(states_grid1)

    S1 = int(D1.shape[0])

    # For now, we give level 1 a simple self-biased transition and uniform prior
    B1 = build_uniform_B(S1, self_bias=1.0)
    E1 = build_uniform_E(S1)

    # No emissions at level 1 in this minimal Lorenz example
    A1 = jnp.zeros((S1, 0), dtype=jnp.float32)

    level1 = LorenzLevel(
        S=S1,
        A=A1,
        B_states=B1,
        E_states=E1,
        D=D1,
        num_paths=num_paths_top,
        B_paths=None,
        E_paths=None,
    )
    levels.append(level1)

    # Path factor at top level
    if num_paths_top > 0:
        B_paths, E_paths = build_path_dynamics(num_paths_top)
        # Attach to the top level (level 1)
        levels[-1].B_paths = B_paths
        levels[-1].E_paths = E_paths

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
) -> LorenzRGMParams:
    """
    Initialize Dirichlet concentrations (A_alpha, B_alpha, E_alpha, C_alpha)
    given an existing LorenzHierarchy.

    This is the starting point for learning. Subsequent updates to these
    alphas will be performed by lorenz_learning.py, and A/B/E/C for inference
    will be reconstructed by normalizing the alphas.

    Args:
        hierarchy: existing LorenzHierarchy (with fixed A/B/E/D)
        K, L: used to determine O0 = K * L for lowest-level outcomes
        alpha_A, alpha_B, alpha_E, alpha_C: symmetric prior strengths

    Returns:
        LorenzRGMParams with initialized Dirichlet alphas.
    """
    A_alpha: List[jnp.ndarray] = []
    B_alpha: List[jnp.ndarray] = []
    E_alpha: List[jnp.ndarray] = []

    for level in hierarchy.levels:
        S = level.S
        O = level.A.shape[1]

        A_alpha_l = jnp.full((S, O), alpha_A, dtype=jnp.float32) if O > 0 else jnp.zeros((S, 0), dtype=jnp.float32)
        B_alpha_l = jnp.full((S, S), alpha_B, dtype=jnp.float32)
        E_alpha_l = jnp.full((S,), alpha_E, dtype=jnp.float32)

        A_alpha.append(A_alpha_l)
        B_alpha.append(B_alpha_l)
        E_alpha.append(E_alpha_l)

    O0 = K * L
    C_alpha = jnp.full((O0,), alpha_C, dtype=jnp.float32)

    return LorenzRGMParams(
        A_alpha=A_alpha,
        B_alpha=B_alpha,
        E_alpha=E_alpha,
        C_alpha=C_alpha,
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
    Rebuild a LorenzHierarchy using learned Dirichlet parameters for A/B/E
    at each level, plus the fixed spatial hierarchy and D matrices.

    Args:
        lorenz_spatial_hierarchy: dict from build_lorenz_spatial_hierarchy
        params: LorenzRGMParams with A_alpha, B_alpha, E_alpha, C_alpha
        num_paths_top: number of paths at the top level

    Returns:
        LorenzHierarchy with A/B/E normalized from alphas, D from
        lorenz_spatial_hierarchy, and path factor at top level.
    """
    # Normalize A, B, E from alphas
    A_list: List[jnp.ndarray] = []
    B_list: List[jnp.ndarray] = []
    E_list: List[jnp.ndarray] = []

    for A_alpha_l, B_alpha_l, E_alpha_l in zip(
        params.A_alpha, params.B_alpha, params.E_alpha
    ):
        # A
        if A_alpha_l.size > 0:
            A_l = A_alpha_l / (A_alpha_l.sum(axis=1, keepdims=True) + 1e-8)
        else:
            A_l = A_alpha_l
        # B
        B_l = B_alpha_l / (B_alpha_l.sum(axis=1, keepdims=True) + 1e-8)
        # E
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
    T = int(lorenz_spatial_hierarchy["T"]) if "T" in lorenz_spatial_hierarchy else level0_states_grid.shape[0]

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
    )
    levels.append(level0)

    # Level 1 (spatial parent)
    spatial_level1 = lorenz_spatial_hierarchy["levels"][1]
    states_grid1 = spatial_level1["states_grid"]
    D1 = spatial_level1["D"]  # (S1, 4)
    states_grids.append(states_grid1)

    A1 = A_list[1]
    B1 = B_list[1]
    E1 = E_list[1]
    S1 = A1.shape[0] if A1.shape[0] > 0 else B1.shape[0]

    level1 = LorenzLevel(
        S=S1,
        A=A1,
        B_states=B1,
        E_states=E1,
        D=D1,
        num_paths=num_paths_top,
        B_paths=None,
        E_paths=None,
    )
    levels.append(level1)

    if num_paths_top > 0:
        B_paths, E_paths = build_path_dynamics(num_paths_top)
        levels[-1].B_paths = B_paths
        levels[-1].E_paths = E_paths

    hierarchy = LorenzHierarchy(
        levels=levels,
        states_grids=states_grids,
        T=T,
        H_blocks=H0,
        W_blocks=W0,
    )
    return hierarchy
