# lorenz_model.py
"""
Lorenz-specific RGM-style model definitions.

This module:
1. Defines Lorenz-level and Lorenz-hierarchy dataclasses that mirror the
   structure we will later generalize to RGMLevel / RGMModel.
2. Constructs a Lorenz hierarchy from:
   - lowest-level patch states and SVD parameters (A-like observation model),
   - spatial renormalization hierarchy (D mappings).
3. Initializes simple temporal dynamics B for states and (optionally) paths.

The goal is to have a concrete, JAX-friendly Lorenz model that can be used
for variational message passing, expected free energy computation, and
Dirichlet learning, in a way that will transfer directly to the full RGM.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from .lorenz_data import build_lorenz_patch_dataset
from .lorenz_renorm import build_lorenz_spatial_hierarchy


# -----------------------------------------------------------------------------
# 1. Dataclasses for Lorenz RGM-style model
# -----------------------------------------------------------------------------

@dataclass
class LorenzLevel:
    """
    A single spatial level in the Lorenz hierarchy.

    Fields are chosen to be compatible with later RGMLevel design:
      - A: observation / child likelihoods (may be None for non-bottom levels)
      - B_states: transition model over state factor (s_t -> s_{t+1} | path)
      - B_paths: (optional) transition model over path factor (u_t -> u_{t+1})
      - D: parent-to-child mapping (deterministic or categorical)
      - E_states: prior over initial state at this level
      - E_paths: prior over initial path at this level (optional)
      - aA, aB_states, aB_paths, aD, aE_states, aE_paths: Dirichlet counts

    The index conventions:
      - For the lowest level:
          A has shape (num_patch_state, num_obs_dim)
          D is None.
      - For higher levels:
          A is None (children are states in the level below, represented by D)
          D has shape (num_parent_state, num_child_config), where we use
          a discrete index for each unique child configuration (e.g. 4-tuple).
    """
    # Likelihood / observation mapping for this level
    A: Optional[jnp.ndarray] = None       # shape depends on level
    aA: Optional[jnp.ndarray] = None

    # State transition model (potentially conditioned on paths)
    # For now we use a simple 2D B(s', s) without explicit path factor.
    B_states: Optional[jnp.ndarray] = None    # (S, S)
    aB_states: Optional[jnp.ndarray] = None   # same shape as B_states

    # Path transition model (placeholder; not used in minimal version)
    B_paths: Optional[jnp.ndarray] = None     # (U, U) or (U, U, ...) later
    aB_paths: Optional[jnp.ndarray] = None

    # Parent-to-child mapping
    # For lowest level this is None; for higher levels it encodes child config.
    D: Optional[jnp.ndarray] = None          # (S_parent, child_config_dim)
    aD: Optional[jnp.ndarray] = None         # Dirichlet over child patterns

    # Initial priors
    E_states: Optional[jnp.ndarray] = None   # (S,)
    aE_states: Optional[jnp.ndarray] = None

    E_paths: Optional[jnp.ndarray] = None    # (U,)
    aE_paths: Optional[jnp.ndarray] = None

    # Metadata
    num_states: int = 0
    num_paths: int = 0
    # For higher levels, we may want to know how many child configs there are.
    num_child_configs: int = 0


@dataclass
class LorenzHierarchy:
    """
    Lorenz-specific hierarchy, ready for inference and learning.

    Fields:
      - levels: list of LorenzLevel from lowest (patch) to highest spatial level.
      - T: number of time steps.
      - H_blocks, W_blocks: grid size at the lowest level.
      - patch_size, img_size: for reconstruction / visualization.
      - K, L: SVD components / quantization levels.
      - svd_mean, svd_basis: parameters for reconstructing patches/images.
      - states_grids: list of state grids per level (T, H_l, W_l).
    """
    levels: List[LorenzLevel]
    T: int
    H_blocks: int
    W_blocks: int
    patch_size: int
    img_size: int
    K: int
    L: int
    svd_mean: jnp.ndarray
    svd_basis: jnp.ndarray
    states_grids: List[jnp.ndarray]  # (T, H_l, W_l) per level


# -----------------------------------------------------------------------------
# 2. Helpers to build A and priors for the lowest level
# -----------------------------------------------------------------------------

def build_lowest_level_A(
    q_coeffs: jnp.ndarray,
    states: jnp.ndarray,
    L: int,
    K: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a simple A-like mapping for the lowest level, treating the
    quantized coefficient vector as the "observation" and the encoded
    patch state as the hidden state.

    For now, we use a Dirichlet-parameter representation where:
      - A[s, d] is the categorical probability of observation dimension d
        given state s. In a full implementation, we'd factor this into
        K separate categorical distributions per coefficient dimension,
        but here we flatten them for simplicity.

    Args:
        q_coeffs: (N, K) quantized coefficients in [0, L-1]
        states: (N,) encoded patch states in [0, L^K-1]
        L: number of quantization levels
        K: number of coefficient dimensions

    Returns:
        A: (S, O) array, where S = num_states, O = K * L
           Each coefficient dimension has L possible values; we encode each
           observed q_coeffs[n, k] as a one-hot over an O-dimensional vector.
        aA: same shape as A, initialized with count statistics
    """
    # Number of possible states (max code + 1)
    S = int(states.max()) + 1
    # Observations: flatten K dims with L levels each into O = K*L
    O = K * L

    N = states.shape[0]

    # Build one-hot observation vectors
    # For sample n and dimension k, index in [0, O) is k*L + q_coeffs[n, k]
    indices = (
        jnp.arange(K, dtype=jnp.int32)[None, :] * L + q_coeffs
    )  # (N, K)
    indices_flat = indices.reshape(N, K)

    # One-hot encode into (N, O)
    def one_hot_row(idx_row: jnp.ndarray) -> jnp.ndarray:
        # idx_row: (K,) indices in 0..O-1
        oh = jnp.zeros((O,), dtype=jnp.float32)
        # Add 1 to each index position
        def body_fun(i, arr):
            return arr.at[idx_row[i]].add(1.0)
        oh = jax.lax.fori_loop(0, K, body_fun, oh)
        # Normalize to sum 1 (so each sample is a distribution over O)
        return oh / (oh.sum() + 1e-8)

    obs_one_hot = jax.vmap(one_hot_row)(indices_flat)  # (N, O)

    # Initialize Dirichlet counts aA with small pseudocounts and data counts
    aA = jnp.ones((S, O), dtype=jnp.float32) * 0.1

    # Accumulate counts: for each sample n, add obs_one_hot[n] to row states[n]
    def body_fun(n, aA_):
        s = states[n]
        return aA_.at[s].add(obs_one_hot[n])

    aA = jax.lax.fori_loop(0, N, body_fun, aA)
    # Compute A as normalized aA
    A = aA / (aA.sum(axis=1, keepdims=True) + 1e-8)

    return A, aA


def build_initial_state_prior(num_states: int,
                              alpha: float = 1.0
                              ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a symmetric Dirichlet prior over initial states.

    Args:
        num_states: number of hidden states
        alpha: symmetric Dirichlet concentration

    Returns:
        E: (S,) prior over initial state (normalized)
        aE: (S,) Dirichlet parameters
    """
    aE = jnp.ones((num_states,), dtype=jnp.float32) * alpha
    E = aE / aE.sum()
    return E, aE


def build_simple_B_states(num_states: int,
                          alpha: float = 1.0
                          ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a simple, near-uniform state transition model B(s'|s).

    Args:
        num_states: number of states S
        alpha: symmetric Dirichlet concentration for each column.

    Returns:
        B: (S, S) transition probabilities
        aB: (S, S) Dirichlet counts
    """
    aB = jnp.ones((num_states, num_states), dtype=jnp.float32) * alpha
    B = aB / (aB.sum(axis=0, keepdims=True) + 1e-8)
    return B, aB


# -----------------------------------------------------------------------------
# 3. Building LorenzLevel objects from spatial hierarchy
# -----------------------------------------------------------------------------

def build_lorenz_levels_from_hierarchy(
    lorenz_data_dict: Dict[str, Any],
    spatial_hierarchy: Dict[str, Any],
) -> Tuple[List[LorenzLevel], List[jnp.ndarray]]:
    """
    Construct LorenzLevel objects from:
      - lowest-level patch data (A-like mapping and states),
      - spatial hierarchy (D matrices and state grids).

    We create:
      - Level 0: patch level, with A and simple B/E
      - Level k>0: parent levels, with D, simple B/E, and A=None.

    Args:
        lorenz_data_dict: output of build_lorenz_patch_dataset
        spatial_hierarchy: output of build_lorenz_spatial_hierarchy

    Returns:
        levels: list of LorenzLevel from lowest to highest
        states_grids: list of (T, H_l, W_l) state grids per level
    """
    T = int(lorenz_data_dict["T"])
    H0 = int(lorenz_data_dict["H_blocks"])
    W0 = int(lorenz_data_dict["W_blocks"])
    K = int(lorenz_data_dict["K"])
    L = int(lorenz_data_dict["L"])

    states_flat = lorenz_data_dict["states"]   # (T*H0*W0,)
    q_coeffs = lorenz_data_dict["q_coeffs"]    # (T*H0*W0, K)

    # Build A and Dirichlet aA for level 0
    A0, aA0 = build_lowest_level_A(q_coeffs, states_flat, L=L, K=K)
    S0 = A0.shape[0]

    # Simple prior and transition at level 0
    E0, aE0 = build_initial_state_prior(S0, alpha=1.0)
    B0, aB0 = build_simple_B_states(S0, alpha=1.0)

    # Level 0 states_grid from spatial_hierarchy
    levels_info = spatial_hierarchy["levels"]
    states_grid_0 = levels_info[0]["states_grid"]  # (T, H0, W0)

    level0 = LorenzLevel(
        A=A0,
        aA=aA0,
        B_states=B0,
        aB_states=aB0,
        B_paths=None,
        aB_paths=None,
        D=None,
        aD=None,
        E_states=E0,
        aE_states=aE0,
        E_paths=None,
        aE_paths=None,
        num_states=S0,
        num_paths=0,
        num_child_configs=0,
    )

    levels: List[LorenzLevel] = [level0]
    states_grids: List[jnp.ndarray] = [states_grid_0]

    # Higher levels from spatial hierarchy
    # Each level_info[k] contains:
    #   'states_grid': (T, H_k, W_k)
    #   'D': (num_parent_states_k, 4) or None for level 0
    n_hierarchy_levels = len(levels_info)

    # Note: levels_info[0] is level 0 (patches); we already handled that.
    for k in range(1, n_hierarchy_levels):
        info_k = levels_info[k]
        states_grid_k = info_k["states_grid"]  # (T, H_k, W_k)
        D_k = info_k["D"]                      # (S_k, 4)
        Hk, Wk = info_k["group_shape"]
        assert states_grid_k.shape[1] == Hk and states_grid_k.shape[2] == Wk

        Sk = int(D_k.shape[0])  # number of parent states at this level
        # For now, child_config_dim = 4 (2x2 pattern) but we keep it generic.
        child_config_dim = int(D_k.shape[1])

        # Dirichlet prior over D_k (deterministic mapping): we treat each parent
        # as having a categorical distribution over child configurations, with
        # all mass on the observed pattern. Here, we initialize aD with
        # small pseudocounts; learning will refine it.
        aDk = jnp.ones_like(D_k, dtype=jnp.float32) * 0.1

        # Simple symmetric prior E and transitions B at this level
        Ek, aEk = build_initial_state_prior(Sk, alpha=1.0)
        Bk, aBk = build_simple_B_states(Sk, alpha=1.0)

        levelk = LorenzLevel(
            A=None,
            aA=None,
            B_states=Bk,
            aB_states=aBk,
            B_paths=None,
            aB_paths=None,
            D=D_k,
            aD=aDk,
            E_states=Ek,
            aE_states=aEk,
            E_paths=None,
            aE_paths=None,
            num_states=Sk,
            num_paths=0,
            num_child_configs=child_config_dim,
        )
        levels.append(levelk)
        states_grids.append(states_grid_k)

    return levels, states_grids


# -----------------------------------------------------------------------------
# 4. High-level constructor for LorenzHierarchy
# -----------------------------------------------------------------------------

def build_lorenz_hierarchy(
    T: int = 1000,
    dt: float = 0.01,
    img_size: int = 64,
    patch_size: int = 4,
    K: int = 4,
    L: int = 7,
    thickness: int = 1,
    num_spatial_levels: int = 1,
) -> LorenzHierarchy:
    """
    Build a full LorenzHierarchy object:
      - simulate Lorenz
      - discretize images into patch states (lorenz_data)
      - build spatial hierarchy (lorenz_renorm)
      - construct LorenzLevel objects with A, B, D, E, and Dirichlet params.

    Args:
        T: number of time steps
        dt: time step
        img_size: image size
        patch_size: patch size
        K: SVD rank
        L: quantization levels
        thickness: line thickness in rendering
        num_spatial_levels: number of RG levels above lowest level

    Returns:
        LorenzHierarchy instance
    """
    # Step 1: build patch dataset
    lorenz_data_dict = build_lorenz_patch_dataset(
        T=T,
        dt=dt,
        img_size=img_size,
        patch_size=patch_size,
        K=K,
        L=L,
        thickness=thickness,
    )

    # Step 2: build spatial hierarchy
    spatial_hierarchy = build_lorenz_spatial_hierarchy(
        lorenz_data_dict,
        num_levels=num_spatial_levels,
    )

    # Step 3: build LorenzLevel objects and state grids
    levels, states_grids = build_lorenz_levels_from_hierarchy(
        lorenz_data_dict,
        spatial_hierarchy,
    )

    # Compose LorenzHierarchy
    hierarchy = LorenzHierarchy(
        levels=levels,
        T=int(lorenz_data_dict["T"]),
        H_blocks=int(lorenz_data_dict["H_blocks"]),
        W_blocks=int(lorenz_data_dict["W_blocks"]),
        patch_size=int(lorenz_data_dict["patch_size"]),
        img_size=int(lorenz_data_dict["img_size"]),
        K=int(lorenz_data_dict["K"]),
        L=int(lorenz_data_dict["L"]),
        svd_mean=lorenz_data_dict["mean"],
        svd_basis=lorenz_data_dict["basis"],
        states_grids=states_grids,
    )
    return hierarchy
