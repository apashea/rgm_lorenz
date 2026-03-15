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
        aA: same shape
