# lorenz_renorm.py
"""
Spatial renormalization (RG) for the Lorenz RGM example.

This module:
1. Takes lowest-level patch states on a 2D grid over time.
2. Groups non-overlapping 2x2 blocks of patches into parent "groups".
3. Assigns a unique parent state to each distinct child pattern within a group.
4. Builds a D tensor encoding the deterministic mapping from parent states
   to child configurations, with:
     - no shared children between different groups
     - unique child patterns (columns) for each parent state
5. Supports recursive application to build multiple spatial levels.

This is the spatial component of the renormalizing generative model (RGM)
for the Lorenz image example.
"""

from typing import Dict, Any, Tuple, List

import numpy as np
import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# 1. Utility: reshape flat patch states into (T, H_blocks, W_blocks)
# -----------------------------------------------------------------------------

def states_flat_to_grid(states_flat: jnp.ndarray,
                        T: int,
                        H_blocks: int,
                        W_blocks: int
                        ) -> jnp.ndarray:
    """
    Reshape flat patch states into a 3D grid over time.

    Args:
        states_flat: (T * H_blocks * W_blocks,) integer states
        T: number of time steps
        H_blocks: number of patch rows
        W_blocks: number of patch columns

    Returns:
        states_grid: (T, H_blocks, W_blocks) integer states
    """
    return states_flat.reshape(T, H_blocks, W_blocks)


def states_grid_to_flat(states_grid: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten (T, H_blocks, W_blocks) states back to a 1D array.

    Args:
        states_grid: (T, H_blocks, W_blocks)

    Returns:
        states_flat: (T * H_blocks * W_blocks,)
    """
    T, H, W = states_grid.shape
    return states_grid.reshape(T * H * W,)


# -----------------------------------------------------------------------------
# 2. Single RG step over space (2x2 groups)
# -----------------------------------------------------------------------------

def _build_parent_mapping_for_group(
    child_patterns: np.ndarray
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int]]:
    """
    Build a mapping from unique child 4-tuples to parent state indices
    for a single group-location identity.

    Args:
        child_patterns: (N, 4) array of integer child states,
                        where N = number of time points * number of group-sites.

    Returns:
        parent_ids: (N,) array of parent state indices
        pattern_to_parent: dict mapping pattern tuples to parent indices
    """
    # Each row is a pattern of 4 child states
    # We assign a unique parent index per distinct pattern.
    patterns_as_tuples = [tuple(row.tolist()) for row in child_patterns]
    unique_patterns, inverse = np.unique(patterns_as_tuples,
                                         return_inverse=True)
    # unique_patterns is array of tuples; indices 0..(n_unique-1)
    parent_ids = inverse.astype(np.int32)
    pattern_to_parent = {
        pattern: idx for idx, pattern in enumerate(unique_patterns)
    }
    return parent_ids, pattern_to_parent


def rg_step_level(
    states_grid: jnp.ndarray,
) -> Dict[str, Any]:
    """
    Perform a single spatial RG step over a grid of patch states.

    We:
      - assume H and W are even (non-overlapping 2x2 groups),
      - group 2x2 blocks into "groups" with four child positions,
      - for each group identity (top-left coordinate), build a local
        mapping from unique child patterns to parent states, ensuring:
          * no shared children between different groups (by construction),
          * unique columns in D for each parent.

    Args:
        states_grid: (T, H, W) integer states at current level.

    Returns:
        A dict with:
          'parent_states_grid': (T, H2, W2) integer parent states
          'group_pattern_maps': list of length (H2 * W2), each entry is a dict:
                               { pattern_tuple (4,) -> parent_state_idx (int) }
          'D': (num_parent_states_total, 4) array of child state patterns
               for each global parent state index
          'group_shape': (H2, W2)
          'num_parent_states_total': total number of parent states across groups
    """
    T, H, W = states_grid.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for 2x2 grouping."

    H2, W2 = H // 2, W // 2

    # Extract 2x2 child patterns per group-site and time:
    # indices:
    #  (0,0): [0::2, 0::2]
    #  (0,1): [0::2, 1::2]
    #  (1,0): [1::2, 0::2]
    #  (1,1): [1::2, 1::2]
    c00 = states_grid[:, 0::2, 0::2]
    c01 = states_grid[:, 0::2, 1::2]
    c10 = states_grid[:, 1::2, 0::2]
    c11 = states_grid[:, 1::2, 1::2]

    # Stack into (T, H2, W2, 4)
    combo = jnp.stack([c00, c01, c10, c11], axis=-1)

    # We'll build parent mapping per group-site using NumPy on host.
    combo_np = np.array(combo, dtype=np.int32)  # (T, H2, W2, 4)

    # For each group-site (h2, w2), gather all timepoints' child patterns,
    # and map unique patterns -> local parent indices.
    group_pattern_maps: List[Dict[Tuple[int, ...], int]] = []
    parent_states_grid_np = np.zeros((T, H2, W2), dtype=np.int32)

    # We'll also collect all (global) D rows: each parent corresponds to one
    # unique 4-tuple of child states, but local indices per group-site
    # must be offset to get global indices.
    D_rows: List[Tuple[int, ...]] = []
    global_parent_index = 0

    for h2 in range(H2):
        for w2 in range(W2):
            # For this group-site, get all timepoints: shape (T, 4)
            patterns_hw = combo_np[:, h2, w2, :]  # (T, 4)

            parent_ids_local, pattern_to_parent_local = \
                _build_parent_mapping_for_group(patterns_hw)

            # Assign global parent indices for each local parent
            local_to_global = {}
            for pattern, local_idx in pattern_to_parent_local.items():
                global_idx = global_parent_index
                global_parent_index += 1
                local_to_global[local_idx] = global_idx
                D_rows.append(pattern)  # each pattern is the 4-child tuple

            # Fill parent_states_grid for this group-site, time by time
            parent_ids_global = np.array(
                [local_to_global[loc_idx] for loc_idx in parent_ids_local],
                dtype=np.int32
            )
            parent_states_grid_np[:, h2, w2] = parent_ids_global

            # Store mapping for this group-site (use global indices
            # so D_rows and parent_states_grid are aligned)
            pattern_to_global = {
                pattern: local_to_global[local_idx]
                for pattern, local_idx in pattern_to_parent_local.items()
            }
            group_pattern_maps.append(pattern_to_global)

    parent_states_grid = jnp.array(parent_states_grid_np, dtype=jnp.int32)
    D = jnp.array(D_rows, dtype=jnp.int32)  # (num_parent_states_total, 4)
    num_parent_states_total = D.shape[0]

    return {
        "parent_states_grid": parent_states_grid,  # (T, H2, W2)
        "group_pattern_maps": group_pattern_maps,  # length H2*W2
        "D": D,  # (num_parent_states_total, 4) child state patterns
        "group_shape": (H2, W2),
        "num_parent_states_total": int(num_parent_states_total),
    }


# -----------------------------------------------------------------------------
# 3. Recursive RG across multiple spatial levels
# -----------------------------------------------------------------------------

def build_spatial_hierarchy(
    states_flat: jnp.ndarray,
    T: int,
    H_blocks: int,
    W_blocks: int,
    num_levels: int = 1
) -> Dict[str, Any]:
    """
    Build a spatial hierarchy via repeated 2x2 RG steps, starting from
    lowest-level patch states.

    Args:
        states_flat: (T * H_blocks * W_blocks,) integer states at level 0
        T: number of time steps
        H_blocks: number of patch rows at level 0
        W_blocks: number of patch columns at level 0
        num_levels: number of RG levels to build (>= 1)

    Returns:
        A dict with:
          'levels': list of levels, each level is a dict with:
              - 'states_grid': (T, H_l, W_l) states at that level
              - 'D': (num_parent_states_l, 4) child patterns (only for levels > 0)
              - 'group_shape': (H_l, W_l)
              - 'group_pattern_maps': list of mappings (only for levels > 0)
          'T': T
    """
    assert num_levels >= 1, "At least one level (the base level) is required."

    # Level 0: reshape lowest-level patch states to grid
    states_grid_0 = states_flat_to_grid(states_flat, T, H_blocks, W_blocks)

    levels: List[Dict[str, Any]] = []

    # Store level 0
    levels.append({
        "states_grid": states_grid_0,
        "group_shape": (H_blocks, W_blocks),
        "D": None,
        "group_pattern_maps": None,
    })

    current_states_grid = states_grid_0
    current_H, current_W = H_blocks, W_blocks

    for level in range(1, num_levels + 1):
        if current_H % 2 != 0 or current_W % 2 != 0:
            raise ValueError(
                f"Cannot build level {level}: "
                f"current grid size ({current_H}, {current_W}) is not even."
            )

        rg_result = rg_step_level(current_states_grid)
        parent_states_grid = rg_result["parent_states_grid"]
        H2, W2 = rg_result["group_shape"]

        level_dict = {
            "states_grid": parent_states_grid,
            "group_shape": (H2, W2),
            "D": rg_result["D"],
            "group_pattern_maps": rg_result["group_pattern_maps"],
        }
        levels.append(level_dict)

        current_states_grid = parent_states_grid
        current_H, current_W = H2, W2

    return {
        "levels": levels,
        "T": T,
    }


# -----------------------------------------------------------------------------
# 4. Helper: build lowest-level + hierarchy from lorenz_data output
# -----------------------------------------------------------------------------

def build_lorenz_spatial_hierarchy(
    lorenz_data_dict: Dict[str, Any],
    num_levels: int = 1
) -> Dict[str, Any]:
    """
    Convenience wrapper: given the output of lorenz_data.build_lorenz_patch_dataset,
    build the spatial hierarchy via repeated 2x2 RG steps.

    Args:
        lorenz_data_dict: dict returned by build_lorenz_patch_dataset
            (must contain 'states', 'T', 'H_blocks', 'W_blocks')
        num_levels: number of RG levels above the patch level to build

    Returns:
        A dict with:
          'levels': list of level dicts (see build_spatial_hierarchy)
          'T': T
          plus copies of useful fields from lorenz_data_dict:
          'H_blocks', 'W_blocks', 'K', 'L', etc.
    """
    states_flat = lorenz_data_dict["states"]
    T = int(lorenz_data_dict["T"])
    H_blocks = int(lorenz_data_dict["H_blocks"])
    W_blocks = int(lorenz_data_dict["W_blocks"])

    hierarchy = build_spatial_hierarchy(states_flat, T, H_blocks, W_blocks,
                                        num_levels=num_levels)

    # Attach some metadata from lorenz_data
    result = {
        "T": hierarchy["T"],
        "levels": hierarchy["levels"],
        "H_blocks": H_blocks,
        "W_blocks": W_blocks,
        "K": int(lorenz_data_dict["K"]),
        "L": int(lorenz_data_dict["L"]),
        "patch_size": int(lorenz_data_dict["patch_size"]),
        "img_size": int(lorenz_data_dict["img_size"]),
    }

    return result
