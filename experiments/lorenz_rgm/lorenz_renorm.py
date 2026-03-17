# lorenz_renorm.py
"""
Spatial renormalisation (RG) for the Lorenz RGM example.

This module:
1. Takes lowest-level patch states on a 2D grid over time.
2. Groups non-overlapping 2x2 blocks of child sites into parent "groups".
3. Assigns a unique parent state to each distinct 2x2 child pattern
   within each group-site.
4. Builds a D tensor encoding the deterministic mapping from parent
   states to child configurations, with:
   - no shared children between different group-sites,
   - unique child patterns (rows) for each parent state.
5. Supports recursive application to build an arbitrary number of
   spatial levels via repeated 2x2 RG steps.
6. Provides a consistency check to verify that D and the parent states
   are consistent with the original child patterns.

All functions are written in a level-agnostic way so they can be used
for 1, 2, or more RG steps without renaming or re-structuring code.

NOTE:
- The symbol T here denotes the lowest-level temporal horizon (T0) used
  in LorenzHierarchy.
"""

from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import jax.numpy as jnp

# -----------------------------------------------------------------------------
# 1. Utilities: reshape between flat and grid states
# -----------------------------------------------------------------------------

def states_flat_to_grid(
    states_flat: jnp.ndarray,
    T: int,
    H_blocks: int,
    W_blocks: int,
) -> jnp.ndarray:
    """
    Reshape flat patch states into a 3D grid over time.

    Args:
      states_flat: (T * H_blocks * W_blocks,) integer states
      T: number of time steps (T0 at lowest level)
      H_blocks: number of patch rows at level 0
      W_blocks: number of patch columns at level 0

    Returns:
      states_grid: (T, H_blocks, W_blocks) integer states
    """
    return states_flat.reshape(T, H_blocks, W_blocks)


def states_grid_to_flat(states_grid: jnp.ndarray) -> jnp.ndarray:
    """
    Flatten (T, H, W) states back to a 1D array.

    Args:
      states_grid: (T, H, W)

    Returns:
      states_flat: (T * H * W,)
    """
    T, H, W = states_grid.shape
    return states_grid.reshape(T * H * W,)


# -----------------------------------------------------------------------------
# 2. Single RG step over space (2x2 blocks) with explicit mapping
# -----------------------------------------------------------------------------

def _build_parent_mapping_for_group_explicit(
    child_patterns: np.ndarray,
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int]]:
    """
    Build mapping from child 4-tuples to local parent indices for a
    single group-site, scanning over time.

    Args:
      child_patterns: (T, 4) integer child states for one group-site
        across all time points.

    Returns:
      parent_ids: (T,) array of local parent state indices (int64)
      pattern_to_parent: dict mapping pattern tuples -> local index
    """
    T = child_patterns.shape[0]
    pattern_to_parent: Dict[Tuple[int, ...], int] = {}
    parent_ids_list: List[int] = []

    next_idx = 0
    for t in range(T):
        pattern = tuple(child_patterns[t].tolist())  # (4,)
        if pattern in pattern_to_parent:
            idx = pattern_to_parent[pattern]
        else:
            idx = next_idx
            pattern_to_parent[pattern] = idx
            next_idx += 1
        parent_ids_list.append(idx)

    parent_ids = np.asarray(parent_ids_list, dtype=np.int64)  # (T,)
    return parent_ids, pattern_to_parent


def rg_step_level(states_grid: jnp.ndarray) -> Dict[str, Any]:
    """
    Perform a single spatial RG step over a grid of states.

    We:
      - assume H and W are even (non-overlapping 2x2 groups),
      - group 2x2 blocks into "group-sites" with four child positions,
      - for each group-site (h2,w2), build a local mapping from unique
        child patterns to parent states, ensuring:
          * no shared children between different group-sites (by construction),
          * unique rows in D for each parent (one row per pattern).

    Args:
      states_grid: (T, H, W) integer states at current level.

    Returns:
      dict with:
        'parent_states_grid': (T, H2, W2) integer parent states
        'group_pattern_maps': list (length H2 * W2) of dicts
            pattern_tuple (4,) -> parent_state_idx (global int)
        'D': (num_parent_states_total, 4) child state patterns
        'group_shape': (H2, W2)
        'num_parent_states_total': total number of parent states
    """
    T, H, W = states_grid.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even for 2x2 grouping."

    H2, W2 = H // 2, W // 2

    # Extract 2x2 child patterns per group-site and time:
    c00 = states_grid[:, 0::2, 0::2]
    c01 = states_grid[:, 0::2, 1::2]
    c10 = states_grid[:, 1::2, 0::2]
    c11 = states_grid[:, 1::2, 1::2]

    combo = jnp.stack([c00, c01, c10, c11], axis=-1)  # (T, H2, W2, 4)
    combo_np = np.array(combo, dtype=np.int32)

    group_pattern_maps: List[Dict[Tuple[int, ...], int]] = []
    parent_states_grid_np = np.zeros((T, H2, W2), dtype=np.int32)

    D_rows: List[Tuple[int, ...]] = []
    global_parent_index = 0

    for h2 in range(H2):
        for w2 in range(W2):
            # All timepoints for this group-site: (T, 4)
            patterns_hw = combo_np[:, h2, w2, :]  # (T, 4)

            parent_ids_local, pattern_to_parent_local = \
                _build_parent_mapping_for_group_explicit(patterns_hw)

            # Assign global parent indices for each local parent
            local_to_global: Dict[int, int] = {}
            for pattern, local_idx in pattern_to_parent_local.items():
                local_idx_int = int(local_idx)
                global_idx = global_parent_index
                global_parent_index += 1
                local_to_global[local_idx_int] = global_idx
                D_rows.append(pattern)  # row for this global parent

            # Map local parent ids to global parent ids over time
            parent_ids_global = np.array(
                [local_to_global[int(loc_idx)] for loc_idx in parent_ids_local],
                dtype=np.int32,
            )

            parent_states_grid_np[:, h2, w2] = parent_ids_global

            # Store mapping for this group-site (global indices)
            pattern_to_global = {
                pattern: local_to_global[int(local_idx)]
                for pattern, local_idx in pattern_to_parent_local.items()
            }
            group_pattern_maps.append(pattern_to_global)

    parent_states_grid = jnp.array(parent_states_grid_np, dtype=jnp.int32)
    D = jnp.array(D_rows, dtype=jnp.int32)  # (num_parent_states_total, 4)
    num_parent_states_total = int(D.shape[0])

    return {
        "parent_states_grid": parent_states_grid,
        "group_pattern_maps": group_pattern_maps,
        "D": D,
        "group_shape": (H2, W2),
        "num_parent_states_total": num_parent_states_total,
    }


# -----------------------------------------------------------------------------
# 3. Recursive RG across multiple spatial levels (level-agnostic)
# -----------------------------------------------------------------------------

def build_spatial_hierarchy(
    states_flat: jnp.ndarray,
    T: int,
    H_blocks: int,
    W_blocks: int,
    num_levels: int = 1,
) -> Dict[str, Any]:
    """
    Build a spatial hierarchy via repeated 2x2 RG steps, starting from
    lowest-level patch states.

    Args:
      states_flat: (T * H_blocks * W_blocks,) integer states at level 0
      T: number of time steps (T0)
      H_blocks: number of patch rows at level 0
      W_blocks: number of patch columns at level 0
      num_levels: number of RG steps to apply:
        - num_levels = 0: only level 0 (patch level)
        - num_levels = 1: levels 0 and 1 (one RG step)
        - num_levels = 2: levels 0,1,2 (two RG steps), etc.

    Returns:
      dict with:
        'levels': list of level dicts, l = 0..L
          level[0]:
            {'states_grid': (T,H0,W0),
             'group_shape': (H0,W0),
             'D': None,
             'group_pattern_maps': None}
          level[l>0]:
            {'states_grid': (T,H_l,W_l),
             'group_shape': (H_l,W_l),
             'D': (S_l,4),
             'group_pattern_maps': list of dicts}
        'T': T
    """
    assert num_levels >= 0, "num_levels must be >= 0."

    # Level 0: reshape lowest-level patch states to grid
    states_grid_0 = states_flat_to_grid(states_flat, T, H_blocks, W_blocks)

    levels: List[Dict[str, Any]] = []
    levels.append(
        {
            "states_grid": states_grid_0,
            "group_shape": (H_blocks, W_blocks),
            "D": None,
            "group_pattern_maps": None,
        }
    )

    current_states_grid = states_grid_0
    current_H, current_W = H_blocks, W_blocks

    # Apply num_levels RG steps
    for level_idx in range(1, num_levels + 1):
        if current_H % 2 != 0 or current_W % 2 != 0:
            raise ValueError(
                f"Cannot build level {level_idx}: "
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
# 4. Convenience wrapper: from lorenz_data output
# -----------------------------------------------------------------------------

def build_lorenz_spatial_hierarchy(
    lorenz_data_dict: Dict[str, Any],
    num_levels: int = 1,
) -> Dict[str, Any]:
    """
    Given the output of lorenz_data.build_lorenz_patch_dataset, build the
    spatial RG hierarchy via repeated 2x2 steps.

    Args:
      lorenz_data_dict: dict returned by build_lorenz_patch_dataset
        (must contain 'states', 'T', 'H_blocks', 'W_blocks')
      num_levels: number of RG steps above the patch level to build

    Returns:
      dict with:
        'levels': list of level dicts (see build_spatial_hierarchy)
        'T': T (T0)
      plus copies of useful fields from lorenz_data_dict:
        'H_blocks', 'W_blocks', 'K', 'L', 'patch_size', 'img_size'
    """
    states_flat = lorenz_data_dict["states"]
    T = int(lorenz_data_dict["T"])
    H_blocks = int(lorenz_data_dict["H_blocks"])
    W_blocks = int(lorenz_data_dict["W_blocks"])

    hierarchy = build_spatial_hierarchy(
        states_flat,
        T,
        H_blocks,
        W_blocks,
        num_levels=num_levels,
    )

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


# -----------------------------------------------------------------------------
# 5. Consistency check: verify D vs child patterns at level 1
# -----------------------------------------------------------------------------

def check_spatial_hierarchy_consistency(
    lorenz_data_dict: Dict[str, Any],
    spatial_hierarchy: Dict[str, Any],
    num_samples: int = 10,
    rng: Optional[np.random.RandomState] = None,
) -> bool:
    """
    Verify that (at least) the first RG level is consistent:

      - For randomly sampled times and group-sites at level 1,
        the D row corresponding to the parent state matches exactly
        the 4 child states at level 0.

    Args:
      lorenz_data_dict: original data dict (for level-0 states)
      spatial_hierarchy: output of build_lorenz_spatial_hierarchy
      num_samples: number of (t, h1, w1) samples to check
      rng: optional numpy RandomState for reproducibility

    Returns:
      True if all sampled checks pass; raises AssertionError otherwise.
    """
    if rng is None:
        rng = np.random.RandomState(0)

    # Level 0 states grid
    T = int(lorenz_data_dict["T"])
    H0 = int(lorenz_data_dict["H_blocks"])
    W0 = int(lorenz_data_dict["W_blocks"])
    states_flat = lorenz_data_dict["states"]
    states_grid_0 = states_flat_to_grid(states_flat, T, H0, W0)

    levels = spatial_hierarchy["levels"]
    assert len(levels) >= 2, "Need at least one RG level above patches to check."

    level1 = levels[1]
    states_grid_1 = np.array(level1["states_grid"])  # (T, H1, W1)
    D = np.array(level1["D"])  # (num_parent_states_1, 4)

    T1, H1, W1 = states_grid_1.shape
    assert T1 == T
    assert H0 == 2 * H1 and W0 == 2 * W1

    # Sample random (t, h1, w1) triples
    for _ in range(num_samples):
        t = rng.randint(0, T)
        h1 = rng.randint(0, H1)
        w1 = rng.randint(0, W1)

        parent_state = int(states_grid_1[t, h1, w1])  # index into D
        pattern_D = D[parent_state]  # (4,)

        # Extract child states from level 0 at this group-site and time
        h0_0, h0_1 = 2 * h1, 2 * h1 + 1
        w0_0, w0_1 = 2 * w1, 2 * w1 + 1

        s00 = int(states_grid_0[t, h0_0, w0_0])
        s01 = int(states_grid_0[t, h0_0, w0_1])
        s10 = int(states_grid_0[t, h0_1, w0_0])
        s11 = int(states_grid_0[t, h0_1, w0_1])

        pattern_child = np.array([s00, s01, s10, s11], dtype=np.int32)

        if not np.array_equal(pattern_D, pattern_child):
            raise AssertionError(
                f"Inconsistent D at t={t}, h1={h1}, w1={w1}: "
                f"D row {pattern_D} vs children {pattern_child}"
            )

    return True
