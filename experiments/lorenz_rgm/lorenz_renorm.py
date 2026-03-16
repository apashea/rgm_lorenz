# lorenz_renorm.py
"""
Renormalisation (RG) utilities for the Lorenz RGM.

This module constructs a spatial hierarchy of discrete states from
lowest-level patch states by grouping 2x2 blocks of child sites into
parent sites. It produces:

- A list of levels, each with:
    * 'states_grid': (T, H_l, W_l) integer state labels
    * 'D': (S_l, 4) deterministic mapping from each parent state to the
           states of its 4 children (for l > 0)

- Top-level metadata:
    * 'H_blocks', 'W_blocks': lowest-level patch grid size
    * 'T': number of time steps

CURRENT STATUS (to be extended):

- We support one RG step (num_levels = 1) by default:
    level 0: patch-level states
    level 1: first parent-level states (2x2 blocks of level 0)
- D is a fixed, deterministic RG operator built from observed child
  configurations; no structure learning is applied yet.
- Future work will:
    - add a second RG step (level 2) to obtain a 3-level hierarchy, and
    - introduce structure learning updates for D based on state
      co-occurrences across levels.
"""

from typing import Dict, Any, List, Tuple

import numpy as np
import jax.numpy as jnp


def build_level_from_children(
    child_states_grid: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Given a child level with states_grid of shape (T, H_child, W_child),
    build the next parent level by grouping each 2x2 block of child sites
    into a single parent site.

    This function:
    - Enumerates all unique 2x2 child state configurations observed over
      time and space.
    - Assigns each unique configuration an integer parent state index s_parent.
    - Constructs:
        * parent_states_grid: (T, H_parent, W_parent)
        * D_parent: (S_parent, 4) mapping parent state -> 4 child states

    Args:
        child_states_grid: (T, H_child, W_child) integer child states

    Returns:
        parent_states_grid: (T, H_parent, W_parent) integer parent states
        D_parent: (S_parent, 4) array of child state patterns
    """
    T, Hc, Wc = child_states_grid.shape
    assert Hc % 2 == 0 and Wc % 2 == 0, "Child grid must have even dimensions."

    Hp = Hc // 2
    Wp = Wc // 2

    # Collect all 2x2 configurations over time and space
    configs = []
    for t in range(T):
        for h in range(Hp):
            for w in range(Wp):
                c00 = int(child_states_grid[t, 2 * h, 2 * w])
                c01 = int(child_states_grid[t, 2 * h, 2 * w + 1])
                c10 = int(child_states_grid[t, 2 * h + 1, 2 * w])
                c11 = int(child_states_grid[t, 2 * h + 1, 2 * w + 1])
                configs.append((c00, c01, c10, c11))

    # Unique configurations -> parent state indices
    unique_configs = sorted(set(configs))
    pattern_to_parent = {cfg: idx for idx, cfg in enumerate(unique_configs)}
    S_parent = len(unique_configs)

    # Build D_parent mapping
    D_parent = np.zeros((S_parent, 4), dtype=np.int32)
    for s_parent, cfg in enumerate(unique_configs):
        c00, c01, c10, c11 = cfg
        D_parent[s_parent, 0] = c00
        D_parent[s_parent, 1] = c01
        D_parent[s_parent, 2] = c10
        D_parent[s_parent, 3] = c11
    D_parent = jnp.array(D_parent)

    # Build parent_states_grid
    parent_states_grid = np.zeros((T, Hp, Wp), dtype=np.int32)
    for t in range(T):
        for h in range(Hp):
            for w in range(Wp):
                c00 = int(child_states_grid[t, 2 * h, 2 * w])
                c01 = int(child_states_grid[t, 2 * h, 2 * w + 1])
                c10 = int(child_states_grid[t, 2 * h + 1, 2 * w])
                c11 = int(child_states_grid[t, 2 * h + 1, 2 * w + 1])
                cfg = (c00, c01, c10, c11)
                parent_states_grid[t, h, w] = pattern_to_parent[cfg]

    parent_states_grid = jnp.array(parent_states_grid)

    return parent_states_grid, D_parent


def build_lorenz_spatial_hierarchy(
    lorenz_data_dict: Dict[str, Any],
    num_levels: int = 1,
) -> Dict[str, Any]:
    """
    Build a spatial RG hierarchy of discrete states for the Lorenz data.

    Starting from lowest-level patch states, we apply num_levels steps of
    2x2 grouping to obtain coarser spatial levels.

    Args:
        lorenz_data_dict: output from build_lorenz_patch_dataset, containing:
            - 'states': (T * H_blocks * W_blocks,) lowest-level state labels
            - 'T': number of time steps
            - 'H_blocks', 'W_blocks': patch grid dimensions
        num_levels: number of RG steps to apply (currently 1 by default):
            num_levels = 0: only level 0 (patch level)
            num_levels = 1: levels 0 and 1 (one RG step)

    Returns:
        spatial_hierarchy: dict with keys:
            - 'levels': list of dicts, level 0..L:
                * level[0]: {'states_grid': (T,H0,W0)}
                * level[1]: {'states_grid': (T,H1,W1), 'D': (S1,4)}
                * (future) level[2]: {'states_grid': (T,H2,W2), 'D': (S2,4)}, ...
            - 'H_blocks', 'W_blocks': lowest-level H0, W0
            - 'T': number of time steps
    """
    T = int(lorenz_data_dict["T"])
    H0 = int(lorenz_data_dict["H_blocks"])
    W0 = int(lorenz_data_dict["W_blocks"])

    states_flat = lorenz_data_dict["states"]  # (T * H0 * W0,)
    states_grid0 = jnp.array(
        states_flat.reshape(T, H0, W0)
    )  # (T, H0, W0)

    levels: List[Dict[str, Any]] = []
    levels.append({"states_grid": states_grid0})

    current_states_grid = states_grid0

    # Apply num_levels RG steps
    for _ in range(num_levels):
        parent_states_grid, D_parent = build_level_from_children(current_states_grid)
        levels.append(
            {
                "states_grid": parent_states_grid,
                "D": D_parent,
            }
        )
        current_states_grid = parent_states_grid

    spatial_hierarchy: Dict[str, Any] = {
        "levels": levels,
        "H_blocks": H0,
        "W_blocks": W0,
        "T": T,
    }
    return spatial_hierarchy
