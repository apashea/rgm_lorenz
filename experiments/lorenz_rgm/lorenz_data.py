# lorenz_data.py
"""
Data generation and preprocessing for the Lorenz RGM.

This module:
- Simulates the Lorenz system to produce a trajectory in R^3.
- Renders the trajectory into grayscale images over time.
- Partitions images into non-overlapping patches and performs SVD
  across patches to define a low-dimensional coefficient space.
- Quantizes these coefficients into discrete bins and encodes them as
  integer patch states for level 0 of the RGM.
- Provides metadata needed for building the spatial RG hierarchy.

IMPORTANT:
- Level-0 discrete states here are defined via SVD + quantization of
  patch intensities, not via direct clustering of raw pixel patterns.
  This is a tractable approximation to the pixel-pattern based RG
  described in the RGM paper and may be revised in future for closer
  fidelity.
"""

from typing import Dict, Any, Tuple

import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp


# -----------------------------------------------------------------------------
# 1. Lorenz simulation
# -----------------------------------------------------------------------------

def simulate_lorenz(
    T: int,
    dt: float,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: float = 1.0,
    y0: float = 1.0,
    z0: float = 1.0,
) -> np.ndarray:
    """
    Simulate the Lorenz system for T steps with step size dt.

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

    Args:
        T: number of time steps
        dt: step size
        sigma, rho, beta: Lorenz parameters
        x0, y0, z0: initial conditions

    Returns:
        traj: (T, 3) array of [x, y, z] over time
    """
    def lorenz_rhs(t, xyz):
        x, y, z = xyz
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t_span = (0.0, (T - 1) * dt)
    t_eval = np.linspace(t_span[0], t_span[1], T)
    sol = solve_ivp(lorenz_rhs, t_span, [x0, y0, z0], t_eval=t_eval, rtol=1e-6, atol=1e-9)
    traj = sol.y.T  # (T, 3)
    return traj


# -----------------------------------------------------------------------------
# 2. Rendering Lorenz trajectory to images
# -----------------------------------------------------------------------------

def render_lorenz_to_images(
    traj: np.ndarray,
    img_size: int,
    thickness: int = 1,
) -> np.ndarray:
    """
    Render a Lorenz trajectory into grayscale images.

    For each time t, we:
    - Compute a polar coordinate (r, theta) from (x, y).
    - Map theta to angle in [0, 2π) and r to radius in [0, 1].
    - Draw a thin line at the corresponding position in a 2D image.

    Args:
        traj: (T, 3) Lorenz trajectory
        img_size: image height and width
        thickness: integer controlling line thickness (dilation)

    Returns:
        images: (T, img_size, img_size) grayscale images in [0, 1]
    """
    T = traj.shape[0]
    images = np.zeros((T, img_size, img_size), dtype=np.float32)

    xs = traj[:, 0]
    ys = traj[:, 1]

    # Normalize to unit circle
    r = np.sqrt(xs**2 + ys**2)
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    theta = np.arctan2(ys, xs)  # [-π, π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2π)

    # Map polar coords to pixel coordinates
    center = img_size / 2.0
    max_radius = 0.45 * img_size

    for t in range(T):
        rr = r[t] * max_radius
        th = theta[t]
        x_t = center + rr * np.cos(th)
        y_t = center + rr * np.sin(th)
        ix = int(np.clip(np.round(x_t), 0, img_size - 1))
        iy = int(np.clip(np.round(y_t), 0, img_size - 1))
        images[t, iy, ix] = 1.0

        if thickness > 1:
            for dx in range(-thickness, thickness + 1):
                for dy in range(-thickness, thickness + 1):
                    jx = int(np.clip(ix + dx, 0, img_size - 1))
                    jy = int(np.clip(iy + dy, 0, img_size - 1))
                    images[t, jy, jx] = 1.0

    return images


# -----------------------------------------------------------------------------
# 3. Patch extraction and SVD over patches
# -----------------------------------------------------------------------------

def extract_patches(
    images: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, int, int]:
    """
    Extract non-overlapping patches from images.

    Args:
        images: (T, H, W) grayscale images
        patch_size: patch dimension (patch_size x patch_size)

    Returns:
        patches: (N, P) flattened patches, N = T * H_blocks * W_blocks,
                 P = patch_size * patch_size
        H_blocks: number of patches vertically
        W_blocks: number of patches horizontally
    """
    T, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dims must be multiples of patch_size."

    H_blocks = H // patch_size
    W_blocks = W // patch_size
    P = patch_size * patch_size

    patches = np.zeros((T * H_blocks * W_blocks, P), dtype=np.float32)

    idx = 0
    for t in range(T):
        for hb in range(H_blocks):
            for wb in range(W_blocks):
                h0 = hb * patch_size
                w0 = wb * patch_size
                patch = images[t, h0:h0 + patch_size, w0:w0 + patch_size]
                patches[idx, :] = patch.reshape(-1)
                idx += 1

    return patches, H_blocks, W_blocks


def compute_svd_basis_over_patches(
    patches: np.ndarray,
    K: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute an SVD basis over all patches.

    Args:
        patches: (N, P) flattened patches
        K: number of singular vectors to keep

    Returns:
        U: (N, K) left singular vectors (per-patch coefficients, unscaled)
        S: (K,) singular values
        Vt: (K, P) right singular vectors (basis images)
    """
    # Center patches
    patches_centered = patches - patches.mean(axis=0, keepdims=True)

    # Compute SVD via numpy
    U_full, S_full, Vt_full = np.linalg.svd(patches_centered, full_matrices=False)
    U = U_full[:, :K]
    S = S_full[:K]
    Vt = Vt_full[:K, :]
    return U, S, Vt


# -----------------------------------------------------------------------------
# 4. Quantize SVD coefficients into discrete bins
# -----------------------------------------------------------------------------

def quantize_coefficients(
    coeffs: np.ndarray,
    L: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize SVD coefficients into L bins per component.

    Args:
        coeffs: (N, K) SVD coefficients (U or U*S)
        L: number of quantization levels per component

    Returns:
        q_coeffs: (N, K) integer quantized indices in {0,...,L-1}
        bin_edges: (K, L+1) bin boundaries per component
        bin_centers: (K, L) bin centers per component
    """
    N, K = coeffs.shape
    q_coeffs = np.zeros((N, K), dtype=np.int32)
    bin_edges = np.zeros((K, L + 1), dtype=np.float32)
    bin_centers = np.zeros((K, L), dtype=np.float32)

    for k in range(K):
        c_k = coeffs[:, k]
        c_min, c_max = np.percentile(c_k, [0.5, 99.5])
        edges = np.linspace(c_min, c_max, L + 1)
        bin_edges[k] = edges
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_centers[k] = centers

        # Digitize coefficients into bins
        idxs = np.digitize(c_k, edges) - 1
        idxs = np.clip(idxs, 0, L - 1)
        q_coeffs[:, k] = idxs

    return q_coeffs, bin_edges, bin_centers


def encode_mixed_radix_states(
    q_coeffs: np.ndarray,
    L: int,
) -> np.ndarray:
    """
    Encode quantized coefficients into a single mixed-radix state index
    per patch.

    For K components and L levels per component, each patch's discrete
    state is:

        s = sum_{k=0..K-1} q_coeffs[n, k] * L^k

    Args:
        q_coeffs: (N, K) integer quantized coefficients
        L: number of levels per component

    Returns:
        states: (N,) integer encoded states in {0,...,L^K-1}
    """
    N, K = q_coeffs.shape
    bases = (L ** np.arange(K, dtype=np.int64)).reshape(1, K)  # (1, K)
    states = (q_coeffs.astype(np.int64) * bases).sum(axis=1)   # (N,)
    return states


# -----------------------------------------------------------------------------
# 5. High-level data builder for Lorenz RGM
# -----------------------------------------------------------------------------

def build_lorenz_patch_dataset(
    T: int,
    dt: float,
    img_size: int,
    patch_size: int,
    K: int,
    L: int,
    thickness: int = 1,
) -> Dict[str, Any]:
    """
    Build a dataset for the Lorenz RGM at the patch level.

    Steps:
      1. Simulate a Lorenz trajectory of length T.
      2. Render the trajectory into grayscale images of size img_size x img_size.
      3. Extract non-overlapping patches of size patch_size x patch_size
         from each image.
      4. Compute an SVD basis over all patches, keep top K components.
      5. Quantize coefficients into L bins per component.
      6. Encode quantized coefficients into a single mixed-radix state index
         per patch, giving discrete level-0 states.

    Args:
        T: number of time steps
        dt: time step for Lorenz simulation
        img_size: size of rendered images (pixels)
        patch_size: patch side length (pixels)
        K: number of SVD components to keep
        L: number of quantization levels per component
        thickness: line thickness for rendering Lorenz trajectory

    Returns:
        lorenz_data_dict with keys:
            - 'traj': (T, 3) Lorenz states
            - 'images': (T, img_size, img_size) grayscale images
            - 'patches': (N, P) flattened patches
            - 'q_coeffs': (N, K) quantized coefficient indices
            - 'states': (N,) integer discrete states per patch
            - 'T': T
            - 'dt': dt
            - 'img_size': img_size
            - 'patch_size': patch_size
            - 'H_blocks', 'W_blocks': patch grid dimensions
            - 'K', 'L': SVD/quantization config
    """
    # 1. Simulate Lorenz
    traj = simulate_lorenz(T=T, dt=dt)

    # 2. Render to images
    images = render_lorenz_to_images(traj, img_size=img_size, thickness=thickness)

    # 3. Extract patches
    patches, H_blocks, W_blocks = extract_patches(images, patch_size=patch_size)

    # 4. SVD over patches
    U, S, Vt = compute_svd_basis_over_patches(patches, K=K)

    # Use left singular vectors U as coefficients (optionally scaled by S)
    coeffs = U * S[None, :]  # (N, K)

    # 5. Quantize coefficients
    q_coeffs, bin_edges, bin_centers = quantize_coefficients(coeffs, L=L)

    # 6. Encode into discrete states
    states = encode_mixed_radix_states(q_coeffs, L=L)  # (N,)

    lorenz_data_dict: Dict[str, Any] = {
        "traj": traj,
        "images": images,
        "patches": patches,
        "q_coeffs": q_coeffs,
        "states": states,
        "T": T,
        "dt": dt,
        "img_size": img_size,
        "patch_size": patch_size,
        "H_blocks": H_blocks,
        "W_blocks": W_blocks,
        "K": K,
        "L": L,
        "svd_U": U,
        "svd_S": S,
        "svd_Vt": Vt,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
    }
    return lorenz_data_dict
