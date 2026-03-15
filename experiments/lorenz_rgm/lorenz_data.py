# lorenz_data.py
"""
Lorenz simulator and image discretization utilities for the Lorenz RGM example.

This module:
1. Simulates Lorenz trajectories in continuous space.
2. Renders trajectories as grayscale images on a 2D grid.
3. Tiles images into patches and computes an SVD basis over patches.
4. Projects patches onto the SVD basis and quantizes coefficients to discrete levels.
5. Encodes quantized coefficient vectors as categorical patch states.

The outputs are designed to be used as the lowest-level "observation / state"
representation in the renormalizing generative model (RGM).
"""

from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# 1. Lorenz dynamical system
# -----------------------------------------------------------------------------

def lorenz_step(state: jnp.ndarray,
                params: jnp.ndarray,
                dt: float) -> jnp.ndarray:
    """
    Single Euler step for Lorenz system.

    Args:
        state: (3,) array [x, y, z]
        params: (3,) array [sigma, rho, beta]
        dt: time step

    Returns:
        next_state: (3,) array
    """
    x, y, z = state
    sigma, rho, beta = params

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    return state + dt * jnp.array([dx, dy, dz], dtype=state.dtype)


def simulate_lorenz(T: int = 1000,
                    dt: float = 0.01,
                    sigma: float = 10.0,
                    rho: float = 28.0,
                    beta: float = 8.0 / 3.0,
                    x0: Tuple[float, float, float] = (1.0, 1.0, 1.0)
                    ) -> jnp.ndarray:
    """
    Simulate Lorenz trajectory.

    Args:
        T: number of time steps
        dt: time step
        sigma, rho, beta: Lorenz parameters
        x0: initial state

    Returns:
        traj: (T, 3) array of [x, y, z] over time
    """
    params = jnp.array([sigma, rho, beta], dtype=jnp.float32)
    init_state = jnp.array(x0, dtype=jnp.float32)

    @jax.jit
    def scan_body(carry, _):
        next_state = lorenz_step(carry, params, dt)
        return next_state, next_state

    _, traj = jax.lax.scan(scan_body, init_state, None, length=T)
    return traj  # (T, 3)


# -----------------------------------------------------------------------------
# 2. Rendering Lorenz trajectory into images
# -----------------------------------------------------------------------------

def render_lorenz_images(traj: jnp.ndarray,
                         img_size: int = 64,
                         thickness: int = 1,
                         normalize_each: bool = False
                         ) -> jnp.ndarray:
    """
    Render Lorenz (x, y) trajectory into a sequence of grayscale images.

    Args:
        traj: (T, 3) array, Lorenz trajectory
        img_size: spatial resolution (H = W = img_size)
        thickness: controls drawing thickness (simple dilation in a 3x3 neighborhood
                   repeated 'thickness' times)
        normalize_each: if True, normalize each frame to [0,1] separately;
                        otherwise normalize over the whole trajectory.

    Returns:
        images: (T, img_size, img_size) array in [0, 1]
    """
    T = traj.shape[0]
    x, y = traj[:, 0], traj[:, 1]

    if normalize_each:
        # Normalize per-frame (less typical; usually global)
        x_min = x.reshape(T, 1).min(axis=1)
        x_max = x.reshape(T, 1).max(axis=1)
        y_min = y.reshape(T, 1).min(axis=1)
        y_max = y.reshape(T, 1).max(axis=1)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        y_norm = (y - y_min) / (y_max - y_min + 1e-8)
    else:
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

    xs = (x_norm * (img_size - 1)).astype(jnp.int32)
    ys = (y_norm * (img_size - 1)).astype(jnp.int32)

    images = jnp.zeros((T, img_size, img_size), dtype=jnp.float32)

    def draw_point(i, imgs):
        return imgs.at[i, ys[i], xs[i]].set(1.0)

    images = jax.lax.fori_loop(0, T, draw_point, images)

    if thickness > 1:
        # Simple morphological dilation with a 3x3 kernel repeated 'thickness' times
        kernel = jnp.array([[0., 1., 0.],
                            [1., 1., 1.],
                            [0., 1., 0.]], dtype=jnp.float32)

        def dilate_once(imgs_):
            # imgs_: (T, H, W)
            # naive implementation using conv with padding
            def convolve_single(im):
                # im: (H, W)
                H, W = im.shape
                # pad
                im_p = jnp.pad(im, ((1, 1), (1, 1)))
                # sliding 3x3
                def body(r, acc):
                    def inner_body(c, acc2):
                        patch = im_p[r:r+3, c:c+3]
                        val = (patch * kernel).max()
                        acc2 = acc2.at[r, c].set(val)
                        return acc2
                    acc = jax.lax.fori_loop(0, W, inner_body, acc)
                    return acc

                out = jnp.zeros((H, W), dtype=jnp.float32)
                out = jax.lax.fori_loop(0, H, body, out)
                return jnp.clip(out, 0.0, 1.0)

            return jax.vmap(convolve_single)(imgs_)

        for _ in range(thickness - 1):
            images = dilate_once(images)

    return images


# -----------------------------------------------------------------------------
# 3. Patch extraction and SVD basis computation
# -----------------------------------------------------------------------------

def extract_patches(images: jnp.ndarray,
                    patch_size: int = 4
                    ) -> Tuple[jnp.ndarray, int, int]:
    """
    Extract non-overlapping patches from images.

    Args:
        images: (T, H, W) array
        patch_size: size P of square patches (H, W must be divisible by P)

    Returns:
        patches: (N_patches, P*P) array
        H_blocks: number of patches along height
        W_blocks: number of patches along width
    """
    T, H, W = images.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "H and W must be divisible by patch_size"

    H_blocks, W_blocks = H // P, W // P

    # Reshape into (T, H_blocks, P, W_blocks, P) then reorder to put patches together
    patches = images.reshape(T, H_blocks, P, W_blocks, P)
    patches = jnp.transpose(patches, (0, 1, 3, 2, 4))  # (T, H_blocks, W_blocks, P, P)
    patches = patches.reshape(T * H_blocks * W_blocks, P * P)
    return patches, H_blocks, W_blocks


def compute_svd_basis(patches: jnp.ndarray,
                      K: int = 4
                      ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute an SVD basis over patches.

    Args:
        patches: (N, D) array of flattened patches
        K: number of singular vectors to retain

    Returns:
        mean: (D,) mean patch
        basis: (K, D) matrix (rows are basis vectors)
    """
    # Center patches
    mean = patches.mean(axis=0, keepdims=False)
    X = patches - mean

    # SVD on (N, D) matrix: X = U S V^T, with V: (D, D)
    # For D up to e.g. 16 or 64, this is fine.
    U, S, Vt = jnp.linalg.svd(X, full_matrices=False)
    basis = Vt[:K, :]  # (K, D)
    return mean, basis


# -----------------------------------------------------------------------------
# 4. Projection, quantization, encoding
# -----------------------------------------------------------------------------

def project_patches(patches: jnp.ndarray,
                    mean: jnp.ndarray,
                    basis: jnp.ndarray
                    ) -> jnp.ndarray:
    """
    Project patches onto SVD basis.

    Args:
        patches: (N, D)
        mean: (D,)
        basis: (K, D)

    Returns:
        coeffs: (N, K)
    """
    X = patches - mean  # (N, D)
    coeffs = X @ basis.T  # (N, K)
    return coeffs


def quantize_coeffs(coeffs: jnp.ndarray,
                    L: int = 7
                    ) -> jnp.ndarray:
    """
    Quantize each coefficient dimension independently into L discrete levels.

    Args:
        coeffs: (N, K)
        L: number of quantization levels (bins)

    Returns:
        q: (N, K) integer array with entries in [0, L-1]
    """
    N, K = coeffs.shape

    def quantize_dim(c: jnp.ndarray) -> jnp.ndarray:
        c_min = c.min()
        c_max = c.max()
        # Avoid degenerate case
        range_ = c_max - c_min
        range_ = jnp.where(range_ < 1e-8, 1.0, range_)
        c_norm = (c - c_min) / range_
        # bins edges, L+1 edges correspond to L bins
        bins = jnp.linspace(0.0, 1.0, L + 1)
        # digitize: returns indices in 1..L, clip to [1, L]
        idx = jnp.digitize(c_norm, bins) - 1
        idx = jnp.clip(idx, 0, L - 1)
        return idx.astype(jnp.int32)

    # vmap over the K dimensions
    q = jax.vmap(quantize_dim, in_axes=1, out_axes=1)(coeffs)
    return q  # (N, K)


def encode_quantized_coeffs(q: jnp.ndarray,
                            L: int
                            ) -> jnp.ndarray:
    """
    Encode a K-dimensional quantized coefficient vector into a single integer state.

    Uses mixed-radix encoding with base L in each dimension.

    Args:
        q: (N, K) integer array with entries in [0, L-1]
        L: radix

    Returns:
        states: (N,) integer array in [0, L^K - 1]
    """
    N, K = q.shape
    multipliers = (L ** jnp.arange(K)).astype(jnp.int32)  # [1, L, L^2, ...]
    states = (q * multipliers).sum(axis=1)
    return states


def decode_quantized_coeffs(states: jnp.ndarray,
                            L: int,
                            K: int
                            ) -> jnp.ndarray:
    """
    Inverse of encode_quantized_coeffs: decode integer states back to K-dim codes.

    Args:
        states: (N,) integer array in [0, L^K - 1]
        L: radix
        K: number of coefficient dimensions

    Returns:
        q: (N, K) integer array in [0, L-1]
    """
    states = states.astype(jnp.int32)
    multipliers = (L ** jnp.arange(K)).astype(jnp.int32)

    # We compute digits base L
    def decode_single(s: jnp.ndarray) -> jnp.ndarray:
        digits = []
        for k in range(K):
            digit = (s // multipliers[k]) % L
            digits.append(digit)
        return jnp.stack(digits, axis=0)

    q = jax.vmap(decode_single)(states)
    return q  # (N, K)


def reconstruct_patches_from_coeffs(coeffs: jnp.ndarray,
                                    mean: jnp.ndarray,
                                    basis: jnp.ndarray
                                    ) -> jnp.ndarray:
    """
    Reconstruct patches from SVD coefficients.

    Args:
        coeffs: (N, K)
        mean: (D,)
        basis: (K, D)

    Returns:
        patches: (N, D)
    """
    patches = coeffs @ basis  # (N, D)
    patches = patches + mean
    return patches


def reconstruct_patches_from_quantized(q: jnp.ndarray,
                                       L: int,
                                       mean: jnp.ndarray,
                                       basis: jnp.ndarray
                                       ) -> jnp.ndarray:
    """
    Reconstruct patches from quantized coefficient codes, using bin centers.

    Args:
        q: (N, K) integer array in [0, L-1]
        L: number of quantization levels
        mean: (D,)
        basis: (K, D)

    Returns:
        patches: (N, D)
    """
    K = q.shape[1]
    # Bin centers in [0,1]; we do not invert original scale exactly here.
    bin_centers = (jnp.arange(L, dtype=jnp.float32) + 0.5) / L  # (L,)
    # Map q to [0,1] via bin centers
    coeffs_norm = bin_centers[q]  # (N, K)
    # Note: the scale of coeffs_norm is arbitrary relative to original coeffs.
    # For qualitative reconstruction this is often sufficient; for exact
    # reconstruction we would store per-dimension min/max.
    patches = reconstruct_patches_from_coeffs(coeffs_norm, mean, basis)
    return patches


# -----------------------------------------------------------------------------
# 5. End-to-end pipeline helpers
# -----------------------------------------------------------------------------

def build_lorenz_patch_dataset(T: int = 1000,
                               dt: float = 0.01,
                               img_size: int = 64,
                               patch_size: int = 4,
                               K: int = 4,
                               L: int = 7,
                               thickness: int = 1
                               ) -> Dict[str, Any]:
    """
    End-to-end pipeline:
    - simulate Lorenz
    - render images
    - extract patches
    - compute SVD basis
    - quantize coefficients
    - encode patch states

    Args:
        T: number of time steps in Lorenz trajectory
        dt: time step
        img_size: image size
        patch_size: patch side length
        K: number of singular vectors
        L: number of quantization levels
        thickness: drawing thickness in rendering

    Returns:
        A dict with:
            'traj': (T, 3) Lorenz trajectory
            'images': (T, H, W) raw images
            'patches': (N_patches, P*P) raw patches
            'mean': (P*P,) patch mean
            'basis': (K, P*P) SVD basis
            'coeffs': (N_patches, K) SVD coefficients
            'q_coeffs': (N_patches, K) quantized coeffs
            'states': (N_patches,) encoded patch states
            'H_blocks': int number of patches along height
            'W_blocks': int number of patches along width
            'K': int, number of components
            'L': int, number of quantization levels
    """
    traj = simulate_lorenz(T=T, dt=dt)
    images = render_lorenz_images(traj, img_size=img_size, thickness=thickness)

    patches, H_blocks, W_blocks = extract_patches(images, patch_size=patch_size)
    mean, basis = compute_svd_basis(patches, K=K)
    coeffs = project_patches(patches, mean, basis)
    q_coeffs = quantize_coeffs(coeffs, L=L)
    states = encode_quantized_coeffs(q_coeffs, L=L)

    return {
        "traj": traj,
        "images": images,
        "patches": patches,
        "mean": mean,
        "basis": basis,
        "coeffs": coeffs,
        "q_coeffs": q_coeffs,
        "states": states,
        "H_blocks": H_blocks,
        "W_blocks": W_blocks,
        "K": K,
        "L": L,
        "patch_size": patch_size,
        "img_size": img_size,
        "T": T,
        "dt": dt,
    }
