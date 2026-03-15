# lorenz_efe.py
"""
Expected free energy (EFE) computations for the Lorenz RGM example.

This module implements a decomposition of expected free energy G(u_t)
for top-level paths u_t that is aligned with the formulation in
"From pixels to planning: scale-free active inference" (Friston, 2025):
  - risk: mismatch between predicted and preferred outcomes
  - ambiguity: expected conditional entropy of observations given states
  - epistemic value (state information gain): reduction in state uncertainty

We implement these terms in a way that is consistent with the Lorenz
hierarchy:
  - top level: states and paths (LorenzLevel at highest spatial level)
  - lowest level: observations via A (LorenzLevel at level 0)
  - spatial coupling via D has already been used to obtain posteriors
    over states at all levels.
"""

from typing import Optional

import jax
import jax.numpy as jnp

from .lorenz_model import LorenzLevel
from . import maths as rgm_maths  # adapted from pymdp.jax.maths


# -----------------------------------------------------------------------------
# 1. Predictive distributions at the lowest level
# -----------------------------------------------------------------------------

def compute_predictive_state_lowest(
    qs0_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute a spatially averaged predictive state distribution at the
    lowest level.

    Args:
        qs0_grid: (T, H0, W0, S0) posterior over Level-0 states

    Returns:
        qs0_pred: (T, S0) predictive distribution over lowest-level states
    """
    # For now, take the spatial average as our predictive marginal.
    qs0_pred = qs0_grid.mean(axis=(1, 2))  # (T, S0)
    # Ensure normalization
    qs0_pred = qs0_pred / (qs0_pred.sum(axis=1, keepdims=True) + 1e-8)
    return qs0_pred


def compute_predictive_obs_lowest(
    qs0_pred: jnp.ndarray,
    level0: LorenzLevel,
) -> jnp.ndarray:
    """
    Compute predicted observation distribution at the lowest level
    given predictive state distribution.

    Args:
        qs0_pred: (T, S0) predictive state distribution
        level0: LorenzLevel with A (S0, O)

    Returns:
        qo_pred: (T, O) predictive observation distribution
    """
    A0 = level0.A  # (S0, O)
    T, S0 = qs0_pred.shape
    O = A0.shape[1]

    def pred_obs_t(qs_t: jnp.ndarray) -> jnp.ndarray:
        qo = qs_t @ A0  # (O,)
        qo = qo / (qo.sum() + 1e-8)
        return qo

    qo_pred = jax.vmap(pred_obs_t)(qs0_pred)  # (T, O)
    return qo_pred


# -----------------------------------------------------------------------------
# 2. Risk and ambiguity (observation-model terms)
# -----------------------------------------------------------------------------

def compute_risk_term(
    qo_pred: jnp.ndarray,
    C: jnp.ndarray,
) -> jnp.ndarray:
    """
    Risk term: mismatch between predicted and preferred outcomes.

    In the risk–ambiguity decomposition, risk is often expressed as
    a KL divergence between predicted q(o') and preferences C(o'):

        risk_t = KL[q(o'_t) || C]

    Args:
        qo_pred: (T, O) predictive observation distributions
        C: (O,) preference distribution over outcomes

    Returns:
        risk: (T,) risk per time step
    """
    log_q = jnp.log(jnp.clip(qo_pred, 1e-16))
    log_C = jnp.log(jnp.clip(C[None, :], 1e-16))
    # KL[q || C] = sum_o q(o) [log q(o) - log C(o)]
    kl = (qo_pred * (log_q - log_C)).sum(axis=1)
    return kl  # (T,)


def compute_ambiguity_term(
    qs0_pred: jnp.ndarray,
    level0: LorenzLevel,
) -> jnp.ndarray:
    """
    Ambiguity term: expected conditional entropy of observations given
    predictive states:

        ambiguity_t = E_{q(s'_t)} [ H[p(o' | s'_t)] ]

    Where H[p(o|s)] is the entropy of the observation model for state s.

    Args:
        qs0_pred: (T, S0) predictive state distribution
        level0: LorenzLevel with A (S0, O)

    Returns:
        ambiguity: (T,) ambiguity per time step
    """
    A0 = level0.A  # (S0, O)
    S0, O = A0.shape

    # Entropy of p(o|s) for each state s
    # H[o|s] = - sum_o A0[s, o] log A0[s, o]
    A0_clipped = jnp.clip(A0, 1e-16)
    logA0 = jnp.log(A0_clipped)
    H_o_given_s = - (A0_clipped * logA0).sum(axis=1)  # (S0,)

    def amb_t(qs_t: jnp.ndarray) -> float:
        return (qs_t * H_o_given_s).sum()

    ambiguity = jax.vmap(amb_t)(qs0_pred)  # (T,)
    return ambiguity


# -----------------------------------------------------------------------------
# 3. Epistemic (state-information) term
# -----------------------------------------------------------------------------

def compute_epistemic_term(
    qs1_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Epistemic term: information gain about states (simplified).

    A full RGM treatment would compute mutual information between
    future observations and states. Here we use a proxy based on
    the expected reduction in entropy of top-level states:

        epistemic_t ∝ - H[ q(s^1_t) ]

    where q(s^1_t) is the marginal over top-level states at time t.

    Args:
        qs1_grid: (T, H1, W1, S1) top-level state posteriors

    Returns:
        epistemic: (T,) epistemic term per time step (negative entropy proxy)
    """
    # Spatial mean over sites
    def ent_t(qs1_t: jnp.ndarray) -> float:
        # qs1_t: (H1, W1, S1)
        qs1_mean = qs1_t.mean(axis=(0, 1))  # (S1,)
        qs1_mean = qs1_mean / (qs1_mean.sum() + 1e-8)
        log_q = jnp.log(jnp.clip(qs1_mean, 1e-16))
        H = - (qs1_mean * log_q).sum()
        # epistemic ~ -H (prefer lower entropy / higher information)
        return -H

    epistemic = jax.vmap(ent_t)(qs1_grid)  # (T,)
    return epistemic


# -----------------------------------------------------------------------------
# 4. Combined G(u) and path posterior update
# -----------------------------------------------------------------------------

def compute_expected_free_energy_paths(
    level_top: LorenzLevel,
    level0: LorenzLevel,
    qs1_grid: jnp.ndarray,
    qs0_grid: jnp.ndarray,
    C: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute expected free energy G_t (per time) for paths at the top level,
    using a risk–ambiguity–epistemic decomposition aligned with the
    "From pixels to planning" formulation.

    At this stage, we do not yet condition G on specific path values u,
    because we have not introduced path-dependent transitions B(s'|s,u)
    explicitly in the Lorenz model. Thus G_t is the same for all u and
    is used as a scalar cost at each time.

    Args:
        level_top: top-level LorenzLevel (with num_paths > 0)
        level0: lowest-level LorenzLevel (with A)
        qs1_grid: (T, H1, W1, S1) top-level state posteriors
        qs0_grid: (T, H0, W0, S0) lowest-level state posteriors
        C: (O,) preference distribution over lowest-level outcomes

    Returns:
        G_tu: (T, U) expected free energy per time and path (same across u)
    """
    U = level_top.num_paths
    if U == 0:
        raise ValueError("compute_expected_free_energy_paths called with num_paths=0")

    # Predictive states and observations at lowest level
    qs0_pred = compute_predictive_state_lowest(qs0_grid)       # (T, S0)
    qo_pred = compute_predictive_obs_lowest(qs0_pred, level0)  # (T, O)

    # Risk and ambiguity terms
    risk = compute_risk_term(qo_pred, C)                       # (T,)
    ambiguity = compute_ambiguity_term(qs0_pred, level0)       # (T,)

    # Epistemic (state-info) term
    epistemic = compute_epistemic_term(qs1_grid)               # (T,)

    # Combine into G_t; sign convention:
    # G_t = risk + ambiguity - epistemic
    G_t = risk + ambiguity - epistemic  # (T,)

    # Broadcast over paths (all paths share same G in this simplified Lorenz case)
    G_tu = jnp.broadcast_to(G_t[:, None], (G_t.shape[0], U))
    return G_tu


def update_path_posterior_from_G(
    level_top: LorenzLevel,
    G_tu: jnp.ndarray,
    gamma: float = 16.0,
    num_iter: int = 2,
) -> jnp.ndarray:
    """
    Update the path posterior q(u_t) using G_tu and the path transition
    model at the top level.

        p(u_t) ∝ exp(-gamma * G_tu)
        q(u) is then updated via a simple VMP using B_paths and E_paths.

    Args:
        level_top: top-level LorenzLevel with B_paths, E_paths, num_paths > 0
        G_tu: (T, U) expected free energy per time and path
        gamma: precision over expected free energy
        num_iter: number of VMP iterations over the path chain

    Returns:
        qu_t: (T, U) posterior over paths at each time
    """
    U = level_top.num_paths
    if U == 0:
        raise ValueError("update_path_posterior_from_G called with num_paths=0")

    B_paths = level_top.B_paths  # (U, U)
    E_paths = level_top.E_paths  # (U,)

    T = G_tu.shape[0]

    # Prior over u_t based on EFE: p(u_t) ∝ exp(-gamma * G_tu)
    def prior_u_t(G_t: jnp.ndarray) -> jnp.ndarray:
        logits = -gamma * G_t  # (U,)
        return jnp.exp(logits) / (jnp.exp(logits).sum() + 1e-8)

    p_u_t = jax.vmap(prior_u_t)(G_tu)  # (T, U)

    # Initialize q(u_t) ~ uniform
    qu = jnp.full((T, U), 1.0 / U, dtype=jnp.float32)

    def forward_messages(qu_):
        msgs = []
        prev_q = qu_[0]
        msgs.append(jnp.log(jnp.clip(E_paths, 1e-16)))
        for t in range(1, T):
            msg = jnp.log(jnp.clip(B_paths @ prev_q, 1e-16))
            msgs.append(msg)
            prev_q = qu_[t]
        return jnp.stack(msgs, axis=0)

    def backward_messages(qu_):
        msgs = []
        next_q = qu_[-1]
        msgs.append(jnp.zeros_like(next_q))
        for t in range(T - 2, -1, -1):
            msg = jnp.log(jnp.clip(B_paths.T @ next_q, 1e-16))
            msgs.append(msg)
            next_q = qu_[t]
        msgs = msgs[::-1]
        return jnp.stack(msgs, axis=0)

    def update_once(qu_):
        m_plus = forward_messages(qu_)
        m_minus = backward_messages(qu_)
        ln_qu = jnp.log(jnp.clip(p_u_t, 1e-16)) + m_plus + m_minus
        return jnp.exp(ln_qu) / (jnp.exp(ln_qu).sum(axis=1, keepdims=True) + 1e-8)

    def body_fun(_, qu_):
        return update_once(qu_)

    qu_final = jax.lax.fori_loop(0, num_iter, body_fun, qu)
    return qu_final
