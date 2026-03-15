# lorenz_efe.py
"""
Expected free energy (EFE) computations for the Lorenz RGM example.

This module implements a decomposition of expected free energy G(t, u)
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

Crucially, transitions among top-level states depend on paths u
via a path-dependent transition tensor B_states_paths[u, s', s].
Expected free energy is therefore computed per-path by rolling forward
multi-step predictive trajectories under each B^{(u)} over a finite
horizon tau.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .lorenz_model import LorenzLevel
from . import maths as rgm_maths  # adapted from pymdp.jax.maths


# -----------------------------------------------------------------------------
# 1. Risk and ambiguity (observation-model terms)
# -----------------------------------------------------------------------------

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

    def pred_obs_t(qs_t: jnp.ndarray) -> jnp.ndarray:
        qo = qs_t @ A0  # (O,)
        qo = qo / (qo.sum() + 1e-8)
        return qo

    qo_pred = jax.vmap(pred_obs_t)(qs0_pred)  # (T, O)
    return qo_pred


def compute_risk_term(
    qo_pred: jnp.ndarray,
    C: jnp.ndarray,
) -> jnp.ndarray:
    """
    Risk term: mismatch between predicted and preferred outcomes.

        risk_t = KL[q(o'_t) || C]

    Args:
        qo_pred: (T, O) predictive observation distributions
        C: (O,) preference distribution over outcomes

    Returns:
        risk: (T,) risk per time step
    """
    log_q = jnp.log(jnp.clip(qo_pred, 1e-16))
    log_C = jnp.log(jnp.clip(C[None, :], 1e-16))
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

    Args:
        qs0_pred: (T, S0) predictive state distribution
        level0: LorenzLevel with A (S0, O)

    Returns:
        ambiguity: (T,) ambiguity per time step
    """
    A0 = level0.A  # (S0, O)
    A0_clipped = jnp.clip(A0, 1e-16)
    logA0 = jnp.log(A0_clipped)
    H_o_given_s = - (A0_clipped * logA0).sum(axis=1)  # (S0,)

    def amb_t(qs_t: jnp.ndarray) -> float:
        return (qs_t * H_o_given_s).sum()

    ambiguity = jax.vmap(amb_t)(qs0_pred)  # (T,)
    return ambiguity


# -----------------------------------------------------------------------------
# 2. Epistemic (state-information) term
# -----------------------------------------------------------------------------

def compute_epistemic_term_from_qs1_pred(
    qs1_pred_grid: jnp.ndarray,
) -> jnp.ndarray:
    """
    Epistemic term: information gain about top-level states (proxy).

    We use a proxy based on the negative entropy of predicted top-level
    state marginals, averaged over space:

        epistemic_t(u) ∝ - H[ q(s^1_t | u) ]

    Args:
        qs1_pred_grid: (T, H1, W1, S1) predicted top-level state distributions

    Returns:
        epistemic: (T,) per-time epistemic term (negative entropy proxy)
    """
    def ent_t(qs1_t: jnp.ndarray) -> float:
        qs1_mean = qs1_t.mean(axis=(0, 1))  # (S1,)
        qs1_mean = qs1_mean / (qs1_mean.sum() + 1e-8)
        log_q = jnp.log(jnp.clip(qs1_mean, 1e-16))
        H = - (qs1_mean * log_q).sum()
        return -H

    epistemic = jax.vmap(ent_t)(qs1_pred_grid)  # (T,)
    return epistemic


# -----------------------------------------------------------------------------
# 3. Multi-step path-specific predictive rollouts
# -----------------------------------------------------------------------------

def rollout_predictive_states_under_path_tau(
    qs0_grid: jnp.ndarray,
    qs1_grid: jnp.ndarray,
    level_top: LorenzLevel,
    level0: LorenzLevel,
    u_idx: int,
    tau: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    For a given path index u, construct a multi-step predictive trajectory
    over a horizon tau, using:

      - B_states_paths[u] for top-level transitions.
      - Current posteriors as proxies for the present state at time t.
      - A simple Markovian rollout over tau steps (no control variation).

    We return sequences of length tau, starting from the current
    posterior marginals; in practice, G(t, u) will be computed by
    aggregating over this local horizon.

    Args:
        qs0_grid: (T, H0, W0, S0) posterior over Level-0 states (data-driven)
        qs1_grid: (T, H1, W1, S1) posterior over Level-1 states (data-driven)
        level_top: top-level LorenzLevel with B_states_paths[u]
        level0: lowest-level LorenzLevel with A
        u_idx: integer index of the path (0 <= u_idx < U)
        tau: planning horizon length (number of predictive steps)

    Returns:
        qs0_pred_u: (tau, S0) predictive lowest-level state marginals under path u
                    (here, taken from current spatial averages as a simple proxy)
        qs1_pred_u_grid: (tau, H1, W1, S1) predictive top-level state grid under u
        qo_pred_u: (tau, O) predictive observations under path u
    """
    # Current posteriors as base "starting" states at some reference time.
    # Here we use the final time slice as the current time for rollout.
    qs0_T = qs0_grid[-1]  # (H0, W0, S0)
    qs0_base = qs0_T.mean(axis=(0, 1))  # (S0,)
    qs0_base = qs0_base / (qs0_base.sum() + 1e-8)

    qs1_T = qs1_grid[-1]  # (H1, W1, S1)

    if level_top.B_states_paths is None:
        # No path dependence: propagate with identity (no change)
        B_u = jnp.eye(qs1_T.shape[-1], dtype=jnp.float32)
    else:
        B_u = level_top.B_states_paths[u_idx]  # (S1, S1)

    H1, W1, S1 = qs1_T.shape

    def step_site(qs1_hw: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        q_next = qs1_hw @ B  # (S1,)
        q_next = q_next / (q_next.sum() + 1e-8)
        return q_next

    def step_grid(qs1_grid_t: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(jax.vmap(step_site, in_axes=(0, None)), in_axes=(0, None))(qs1_grid_t, B)

    def scan_step(carry, _):
        qs1_curr = carry  # (H1, W1, S1)
        qs1_next = step_grid(qs1_curr, B_u)
        return qs1_next, qs1_next

    # Initialize with current qs1_T, then roll tau steps forward
    _, qs1_pred_seq = jax.lax.scan(scan_step, qs1_T, None, length=tau)
    # qs1_pred_seq: (tau, H1, W1, S1)

    # For this Lorenz example, we keep qs0_pred_u constant over tau
    qs0_pred_u = jnp.broadcast_to(qs0_base[None, :], (tau, qs0_base.shape[0]))  # (tau, S0)

    qo_pred_u = compute_predictive_obs_lowest(qs0_pred_u, level0)  # (tau, O)

    return qs0_pred_u, qs1_pred_seq, qo_pred_u


# -----------------------------------------------------------------------------
# 4. Combined G(t, u) with horizon tau and path posterior update
# -----------------------------------------------------------------------------

def compute_expected_free_energy_paths(
    level_top: LorenzLevel,
    level0: LorenzLevel,
    qs1_grid: jnp.ndarray,
    qs0_grid: jnp.ndarray,
    C: jnp.ndarray,
    tau: int = 3,
) -> jnp.ndarray:
    """
    Compute expected free energy G_tu (per current time t and path u) at
    the top level, using a risk–ambiguity–epistemic decomposition over
    a multi-step horizon tau.

    In this implementation, we approximate G(t, u) by:

        G(t, u) ≈ sum_{k=1..tau} [ risk_{t+k|u} + ambiguity_{t+k|u}
                                   - epistemic_{t+k|u} ]

    where predictive distributions q(s, o | u) over the horizon are
    generated by rolling forward under B_states_paths[u].

    For simplicity and computational tractability in the Lorenz example,
    we:
      - Use the last observed/posterior time point as the "current" time.
      - Roll forward tau steps from that time for each path u.
      - Treat this G(t, u) as time-constant across t (broadcast).

    Args:
        level_top: top-level LorenzLevel (with num_paths > 0 and B_states_paths)
        level0: lowest-level LorenzLevel (with A)
        qs1_grid: (T, H1, W1, S1) top-level state posteriors (data)
        qs0_grid: (T, H0, W0, S0) lowest-level state posteriors (data)
        C: (O,) preference distribution over lowest-level outcomes
        tau: integer planning horizon (number of predictive steps)

    Returns:
        G_tu: (T, U) expected free energy per time and path
    """
    U = level_top.num_paths
    if U == 0:
        raise ValueError("compute_expected_free_energy_paths called with num_paths=0")

    T = qs1_grid.shape[0]

    # If no path-specific dynamics, fall back to scalar G_t
    if level_top.B_states_paths is None:
        qs0_pred = qs0_grid.mean(axis=(1, 2))  # (T, S0)
        qs0_pred = qs0_pred / (qs0_pred.sum(axis=1, keepdims=True) + 1e-8)
        qo_pred = compute_predictive_obs_lowest(qs0_pred, level0)
        risk = compute_risk_term(qo_pred, C)
        ambiguity = compute_ambiguity_term(qs0_pred, level0)
        epistemic = compute_epistemic_term_from_qs1_pred(qs1_grid)
        G_t = risk + ambiguity - epistemic
        return jnp.broadcast_to(G_t[:, None], (T, U))

    # Path-specific G_u via tau-step rollout
    def G_for_path(u_idx: int) -> float:
        qs0_pred_u, qs1_pred_seq_u, qo_pred_u = rollout_predictive_states_under_path_tau(
            qs0_grid,
            qs1_grid,
            level_top,
            level0,
            u_idx,
            tau,
        )
        risk_u = compute_risk_term(qo_pred_u, C)                      # (tau,)
        ambiguity_u = compute_ambiguity_term(qs0_pred_u, level0)      # (tau,)
        epistemic_u = compute_epistemic_term_from_qs1_pred(qs1_pred_seq_u)  # (tau,)

        G_seq = risk_u + ambiguity_u - epistemic_u  # (tau,)
        # Aggregate over horizon; here we simply sum (no discounting).
        G_u = G_seq.sum()
        return G_u

    G_u_vec = jax.vmap(G_for_path)(jnp.arange(U))  # (U,)

    # Broadcast the same G_u over all current times t in this simple implementation.
    G_tu = jnp.broadcast_to(G_u_vec[None, :], (T, U))  # (T, U)
    return G_tu


def update_path_posterior_from_G(
    level_top: LorenzLevel,
    G_tu: jnp.ndarray,
    gamma: float = 16.0,
    num_iter: int = 2,
) -> jnp.ndarray:
    """
    Update the path posterior q(u_t) using G_tu and the path transition
    model at the top level:

        p(u_t | G) ∝ exp(-gamma * G_tu)

    Then perform VMP over the path chain using B_paths and E_paths.

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

    if B_paths is None or E_paths is None:
        raise ValueError("Top level must define B_paths and E_paths for path inference.")

    T, U_check = G_tu.shape
    assert U_check == U, "G_tu shape inconsistent with num_paths."

    # Prior over u_t based on EFE: p(u_t) ∝ exp(-gamma * G_tu)
    def prior_u_t(G_t: jnp.ndarray) -> jnp.ndarray:
        logits = -gamma * G_t  # (U,)
        logits = logits - logits.max()
        exp_logits = jnp.exp(logits)
        return exp_logits / (exp_logits.sum() + 1e-8)

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
        ln_qu = ln_qu - ln_qu.max(axis=1, keepdims=True)
        qu_new = jnp.exp(ln_qu)
        qu_new = qu_new / (qu_new.sum(axis=1, keepdims=True) + 1e-8)
        return qu_new

    def body_fun(_, qu_):
        return update_once(qu_)

    qu_final = jax.lax.fori_loop(0, num_iter, body_fun, qu)
    return qu_final
