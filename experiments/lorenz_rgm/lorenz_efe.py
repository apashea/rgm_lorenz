# lorenz_efe.py
"""
Expected free energy (EFE) computations and path inference for the Lorenz RGM.

This module provides:
- Functions to roll out predictive state and observation distributions under
  path-dependent dynamics at a chosen level (typically the top level).
- Computation of expected free energy G(t,u) over a finite horizon tau, using
  risk, ambiguity, and epistemic components.
- Variational message passing over the path chain u_t given G(t,u) and
  path dynamics C_paths, E_paths.

CONVENTIONS (consistent with lorenz_model.py, lorenz_learning.py,
lorenz_inference.py):

- For a level with S states and U paths:

    B_states_paths[s_next, s, u] = P(s_next | s, u)

  with shape (S, S, U).

- Path transitions:

    C_paths[u_next, u] = P(u_next | u)
    E_paths[u]        = P(u_1 = u)

- Lowest-level emissions:

    A0[s, o] = P(o | s)  with shape (S0, O0).
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from .lorenz_model import LorenzLevel

# -----------------------------------------------------------------------------
# 1. Predictive rollouts under path-dependent dynamics
# -----------------------------------------------------------------------------

def rollout_predictive_states_under_path_tau(
    qs0_grid: jnp.ndarray,
    qs_top_grid: jnp.ndarray,
    level_top: LorenzLevel,
    level0: LorenzLevel,
    u_idx: int,
    tau: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Roll out predictive state and observation distributions under a fixed
    path u over a horizon tau.

    For the Lorenz example:

    - Level 0 is the patch level (with A0 and path-dependent transitions
      encoded in B_states_paths^0 with U0=1).
    - level_top is the highest spatial level with path-dependent transitions
      B_states_paths[s_next,s,u].

    We approximate q(s_{t+k} | u) and q(o_{t+k} | u) by:
    - Using the latest posterior at t = T-1 as the starting point.
    - Propagating q(s^top) under B_states_paths[:,:,u_idx].
    - Propagating q(s^0) under B0 (the single-path slice of B_states_paths^0)
      using the mean over spatial sites.

    Args:
      qs0_grid: (T, H0, W0, S0) posterior over level-0 states
      qs_top_grid: (T, H_top, W_top, S_top) posterior over top-level states
      level_top: LorenzLevel for the chosen level (with B_states_paths)
      level0: lowest-level LorenzLevel (with A and B_states_paths)
      u_idx: index of the path u ∈ {0,...,U-1}
      tau: planning horizon

    Returns:
      qs0_pred_u:  (tau, S0)   predictive distributions over level-0 states
      qs_top_pred_u: (tau, S_top) predictive distributions over top-level states
      qo_pred_u:  (tau, O0)   predictive observations at lowest level
    """
    # Lowest level parameters
    if level0.B_states_paths is None:
        raise ValueError("level0 must define B_states_paths for rollout.")
    B0 = level0.B_states_paths[:, :, 0]  # (S0, S0), assume U0=1
    A0 = level0.A                        # (S0, O0)

    # Top level path-dependent transitions
    B_states_paths_top = level_top.B_states_paths  # (S_top, S_top, U) or None
    if B_states_paths_top is None:
        raise ValueError("level_top must define B_states_paths for rollout.")
    S_top = level_top.S

    # Start from last posterior time point
    qs0_last = qs0_grid[-1]     # (H0, W0, S0)
    qs_top_last = qs_top_grid[-1]  # (H_top, W_top, S_top)

    # Aggregate over space to get global marginals
    qs0_global = qs0_last.mean(axis=(0, 1))  # (S0,)
    qs0_global = qs0_global / (qs0_global.sum() + 1e-8)

    qs_top_global = qs_top_last.mean(axis=(0, 1))  # (S_top,)
    qs_top_global = qs_top_global / (qs_top_global.sum() + 1e-8)

    # Determine top-level transition kernel for this path
    U = B_states_paths_top.shape[2]
    if U > 1:
        B_top_u = B_states_paths_top[:, :, u_idx]  # (S_top, S_top)
    else:
        B_top_u = B_states_paths_top[:, :, 0]      # (S_top, S_top)

    def step(carry, _):
        qs0_curr, qs_top_curr = carry  # (S0,), (S_top,)

        # Top level: q_{t+1}^top = B_top_u @ q_t^top
        qs_top_next = B_top_u @ qs_top_curr
        qs_top_next = qs_top_next / (qs_top_next.sum() + 1e-8)

        # Lowest level: q_{t+1}^0 = B0 @ q_t^0
        qs0_next = B0 @ qs0_curr
        qs0_next = qs0_next / (qs0_next.sum() + 1e-8)

        # Predictive observation at lowest level
        qo_next = A0.T @ qs0_next  # (O0,)
        qo_next = qo_next / (qo_next.sum() + 1e-8)

        new_carry = (qs0_next, qs_top_next)
        out = (qs0_next, qs_top_next, qo_next)
        return new_carry, out

    init_carry = (qs0_global, qs_top_global)
    _, (qs0_seq, qs_top_seq, qo_seq) = jax.lax.scan(
        step,
        init_carry,
        xs=None,
        length=tau,
    )
    # qs0_seq: (tau, S0), qs_top_seq: (tau, S_top), qo_seq: (tau, O0)
    return qs0_seq, qs_top_seq, qo_seq


# -----------------------------------------------------------------------------
# 2. Risk, ambiguity, epistemic terms
# -----------------------------------------------------------------------------

def compute_risk_term(
    qo_pred: jnp.ndarray,
    C_pref: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the risk term over a horizon, given predictive observations
    and preferences C over outcomes.

    Risk is approximated as expected negative log preference:

      risk_k ≈ E_{q(o_{t+k}|u)}[ -log C(o_{t+k}) ]

    Args:
      qo_pred: (tau, O0) predictive observation distributions
      C_pref: (O0,) preference distribution

    Returns:
      risk: (tau,) risk per horizon step
    """
    eps = 1e-16
    log_C = jnp.log(jnp.clip(C_pref, eps, 1.0))  # (O0,)
    risk = -jnp.sum(qo_pred * log_C[None, :], axis=1)  # (tau,)
    return risk


def compute_ambiguity_term(
    qs0_pred: jnp.ndarray,
    level0: LorenzLevel,
) -> jnp.ndarray:
    """
    Compute an ambiguity proxy over the horizon using the lowest-level
    emission model A0.

    Ambiguity is approximated as expected conditional entropy H[o|s]:

      ambiguity_k ≈ E_{q(s_{t+k}|u)}[ H[p(o|s_{t+k})] ]

    Args:
      qs0_pred: (tau, S0) predictive states at lowest level
      level0: LorenzLevel with A

    Returns:
      ambiguity: (tau,) ambiguity per horizon step
    """
    A0 = level0.A  # (S0, O0)
    eps = 1e-16

    # H[o|s] for each s
    H_o_given_s = -jnp.sum(
        A0 * jnp.log(jnp.clip(A0, eps, 1.0)),
        axis=1,
    )  # (S0,)

    ambiguity = jnp.sum(
        qs0_pred * H_o_given_s[None, :],
        axis=1,
    )  # (tau,)
    return ambiguity


def compute_epistemic_term_from_qs_top_pred(
    qs_top_pred: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute a simple epistemic term from the entropy of top-level
    predictive states over the horizon.

    We approximate epistemic value as the negative entropy of
    the predictive state distribution:

      epistemic_k ≈ -H[q(s^top_{t+k}|u)]

    so that higher state uncertainty yields lower epistemic "value".

    Args:
      qs_top_pred: (tau, S_top) predictive top-level states

    Returns:
      epistemic: (tau,) epistemic term per horizon step
    """
    eps = 1e-16
    log_q = jnp.log(jnp.clip(qs_top_pred, eps, 1.0))
    entropy = -jnp.sum(qs_top_pred * log_q, axis=1)  # (tau,)
    epistemic = -entropy
    return epistemic


# -----------------------------------------------------------------------------
# 3. Expected free energy over paths
# -----------------------------------------------------------------------------

def compute_expected_free_energy_paths(
    level_top: LorenzLevel,
    level0: LorenzLevel,
    qs_top_grid: jnp.ndarray,
    qs0_grid: jnp.ndarray,
    C_pref: jnp.ndarray,
    tau: int = 3,
) -> jnp.ndarray:
    """
    Compute expected free energy G_tu (per current time t and path u) at
    a chosen level (typically the top level), using a risk–ambiguity–
    epistemic decomposition over a multi-step horizon tau.

    In this implementation, we approximate G(t, u) by:

      G(t, u) ≈ sum_{k=1..tau} [ risk_{t+k|u} + ambiguity_{t+k|u}
                                - epistemic_{t+k|u} ]

    where predictive distributions q(s, o | u) over the horizon are
    generated by rolling forward under B_states_paths[:,:,u].

    For simplicity and computational tractability in the Lorenz example, we:
    - Use the last observed/posterior time point as the "current" time.
    - Roll forward tau steps from that time for each path u.
    - Treat this G(t, u) as time-constant across t (broadcast).

    Args:
      level_top: LorenzLevel (with num_paths > 0 and B_states_paths)
      level0: lowest-level LorenzLevel (with A and B_states_paths)
      qs_top_grid: (T, H_top, W_top, S_top) posteriors at the chosen level
      qs0_grid: (T, H0, W0, S0) lowest-level posteriors (data)
      C_pref: (O0,) preference distribution over lowest-level outcomes
      tau: integer planning horizon (number of predictive steps)

    Returns:
      G_tu: (T, U) expected free energy per time and path
    """
    U = level_top.num_paths
    if U == 0:
        raise ValueError("compute_expected_free_energy_paths called with num_paths=0")

    T = qs_top_grid.shape[0]

    B_states_paths_top = level_top.B_states_paths
    if B_states_paths_top is None or B_states_paths_top.shape[2] <= 1:
        # No path-specific dynamics: approximate predictive distributions
        # by current posteriors (no multi-step lookahead)
        qs0_pred = qs0_grid.mean(axis=(1, 2))  # (T, S0)
        qs0_pred = qs0_pred / (qs0_pred.sum(axis=1, keepdims=True) + 1e-8)

        A0 = level0.A
        qo_pred = jax.vmap(lambda q: A0.T @ q)(qs0_pred)  # (T, O0)
        qo_pred = qo_pred / (qo_pred.sum(axis=1, keepdims=True) + 1e-8)

        risk = compute_risk_term(qo_pred, C_pref)              # (T,)
        ambiguity = compute_ambiguity_term(qs0_pred, level0)   # (T,)
        qs_top_mean = qs_top_grid.mean(axis=(1, 2))            # (T, S_top)
        epistemic = compute_epistemic_term_from_qs_top_pred(qs_top_mean)  # (T,)

        G_t = risk + ambiguity - epistemic
        return jnp.broadcast_to(G_t[:, None], (T, U))

    # Path-specific G_u via tau-step rollout
    def G_for_path(u_idx: int) -> float:
        qs0_pred_u, qs_top_pred_u, qo_pred_u = rollout_predictive_states_under_path_tau(
            qs0_grid,
            qs_top_grid,
            level_top,
            level0,
            u_idx,
            tau,
        )

        risk_u = compute_risk_term(qo_pred_u, C_pref)              # (tau,)
        ambiguity_u = compute_ambiguity_term(qs0_pred_u, level0)   # (tau,)
        epistemic_u = compute_epistemic_term_from_qs_top_pred(
            qs_top_pred_u
        )  # (tau,)

        G_seq = risk_u + ambiguity_u - epistemic_u  # (tau,)
        G_u = G_seq.sum()
        return G_u

    G_u_vec = jax.vmap(G_for_path)(jnp.arange(U))  # (U,)

    # Broadcast the same G_u over all current times t in this simple implementation.
    G_tu = jnp.broadcast_to(G_u_vec[None, :], (T, U))  # (T, U)
    return G_tu


# -----------------------------------------------------------------------------
# 4. Path posterior update given G_tu and path dynamics
# -----------------------------------------------------------------------------

def update_path_posterior_from_G(
    level: LorenzLevel,
    G_tu: jnp.ndarray,
    gamma: float = 16.0,
    num_iter: int = 2,
) -> jnp.ndarray:
    """
    Update the path posterior q(u_t) using G_tu and the path transition
    model at a given level:

      p(u_t | G) ∝ exp(-gamma * G_tu)

    Then perform VMP over the path chain using C_paths and E_paths.

    Args:
      level: LorenzLevel with C_paths, E_paths, num_paths > 0
      G_tu: (T, U) expected free energy per time and path
      gamma: precision over expected free energy
      num_iter: number of VMP iterations over the path chain

    Returns:
      qu_t: (T, U) posterior over paths at each time
    """
    U = level.num_paths
    if U == 0:
        raise ValueError("update_path_posterior_from_G called with num_paths=0")

    C_paths = level.C_paths  # (U, U)
    E_paths = level.E_paths  # (U,)

    if C_paths is None or E_paths is None:
        raise ValueError("Level must define C_paths and E_paths for path inference.")

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
            msg = jnp.log(jnp.clip(C_paths @ prev_q, 1e-16))
            msgs.append(msg)
            prev_q = qu_[t]
        return jnp.stack(msgs, axis=0)

    def backward_messages(qu_):
        msgs = []
        next_q = qu_[-1]
        msgs.append(jnp.zeros_like(next_q))
        for t in range(T - 2, -1, -1):
            msg = jnp.log(jnp.clip(C_paths.T @ next_q, 1e-16))
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
