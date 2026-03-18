# lorenz_efe.py
"""
Expected free energy (EFE) computations and path inference for the Lorenz RGM.

This module provides:
- Functions to roll out predictive state and observation distributions under
  path-dependent dynamics at the top level.
- Computation of expected free energy G(t,u) over a finite horizon tau, using
  risk, ambiguity, and epistemic components.
- Variational message passing over the path chain u_t given G(t,u) and
  path dynamics C_paths, E_paths.

CONVENTIONS (consistent with lorenz_model.py, lorenz_learning.py,
lorenz_inference.py):

- For a level with S states and U paths:

    B_states_paths[s_next, s, u] = P(s_next | s, u)

  with shape (S, S, U).

- Path transitions at that level:

    C_paths[u_next, u] = P(u_{t+1} = u_next | u_t = u)
    E_paths[u] = P(u_1 = u).

- We typically attach path dynamics and B_states_paths to the highest
  spatial level in the Lorenz example, but the functions here are
  written in a level-agnostic way.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from .lorenz_model import LorenzLevel


# -----------------------------------------------------------------------------
# 1. Predictive rollouts under path-dependent dynamics
# -----------------------------------------------------------------------------

def rollout_predictive_states_under_path_tau(
    qs0_start: jnp.ndarray,
    qs_top_start: jnp.ndarray,
    level_top: LorenzLevel,
    level0: LorenzLevel,
    u_idx: int,
    tau: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Roll out predictive state and observation distributions under a fixed
    path u over a horizon tau, starting from given marginals at a single
    time point (not from the end of the sequence).

    Args:
        qs0_start: (S0,) starting marginal over level-0 states
        qs_top_start: (S_top,) starting marginal over top-level states
        level_top: top-level LorenzLevel (with B_states_paths)
        level0: lowest-level LorenzLevel (with A)
        u_idx: index of the path u ∈ {0,...,U-1}
        tau: planning horizon

    Returns:
        qs0_pred_u: (tau, S0) predictive distributions over level-0 states
        qs_top_pred_u: (tau, S_top) predictive distributions over top-level states
        qo_pred_u: (tau, O0) predictive observations at lowest level
    """
    # Lowest level emission
    A0 = level0.A  # (S0, O0)
    S0 = A0.shape[0]

    # Top level path-dependent transitions
    B_states_paths = level_top.B_states_paths  # (S_top, S_top, U) or None
    S_top = level_top.S

    if B_states_paths is None:
        raise ValueError(
            "rollout_predictive_states_under_path_tau called with "
            "level_top.B_states_paths = None."
        )

    # Transition kernel for this path at the top level
    if B_states_paths.shape[2] > 1:
        B_top_u = B_states_paths[:, :, u_idx]  # (S_top, S_top)
    else:
        B_top_u = B_states_paths[:, :, 0]  # (S_top, S_top)

    # For level 0 we do not have explicit path-conditioned dynamics here;
    # we approximate with an identity / persistence kernel.
    B0 = jnp.eye(S0, dtype=jnp.float32)  # (S0, S0)

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

    init_carry = (qs0_start, qs_top_start)
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
    C: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the risk term over a horizon, given predictive observations
    and preferences C over outcomes.

    Risk is approximated as expected negative log preference:

        risk_k ≈ E_{q(o_{t+k}|u)}[ -log C(o_{t+k}) ]

    Args:
        qo_pred: (tau, O0) predictive observation distributions
        C: (O0,) preference distribution

    Returns:
        risk: (tau,) risk per horizon step
    """
    eps = 1e-16
    log_C = jnp.log(jnp.clip(C, eps, 1.0))  # (O0,)
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
# 3. Expected free energy over paths (time-varying G_tu)
# -----------------------------------------------------------------------------

def compute_expected_free_energy_paths(
    level_top: LorenzLevel,
    level0: LorenzLevel,
    qs_top_grid: jnp.ndarray,
    qs0_grid: jnp.ndarray,
    C: jnp.ndarray,
    tau: int,
    U: int,
) -> jnp.ndarray:
    """
    Compute expected free energy G_tu (per time t and path u) at
    the top level, using a risk–ambiguity–epistemic decomposition over
    a multi-step horizon tau.

    New behaviour (Stage 1):
      - G(t, u) is computed separately for each time t.
      - For each t, we take the spatially-averaged marginals at time t,
        roll forward tau steps under each path, and aggregate risk,
        ambiguity, and epistemic terms over the horizon.
      - Near the end of the sequence, we clamp the horizon so that
        t + tau does not exceed T.

    Args:
        level_top: top-level LorenzLevel (with B_states_paths)
        level0: lowest-level LorenzLevel (with A)
        qs_top_grid: (T, H_top, W_top, S_top) top-level posteriors (data)
        qs0_grid: (T, H0, W0, S0) lowest-level posteriors (data)
        C: (O0,) preference distribution over lowest-level outcomes
        tau: integer planning horizon (number of predictive steps)
        U: number of paths (concrete int, passed from outside JIT)

    Returns:
        G_tu: (T, U) expected free energy per time and path
    """
    if U == 0:
        raise ValueError("compute_expected_free_energy_paths called with U=0")

    T = qs_top_grid.shape[0]

    B_states_paths = level_top.B_states_paths

    # If no path-specific dynamics, fall back to scalar G_t per time, then broadcast over U.
    if B_states_paths is None or B_states_paths.shape[2] <= 1:
        # Approximate predictive distributions by current posteriors
        qs0_marg = qs0_grid.mean(axis=(1, 2))  # (T, S0)
        qs0_marg = qs0_marg / (qs0_marg.sum(axis=1, keepdims=True) + 1e-8)

        A0 = level0.A
        qo_marg = jax.vmap(lambda q: A0.T @ q)(qs0_marg)  # (T, O0)
        qo_marg = qo_marg / (qo_marg.sum(axis=1, keepdims=True) + 1e-8)

        risk = compute_risk_term(qo_marg, C)  # (T,)
        ambiguity = compute_ambiguity_term(qs0_marg, level0)  # (T,)

        qs_top_marg = qs_top_grid.mean(axis=(1, 2))  # (T, S_top)
        epistemic = compute_epistemic_term_from_qs_top_pred(qs_top_marg)  # (T,)

        G_t = risk + ambiguity - epistemic  # (T,)
        return jnp.broadcast_to(G_t[:, None], (T, U))

    # General case: path-specific B_states_paths and time-varying G(t,u).
    # We use a per-time, per-path rollout.
    qs0_marg_all = qs0_grid.mean(axis=(1, 2))  # (T, S0)
    qs0_marg_all = qs0_marg_all / (qs0_marg_all.sum(axis=1, keepdims=True) + 1e-8)

    qs_top_marg_all = qs_top_grid.mean(axis=(1, 2))  # (T, S_top)
    qs_top_marg_all = qs_top_marg_all / (qs_top_marg_all.sum(axis=1, keepdims=True) + 1e-8)

    def G_at_time(t: int) -> jnp.ndarray:
        """
        Compute G_t(u) for a fixed time index t, for all paths u.
        We allow a truncated horizon near the end of the sequence.
        """
        # Effective horizon from this t
        remaining = T - 1 - t
        tau_eff = jnp.minimum(tau, remaining + 1)  # at least 1 step if remaining >= 0

        qs0_start = qs0_marg_all[t]      # (S0,)
        qs_top_start = qs_top_marg_all[t]  # (S_top,)

        def G_for_path_at_time(u_idx: int) -> float:
            qs0_pred_u, qs_top_pred_u, qo_pred_u = rollout_predictive_states_under_path_tau(
                qs0_start,
                qs_top_start,
                level_top,
                level0,
                u_idx,
                tau_eff,
            )

            risk_u = compute_risk_term(qo_pred_u, C)  # (tau_eff,)
            ambiguity_u = compute_ambiguity_term(qs0_pred_u, level0)  # (tau_eff,)
            epistemic_u = compute_epistemic_term_from_qs_top_pred(
                qs_top_pred_u
            )  # (tau_eff,)

            G_seq = risk_u + ambiguity_u - epistemic_u  # (tau_eff,)
            G_u = G_seq.sum()
            return G_u

        G_u_vec = jax.vmap(G_for_path_at_time)(jnp.arange(U))  # (U,)
        return G_u_vec

    # Vectorize G_at_time over t
    G_tu = jax.vmap(G_at_time)(jnp.arange(T))  # (T, U)
    return G_tu


# -----------------------------------------------------------------------------
# 4. Path posterior update given G_tu and path dynamics
# -----------------------------------------------------------------------------

def update_path_posterior_from_G(
    level_top: LorenzLevel,
    G_tu: jnp.ndarray,
    U: int,
    gamma: float = 16.0,
    num_iter: int = 2,
) -> jnp.ndarray:
    """
    Update the path posterior q(u_t) using G_tu and the path transition
    model at the top level:

        p(u_t | G) ∝ exp(-gamma * G_tu)

    Then perform VMP over the path chain using C_paths and E_paths.

    Args:
        level_top: top-level LorenzLevel with C_paths, E_paths
        G_tu: (T, U) expected free energy per time and path
        U: number of paths (concrete int, passed from outside JIT)
        gamma: precision over expected free energy
        num_iter: number of VMP iterations over the path chain

    Returns:
        qu_t: (T, U) posterior over paths at each time
    """
    if U == 0:
        raise ValueError("update_path_posterior_from_G called with U=0")

    C_paths = level_top.C_paths  # (U, U)
    E_paths = level_top.E_paths  # (U,)

    if C_paths is None or E_paths is None:
        raise ValueError("Top level must define C_paths and E_paths for path inference.")

    T, U_check = G_tu.shape
    assert U_check == U, "G_tu shape inconsistent with U."

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
