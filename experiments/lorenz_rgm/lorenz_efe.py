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

Crucially, transitions among top-level states now depend on paths u
via a path-dependent transition tensor B_states_paths[u, s', s].
Expected free energy is therefore computed per-path by rolling forward
predictive states under each path-specific B^{(u)}.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .lorenz_model import LorenzLevel
from . import maths as rgm_maths  # adapted from pymdp.jax.maths


# -----------------------------------------------------------------------------
# 1. Predictive distributions at the lowest level (given predictive qs0)
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
    T, S0 = qs0_pred.shape

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
    S0, O = A0.shape

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
# 4. Path-specific predictive rollouts
# -----------------------------------------------------------------------------

def rollout_predictive_states_under_path(
    qs0_grid: jnp.ndarray,
    qs1_grid: jnp.ndarray,
    level_top: LorenzLevel,
    level0: LorenzLevel,
    u_idx: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    For a given path index u, construct simple predictive trajectories
    of top-level and lowest-level states and observations, using:

      - A fixed path slice B_states_paths[u] for top-level transitions.
      - The current posteriors as proxies for initial predictive states.

    This is not a full multi-step planning rollout; it stabilizes the
    EFE computation by using the inferred posteriors as predictive
    marginals under the dynamics specified by path u.

    Args:
        qs0_grid: (T, H0, W0, S0) posterior over Level-0 states (data-driven)
        qs1_grid: (T, H1, W1, S1) posterior over Level-1 states (data-driven)
        level_top: top-level LorenzLevel with B_states_paths[u]
        level0: lowest-level LorenzLevel with A

        u_idx: integer index of the path (0 <= u_idx < U)

    Returns:
        qs0_pred_u: (T, S0) predictive lowest-level state marginals under path u
        qs1_pred_u_grid: (T, H1, W1, S1) predictive top-level state grid under u
        qo_pred_u: (T, O) predictive observations under path u
    """
    # Current posteriors as base predictive states
    qs0_pred_base = qs0_grid.mean(axis=(1, 2))  # (T, S0)
    qs0_pred_base = qs0_pred_base / (qs0_pred_base.sum(axis=1, keepdims=True) + 1e-8)

    # For top-level states, we start from spatial posteriors and apply path slice
    if level_top.B_states_paths is None:
        # No path dependence: use current qs1_grid as predictive
        qs1_pred_grid = qs1_grid
    else:
        B_u = level_top.B_states_paths[u_idx]  # (S1, S1)
        T, H1, W1, S1 = qs1_grid.shape

        def step_t(qs1_t: jnp.ndarray) -> jnp.ndarray:
            # qs1_t: (H1, W1, S1)
            def step_site(qs1_hw: jnp.ndarray) -> jnp.ndarray:
                q_next = qs1_hw @ B_u  # row vector times (S1, S1)
                q_next = q_next / (q_next.sum() + 1e-8)
                return q_next
            return jax.vmap(jax.vmap(step_site))(qs1_t)

        qs1_pred_grid = jax.vmap(step_t)(qs1_grid)  # (T, H1, W1, S1)

    qs0_pred_u = qs0_pred_base
    qo_pred_u = compute_predictive_obs_lowest(qs0_pred_u, level0)  # (T, O)

    return qs0_pred_u, qs1_pred_grid, qo_pred_u


# -----------------------------------------------------------------------------
# 5. Combined G(t, u) and path posterior update
# -----------------------------------------------------------------------------

def compute_expected_free_energy_paths(
    level_top: LorenzLevel,
    level0: LorenzLevel,
    qs1_grid: jnp.ndarray,
    qs0_grid: jnp.ndarray,
    C: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute expected free energy G_tu (per time and path) at the top level,
    using a risk–ambiguity–epistemic decomposition.

    Unlike the earlier simplified version, this function now conditions
    on path-specific dynamics via B_states_paths[u, s', s] at the top
    level. For each path u, we:

      1. Construct predictive state trajectories under B^{(u)}.
      2. Compute:
           risk_tu       = KL[q(o'_t | u) || C]
           ambiguity_tu  = E_{q(s'_t | u)} [ H[p(o'|s'_t)] ]
           epistemic_tu  = - H[ q(s^1_t | u) ] (proxy)
      3. Combine:
           G_tu = risk_tu + ambiguity_tu - epistemic_tu

    Args:
        level_top: top-level LorenzLevel (with num_paths > 0 and B_states_paths)
        level0: lowest-level LorenzLevel (with A)
        qs1_grid: (T, H1, W1, S1) top-level state posteriors (data)
        qs0_grid: (T, H0, W0, S0) lowest-level state posteriors (data)
        C: (O,) preference distribution over lowest-level outcomes

    Returns:
        G_tu: (T, U) expected free energy per time and path
    """
    U = level_top.num_paths
    if U == 0:
        raise ValueError("compute_expected_free_energy_paths called with num_paths=0")

    if level_top.B_states_paths is None:
        # Fall back: no path-specific dynamics; reuse scalar G_t for all u
        qs0_pred = qs0_grid.mean(axis=(1, 2))  # (T, S0)
        qs0_pred = qs0_pred / (qs0_pred.sum(axis=1, keepdims=True) + 1e-8)
        qo_pred = compute_predictive_obs_lowest(qs0_pred, level0)
        risk = compute_risk_term(qo_pred, C)
        ambiguity = compute_ambiguity_term(qs0_pred, level0)
        epistemic = compute_epistemic_term_from_qs1_pred(qs1_grid)
        G_t = risk + ambiguity - epistemic
        return jnp.broadcast_to(G_t[:, None], (G_t.shape[0], U))

    # Path-specific G_tu
    def G_for_path(u_idx: int) -> jnp.ndarray:
        qs0_pred_u, qs1_pred_grid_u, qo_pred_u = rollout_predictive_states_under_path(
            qs0_grid,
            qs1_grid,
            level_top,
            level0,
            u_idx,
        )
        risk_u = compute_risk_term(qo_pred_u, C)                  # (T,)
        ambiguity_u = compute_ambiguity_term(qs0_pred_u, level0)  # (T,)
        epistemic_u = compute_epistemic_term_from_qs1_pred(qs1_pred_grid_u)  # (T,)
        G_tu = risk_u + ambiguity_u - epistemic_u                 # (T,)
        return G_tu

    G_list = jax.vmap(G_for_path)(jnp.arange(U))  # (U, T)
    G_tu = jnp.swapaxes(G_list, 0, 1)             # (T, U)
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
        # stable softmax
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
