"""
ParamEstimationPipeline/nuts_hmc.py

No-U-Turn Sampler (NUTS) with configurable mass matrix preconditioning.

Extends MassMatrixHMC, replacing the fixed leapfrog loop with the NUTS
doubling-tree algorithm (Hoffman & Gelman 2014, Algorithm 3).

Key differences from fixed-step HMC
-------------------------------------
- num_leapfrog is not required: NUTS determines trajectory length by
  doubling until the No-U-Turn criterion fires or max_tree_depth is hit.
- Dual averaging target is the mean acceptance probability over all
  leapfrog steps in the tree (not a single MH alpha).

CCPM
----
All leapfrog evaluations within one NUTS step share the same proposal_seed
and fixed_initial_state, keeping the stochastic Hamiltonian consistent
across the entire tree.

Slice variable (physics sign convention)
-----------------------------------------
H(θ, r) = −log p(θ) + T(r)   (Hamiltonian, high = low probability)
u ~ Uniform(0, exp(−H₀))
threshold = H₀ − log(Uniform(0,1))  ≥ H₀
Leaf (θ', r') is valid iff H(θ', r') ≤ threshold.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, Tuple

from ParamEstimationPipeline.mass_matrix_hmc import MassMatrixHMC

DTYPE     = tf.float32
DELTA_MAX = 1000.0   # energy ceiling for numerical stability in slice check


class NUTSMassMatrixHMC(MassMatrixHMC):
    """
    NUTS with mass matrix preconditioning.

    Parameters (additional to MassMatrixHMC)
    -----------------------------------------
    max_tree_depth : int
        Maximum doubling depth.  2^max_tree_depth leapfrog steps maximum.
        Typical runs stop at depth 3-6; 10 is a safe ceiling.
    """

    def __init__(
        self,
        ssm_builder,
        filter_module,
        prior_log_prob_fn,
        mass_diag         = None,
        mass_L            = None,
        riemannian        : bool  = False,
        riemannian_lambda : float = 0.01,
        max_tree_depth    : int   = 10,
    ) -> None:
        super().__init__(ssm_builder, filter_module, prior_log_prob_fn,
                         mass_diag, mass_L, riemannian, riemannian_lambda)
        self.max_tree_depth = max_tree_depth

    # ------------------------------------------------------------------
    # Single Störmer-Verlet step
    # ------------------------------------------------------------------

    def _leapfrog_single(
        self,
        theta:        tf.Tensor,
        r:            tf.Tensor,
        grad:         tf.Tensor,
        step_size:    float,
        v:            int,
        observations: tf.Tensor,
        fixed_state,
        seed:         int,
        local_m:      Optional[tf.Tensor],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, float]:
        """
        One complete leapfrog step in direction v (+1 forward, −1 backward).

        Input  r   : full momentum at theta (both half-steps applied)
        Output r_new: full momentum at theta_new (both half-steps applied)

        Returns (theta_new, r_new, lp_new, grad_new, H_new).
        """
        eps_v    = v * step_size
        r_half   = r + (eps_v / 2.0) * grad
        theta_new = theta + eps_v * self._mass_inv_p(r_half, local_m)

        tf.random.set_seed(seed)
        lp_new, grad_new = self._compute_log_prob_and_grad(
            theta_new, observations, fixed_state)

        r_new = r_half + (eps_v / 2.0) * grad_new
        H_new = float(-lp_new + self._kinetic_energy(r_new, local_m))
        if not np.isfinite(H_new):
            H_new = np.inf

        return theta_new, r_new, lp_new, grad_new, H_new

    # ------------------------------------------------------------------
    # Recursive tree builder (Hoffman & Gelman 2014, Algorithm 3)
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        theta:        tf.Tensor,
        r:            tf.Tensor,
        grad:         tf.Tensor,
        threshold:    float,
        v:            int,
        j:            int,
        step_size:    float,
        observations: tf.Tensor,
        fixed_state,
        seed:         int,
        H0:           float,
        local_m:      Optional[tf.Tensor],
    ):
        """
        Build a NUTS subtree of depth j rooted at (theta, r).

        Returns
        -------
        tm, rm, gm          — backward (leftmost) leaf (theta, r, grad)
        tp, rp, gp          — forward (rightmost) leaf
        theta_prop, lp_prop, grad_prop — proposed sample from this subtree
        n_prop   : int   — valid leaves in subtree
        s        : int   — 1 = continue growing, 0 = U-turn / out-of-slice
        sum_alpha: float — sum of min(1, exp(H0−Hi)) over leaves
        n_alpha  : int   — leaf count (denominator for average alpha)
        """
        if j == 0:
            # ── Base case: single leapfrog step ───────────────────────
            tp, rp, lp_p, gp, H_p = self._leapfrog_single(
                theta, r, grad, step_size, v,
                observations, fixed_state, seed, local_m)

            n_p   = int(H_p <= threshold)
            s_p   = int(H_p <= threshold + DELTA_MAX)
            alpha = min(1.0, float(np.exp(H0 - H_p))) if np.isfinite(H_p) else 0.0

            return (tp, rp, gp,          # tm/rm/gm (same as forward leaf)
                    tp, rp, gp,          # tp/rp/gp
                    tp, lp_p, gp,        # proposal
                    n_p, s_p, alpha, 1)

        # ── Recursive case ────────────────────────────────────────────
        (tm, rm, gm, tp, rp, gp,
         theta_p, lp_p, grad_p,
         n_p, s_p, sum_a, n_a) = self._build_tree(
            theta, r, grad, threshold, v, j - 1, step_size,
            observations, fixed_state, seed, H0, local_m)

        if s_p:
            if v == -1:
                (tm, rm, gm, _, _, _,
                 theta_pp, lp_pp, grad_pp,
                 n_pp, s_pp, sum_a2, n_a2) = self._build_tree(
                    tm, rm, gm, threshold, v, j - 1, step_size,
                    observations, fixed_state, seed, H0, local_m)
            else:
                (_, _, _, tp, rp, gp,
                 theta_pp, lp_pp, grad_pp,
                 n_pp, s_pp, sum_a2, n_a2) = self._build_tree(
                    tp, rp, gp, threshold, v, j - 1, step_size,
                    observations, fixed_state, seed, H0, local_m)

            if n_pp > 0:
                if np.random.uniform() < float(n_pp) / float(n_p + n_pp):
                    theta_p, lp_p, grad_p = theta_pp, lp_pp, grad_pp

            n_p    += n_pp
            sum_a  += sum_a2
            n_a    += n_a2

            diff    = tp - tm
            no_turn = (float(tf.reduce_sum(diff * rm)) >= 0.0 and
                       float(tf.reduce_sum(diff * rp)) >= 0.0)
            s_p = int(s_pp and no_turn)

        return (tm, rm, gm, tp, rp, gp,
                theta_p, lp_p, grad_p,
                n_p, s_p, sum_a, n_a)

    # ------------------------------------------------------------------
    # NUTS proposal
    # ------------------------------------------------------------------

    def _nuts_step(
        self,
        current_theta: tf.Tensor,
        current_lp:    tf.Tensor,
        current_grad:  tf.Tensor,
        observations:  tf.Tensor,
        step_size:     float,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, bool, float]:
        """
        One NUTS proposal.

        Returns (new_theta, new_lp, new_grad, moved, avg_alpha)
        where avg_alpha is the mean min(1, exp(H0−Hi)) over all tree leaves,
        used by dual averaging for step-size adaptation.
        """
        d = tf.shape(current_theta)[0]

        proposal_seed = int(np.random.randint(0, 2**31))
        fixed_state   = self.filter_module.initialize_state()

        tf.random.set_seed(proposal_seed)
        lp_c, grad_c = self._compute_log_prob_and_grad(
            current_theta, observations, fixed_state)

        local_m = (tf.square(grad_c) + self.riemannian_lambda
                   if self.riemannian else None)

        r0 = self._sample_momentum(d, local_m)
        H0 = float(-lp_c + self._kinetic_energy(r0, local_m))

        if not np.isfinite(H0):
            return current_theta, lp_c, grad_c, False, 0.0

        threshold = H0 - float(np.log(np.random.uniform()))  # H0 + |log v| ≥ H0

        tm = tp = current_theta
        rm = rp = r0
        gm = gp = grad_c

        theta_p   = current_theta
        lp_p      = lp_c
        grad_p    = grad_c
        n         = 1
        s         = 1
        sum_alpha = 0.0
        n_alpha   = 0
        moved     = False

        for j in range(self.max_tree_depth):
            if not s:
                break
            v = int(np.random.choice([-1, 1]))

            if v == -1:
                (tm, rm, gm, _, _, _,
                 theta_pp, lp_pp, grad_pp,
                 n_pp, s_pp, a_pp, na_pp) = self._build_tree(
                    tm, rm, gm, threshold, v, j, step_size,
                    observations, fixed_state, proposal_seed, H0, local_m)
            else:
                (_, _, _, tp, rp, gp,
                 theta_pp, lp_pp, grad_pp,
                 n_pp, s_pp, a_pp, na_pp) = self._build_tree(
                    tp, rp, gp, threshold, v, j, step_size,
                    observations, fixed_state, proposal_seed, H0, local_m)

            if s_pp and n_pp > 0:
                if np.random.uniform() < float(n_pp) / float(n):
                    theta_p = theta_pp
                    lp_p    = lp_pp
                    grad_p  = grad_pp
                    moved   = True

            n         += n_pp
            sum_alpha += a_pp
            n_alpha   += na_pp

            diff    = tp - tm
            no_turn = (float(tf.reduce_sum(diff * rm)) >= 0.0 and
                       float(tf.reduce_sum(diff * rp)) >= 0.0)
            s = int(s_pp and no_turn)

        avg_alpha = sum_alpha / max(n_alpha, 1)
        return theta_p, lp_p, grad_p, moved, avg_alpha

    # ------------------------------------------------------------------
    # Route _hmc_step through NUTS (num_leapfrog is ignored)
    # ------------------------------------------------------------------

    def _hmc_step(
        self,
        current_theta: tf.Tensor,
        current_lp:    tf.Tensor,
        current_grad:  tf.Tensor,
        observations:  tf.Tensor,
        step_size:     float,
        num_leapfrog:  int,       # unused — NUTS sets trajectory length
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, bool, float]:
        return self._nuts_step(
            current_theta, current_lp, current_grad, observations, step_size)
