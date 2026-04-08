import time
import numpy as np
import tensorflow as tf
from typing import Any, Callable, Dict, Optional, Tuple

from StateSpaceModels.ssm_base import SSM
from FilterModules.filter_base import BaseFilter
from ParamEstimationPipeline.hmc_pipeline import HMC, DTYPE


class NUTSAdaptiveHMC(HMC):
    """
    Extends HMC with:
      - NUTS (No-U-Turn Sampler): automatically selects trajectory length by
        building a balanced binary tree and halting at the first U-turn.
      - Dual-averaging step size adaptation: drives the mean acceptance
        probability toward `target_accept_rate` during the warmup phase,
        then fixes epsilon to the smoothed estimate for the sampling phase.

    Reference:
        Hoffman, M. D., & Gelman, A. (2011).
        The No-U-Turn Sampler: Adaptively Setting Path Lengths in
        Hamiltonian Monte Carlo. arXiv:1111.4246.
    """

    # Maximum Hamiltonian error before a trajectory is declared divergent.
    _DELTA_MAX: float = 1000.0

    # Dual-averaging hyperparameters (Hoffman & Gelman §3.2 defaults).
    _DA_GAMMA: float = 0.05   # step-size adaptation scale
    _DA_T0:    float = 10.0   # iteration offset for stability at start
    _DA_KAPPA: float = 0.75   # polynomial decay rate of step-size memory

    def __init__(
        self,
        ssm_builder:        Callable[[tf.Tensor], SSM],
        filter_module:      BaseFilter,
        prior_log_prob_fn:  Callable[[tf.Tensor], tf.Tensor],
        target_accept_rate: float = 0.65,
        max_tree_depth:     int   = 10,
    ):
        super().__init__(ssm_builder, filter_module, prior_log_prob_fn)
        self.target_accept_rate = target_accept_rate
        self.max_tree_depth     = max_tree_depth

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _leapfrog(
        self,
        theta:        tf.Tensor,
        p:            tf.Tensor,
        grad:         tf.Tensor,
        epsilon:      float,
        observations: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Single leapfrog step.  epsilon may be negative (reverse direction)."""
        p_half    = p + 0.5 * epsilon * grad
        theta_new = theta + epsilon * p_half
        lp_new, grad_new = self._compute_log_prob_and_grad(theta_new, observations)
        p_new     = p_half + 0.5 * epsilon * grad_new
        return theta_new, p_new, lp_new, grad_new

    @staticmethod
    def _hamiltonian(log_prob: tf.Tensor, p: tf.Tensor) -> float:
        """H(θ, p) = log π(θ) − ½ pᵀp  (negative Hamiltonian energy)."""
        return float(log_prob) - 0.5 * float(tf.reduce_sum(p ** 2))

    @staticmethod
    def _no_u_turn(
        theta_minus: tf.Tensor, theta_plus:  tf.Tensor,
        p_minus:     tf.Tensor, p_plus:      tf.Tensor,
    ) -> bool:
        """True when the trajectory has NOT yet made a U-turn."""
        delta = theta_plus - theta_minus
        return (
            float(tf.reduce_sum(delta * p_minus)) >= 0.0 and
            float(tf.reduce_sum(delta * p_plus))  >= 0.0
        )

    def _find_reasonable_step_size(
        self, theta: tf.Tensor, observations: tf.Tensor
    ) -> float:
        """
        Heuristic initial step size (Algorithm 4, Hoffman & Gelman).
        Doubles/halves epsilon until the single-step acceptance probability
        crosses 0.5.

        Hard-capped at 50 iterations and terminates early on non-finite
        log_alpha to prevent infinite loops with stochastic likelihoods
        (e.g. particle-filter degeneracy returning -inf log_prob).
        When log_alpha = -inf and a = -1, the original while condition
        (-1)*(-inf) > (-1)*log(0.5) → +inf > 0.693 loops forever.
        """
        epsilon = 1.0
        log_prob, grad = self._compute_log_prob_and_grad(theta, observations)
        p  = tf.random.normal(shape=tf.shape(theta), dtype=DTYPE)
        H0 = self._hamiltonian(log_prob, p)

        if not np.isfinite(H0):
            return epsilon   # degenerate start; return default and let dual-averaging adapt

        _, p_new, lp_new, _ = self._leapfrog(theta, p, grad, epsilon, observations)
        H1        = self._hamiltonian(lp_new, p_new)
        log_alpha = H1 - H0 if np.isfinite(H1) else -np.inf

        a             = 1.0 if log_alpha > np.log(0.5) else -1.0
        prev_epsilon  = epsilon

        for _ in range(50):                                      # hard cap
            if not np.isfinite(log_alpha):                       # -inf or NaN breaks both branches
                epsilon = prev_epsilon                           # revert to last finite result
                break
            if not (a * log_alpha > a * np.log(0.5)):
                break
            prev_epsilon  = epsilon
            epsilon      *= 2.0 ** a
            _, p_new, lp_new, _ = self._leapfrog(theta, p, grad, epsilon, observations)
            H1        = self._hamiltonian(lp_new, p_new)
            log_alpha = H1 - H0 if np.isfinite(H1) else -np.inf

        return float(np.clip(epsilon, 1e-6, 1e3))

    # ------------------------------------------------------------------
    # NUTS tree builder (Algorithm 3, Hoffman & Gelman)
    # ------------------------------------------------------------------

    def _build_tree(
        self,
        theta:        tf.Tensor,
        p:            tf.Tensor,
        grad:         tf.Tensor,
        log_u:        float,
        direction:    float,      # +1 or -1
        depth:        int,
        epsilon:      float,
        H0:           float,
        observations: tf.Tensor,
    ) -> Tuple:
        """
        Recursively build a balanced binary tree of leapfrog trajectories.

        Returns
        -------
        theta_minus, p_minus, grad_minus : left endpoint of the trajectory
        theta_plus,  p_plus,  grad_plus  : right endpoint
        theta_prime, grad_prime, lp_prime : proposed sample from this subtree
        n      : number of valid (slice-accepted) states in the subtree
        s      : 1 if the subtree is valid (no U-turn, no divergence), else 0
        alpha  : cumulative sum of min(1, exp(H_leaf − H0)) over leaf nodes
        n_alpha: number of leaf nodes contributing to alpha
        """
        if depth == 0:
            # ---- Base case: one leapfrog step ----
            t2, p2, lp2, g2 = self._leapfrog(
                theta, p, grad, direction * epsilon, observations
            )
            H2    = self._hamiltonian(lp2, p2)
            n     = 1 if log_u <= H2 else 0
            s     = 1 if H2 - H0 > -self._DELTA_MAX else 0
            alpha = min(1.0, np.exp(np.clip(H2 - H0, -500.0, 0.0)))
            return t2, p2, g2, t2, p2, g2, t2, g2, lp2, n, s, alpha, 1

        # ---- Recursive case: first subtree ----
        (tm, pm, gm, tp, pp, gp,
         theta_p, grad_p, lp_p,
         n_p, s_p, alpha, n_alpha) = self._build_tree(
            theta, p, grad, log_u, direction, depth - 1, epsilon, H0, observations
        )

        if s_p:
            # ---- Second subtree (grown from the far endpoint) ----
            if direction < 0:
                (tm, pm, gm, _, _, _,
                 theta_pp, grad_pp, lp_pp,
                 n_pp, s_pp, alpha2, n_alpha2) = self._build_tree(
                    tm, pm, gm, log_u, direction, depth - 1, epsilon, H0, observations
                )
            else:
                (_, _, _, tp, pp, gp,
                 theta_pp, grad_pp, lp_pp,
                 n_pp, s_pp, alpha2, n_alpha2) = self._build_tree(
                    tp, pp, gp, log_u, direction, depth - 1, epsilon, H0, observations
                )

            # Accept theta_pp with probability proportional to its subtree size
            if n_p + n_pp > 0 and np.random.uniform() < n_pp / (n_p + n_pp):
                theta_p, grad_p, lp_p = theta_pp, grad_pp, lp_pp

            alpha   += alpha2
            n_alpha += n_alpha2
            s_p      = s_pp and self._no_u_turn(tm, tp, pm, pp)
            n_p     += n_pp

        return tm, pm, gm, tp, pp, gp, theta_p, grad_p, lp_p, n_p, s_p, alpha, n_alpha

    # ------------------------------------------------------------------
    # Single NUTS transition
    # ------------------------------------------------------------------

    def _nuts_transition(
        self,
        theta:        tf.Tensor,
        grad:         tf.Tensor,
        log_prob:     tf.Tensor,
        epsilon:      float,
        observations: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, float, int]:
        """
        One full NUTS transition via slice sampling over the trajectory.

        Returns
        -------
        new_theta, new_grad, new_log_prob : next chain state
        mean_alpha : mean acceptance probability (used by dual averaging)
        tree_depth : depth of the tree actually built
        """
        p  = tf.random.normal(shape=tf.shape(theta), dtype=DTYPE)
        H0 = self._hamiltonian(log_prob, p)

        # Slice variable: u ~ Uniform(0, exp(H0))  →  log u ~ Uniform(-inf, H0)
        log_u = H0 + np.log(np.random.uniform() + 1e-300)

        # Trajectory endpoints (initialised at current state)
        tm, pm, gm = theta, p, grad   # minus (left) endpoint
        tp, pp, gp = theta, p, grad   # plus  (right) endpoint

        n, s          = 1, 1
        alpha_sum     = 0.0
        n_alpha       = 0
        curr_theta    = theta
        curr_grad     = grad
        curr_lp       = log_prob
        depth         = 0

        while s and depth < self.max_tree_depth:
            direction = 1.0 if np.random.uniform() > 0.5 else -1.0

            if direction < 0:
                (tm, pm, gm, _, _, _,
                 theta_p, grad_p, lp_p,
                 n_p, s_p, alpha, n_a) = self._build_tree(
                    tm, pm, gm, log_u, direction, depth, epsilon, H0, observations
                )
            else:
                (_, _, _, tp, pp, gp,
                 theta_p, grad_p, lp_p,
                 n_p, s_p, alpha, n_a) = self._build_tree(
                    tp, pp, gp, log_u, direction, depth, epsilon, H0, observations
                )

            # Metropolised subtree acceptance
            if s_p and n_p > 0 and np.random.uniform() < min(1.0, n_p / n):
                curr_theta, curr_grad, curr_lp = theta_p, grad_p, lp_p

            alpha_sum += alpha
            n_alpha   += n_a
            n         += n_p
            s          = s_p and self._no_u_turn(tm, tp, pm, pp)
            depth     += 1

        mean_alpha = alpha_sum / max(n_alpha, 1)
        return curr_theta, curr_grad, curr_lp, mean_alpha, depth

    # ------------------------------------------------------------------
    # Public API (compatible with parent run_chain signature)
    # ------------------------------------------------------------------

    def run_chain(
        self,
        observations:      tf.Tensor,
        init_theta:        tf.Tensor,
        num_iterations:    int,
        burn_in:           int            = 100,
        step_size:         Optional[float] = None,
        num_leapfrog_steps: int           = 10,    # ignored — NUTS sets this automatically
    ) -> Dict[str, Any]:
        """
        Run NUTS with dual-averaging step-size adaptation.

        During the warmup (first `burn_in` iterations) epsilon is adapted to
        drive the mean acceptance probability toward `target_accept_rate`.
        After warmup, the smoothed (time-averaged) step size is frozen.

        `num_leapfrog_steps` is accepted for API compatibility but ignored;
        NUTS selects trajectory length automatically.

        Returns the same keys as `HMC.run_chain`, plus:
          'step_sizes'      – epsilon used at each iteration
          'tree_depths'     – NUTS tree depth at each iteration
          'final_step_size' – the smoothed epsilon fixed after warmup
        """
        print(
            f"Starting NUTS-Adaptive HMC | {num_iterations} iters "
            f"| warmup={burn_in} | target α={self.target_accept_rate}"
        )
        start_time = time.time()

        current_theta = init_theta
        current_lp, current_grad = self._compute_log_prob_and_grad(
            current_theta, observations
        )

        # Initialise step size
        if step_size is not None:
            epsilon = float(step_size)
        else:
            print("Searching for a reasonable initial step size...")
            epsilon = self._find_reasonable_step_size(current_theta, observations)
        print(f"Initial step size ε₀ = {epsilon:.5f}")

        # Dual-averaging state (§3.2, Hoffman & Gelman)
        mu              = np.log(10.0 * epsilon)
        H_bar           = 0.0
        log_epsilon_bar = np.log(epsilon)   # smoothed log step size

        samples_ta   = tf.TensorArray(dtype=DTYPE, size=num_iterations, clear_after_read=False)
        log_probs_ta = tf.TensorArray(dtype=DTYPE, size=num_iterations, clear_after_read=False)
        step_sizes:  list = []
        tree_depths: list = []

        for i in range(num_iterations):
            current_theta, current_grad, current_lp, mean_alpha, t_depth = \
                self._nuts_transition(
                    current_theta, current_grad, current_lp, epsilon, observations
                )

            m = i + 1  # 1-indexed iteration counter for dual averaging

            if i < burn_in:
                # ---- Dual-averaging update ----
                H_bar = (
                    (1.0 - 1.0 / (m + self._DA_T0)) * H_bar
                    + (1.0 / (m + self._DA_T0)) * (self.target_accept_rate - mean_alpha)
                )
                log_epsilon     = mu - (np.sqrt(m) / self._DA_GAMMA) * H_bar
                log_epsilon_bar = (
                    m ** (-self._DA_KAPPA) * log_epsilon
                    + (1.0 - m ** (-self._DA_KAPPA)) * log_epsilon_bar
                )
                epsilon = np.exp(log_epsilon)
            else:
                # ---- Sampling phase: freeze to the smoothed estimate ----
                epsilon = np.exp(log_epsilon_bar)

            samples_ta   = samples_ta.write(i, current_theta)
            log_probs_ta = log_probs_ta.write(i, current_lp)
            step_sizes.append(epsilon)
            tree_depths.append(t_depth)

            if (i + 1) % 10 == 0:
                phase = "Warmup  " if i < burn_in else "Sampling"
                print(
                    f"[{phase}] {i+1:>{len(str(num_iterations))}}/{num_iterations}"
                    f" | α̂={mean_alpha:.3f}"
                    f" | ε={epsilon:.5f}"
                    f" | depth={t_depth}"
                    f" | LL={float(current_lp):.2f}"
                )

        total_time    = time.time() - start_time
        all_samples   = samples_ta.stack()
        all_log_probs = log_probs_ta.stack()

        return {
            'samples':         all_samples[burn_in:],
            'log_probs':       all_log_probs[burn_in:],
            'accepted':        None,          # NUTS has no single accept/reject flag
            'acceptance_rate': None,          # use step_sizes trajectory instead
            'step_sizes':      step_sizes,
            'tree_depths':     tree_depths,
            'final_step_size': float(np.exp(log_epsilon_bar)),
            'time':            total_time,
        }
