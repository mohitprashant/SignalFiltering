"""
ParamEstimationPipeline/mass_matrix_hmc.py

HMC with configurable mass matrix preconditioning and windowed warmup.

MassMatrixHMC
-------------
Extends DeepONetHMC with four mass matrix schemes:

  identity    M = I
              No preconditioning; momentum p ~ N(0, I).

  diagonal    M = diag(m₁, …, m_d),  m_i = 1/σ̂_i²
              Estimated from warmup samples via WelfordVariance.
              Scales independently per dimension.

  dense       M = L Lᵀ  (Cholesky of precision matrix Σ̂⁻¹)
              Estimated via WelfordCovariance.
              Captures off-diagonal correlations.

  riemannian  M(θ) = diag(|∇U(θ)|² + λ)
              Local diagonal Fisher approximation, computed per proposal
              from the gradient already available at the current position.
              No implicit integrator required (quasi-Riemannian).

Leapfrog equations (all schemes share this structure):
  p  ~ N(0, M)
  r  ← p + ε/2 · ∇log p(θ)
  for step in L:
      θ ← θ + ε · M⁻¹ r
      r ← r + ε · ∇log p(θ)   (except on last step)
  r  ← r + ε/2 · ∇log p(θ)
  T(p) = pᵀ M⁻¹ p / 2

Windowed warmup (run_warmup_and_chain)
--------------------------------------
  W0 (25%): identity M, dual-averaging step size only
  W1  (8%): first Welford snapshot → update M
  W2 (22%): expanding window, update Welford + dual-avg
  W3 (45%): final expansion, same
  Sampling: M and ε frozen; main chain runs
"""

import time
import numpy as np
import tensorflow as tf
from typing import Callable, Dict, Optional, Tuple

from ParamEstimationPipeline.deeponet_hmc   import DeepONetHMC
from ParamEstimationPipeline.online_stats   import WelfordVariance, WelfordCovariance
from ParamEstimationPipeline.dual_averaging import DualAveraging

DTYPE = tf.float32


class MassMatrixHMC(DeepONetHMC):
    """
    HMC with configurable mass matrix built on DeepONetHMC.

    The active scheme is controlled by the mass_* attributes, which can be
    set at construction time or mutated via the set_mass_* helpers during
    windowed warmup.

    Parameters
    ----------
    ssm_builder        : Callable theta → SSM
    filter_module      : DeepONetSinkhornLEDHFilter instance
    prior_log_prob_fn  : Callable theta → log-prior scalar
    mass_diag          : (d,) float32 tensor — diagonal mass values, or None
    mass_L             : (d,d) float32 tensor — lower Cholesky of dense M, or None
    riemannian         : bool  — if True, local Fisher diagonal used per proposal
    riemannian_lambda  : float — regularisation for local Fisher mass
    """

    def __init__(
        self,
        ssm_builder:       Callable,
        filter_module,
        prior_log_prob_fn: Callable,
        mass_diag:         Optional[tf.Tensor] = None,
        mass_L:            Optional[tf.Tensor] = None,
        riemannian:        bool  = False,
        riemannian_lambda: float = 0.01,
    ) -> None:
        super().__init__(ssm_builder, filter_module, prior_log_prob_fn)
        self.mass_diag         = mass_diag
        self.mass_L            = mass_L
        self.riemannian        = riemannian
        self.riemannian_lambda = riemannian_lambda

    # ------------------------------------------------------------------
    # Mass setters
    # ------------------------------------------------------------------

    def set_mass_identity(self) -> None:
        self.mass_diag  = None
        self.mass_L     = None
        self.riemannian = False

    def set_mass_diagonal(self, mass_diag: tf.Tensor) -> None:
        self.mass_diag  = mass_diag
        self.mass_L     = None
        self.riemannian = False

    def set_mass_dense(self, mass_L: tf.Tensor) -> None:
        self.mass_diag  = None
        self.mass_L     = mass_L
        self.riemannian = False

    def set_mass_riemannian(self, lam: float) -> None:
        self.mass_diag         = None
        self.mass_L            = None
        self.riemannian        = True
        self.riemannian_lambda = lam

    # ------------------------------------------------------------------
    # Kinetic-energy primitives
    # ------------------------------------------------------------------

    def _sample_momentum(
        self, d: int, local_m: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Sample p ~ N(0, M)."""
        z = tf.random.normal([d], dtype=DTYPE)
        if self.mass_L is not None:        # Dense:      p = L z
            return tf.linalg.matvec(self.mass_L, z)
        if self.mass_diag is not None:     # Diagonal:   p_i = √m_i · z_i
            return z * tf.sqrt(self.mass_diag)
        if local_m is not None:            # Riemannian: p_i = √m_i(θ) · z_i
            return z * tf.sqrt(local_m)
        return z                           # Identity

    def _kinetic_energy(
        self, p: tf.Tensor, local_m: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """T(p) = pᵀ M⁻¹ p / 2."""
        if self.mass_L is not None:
            y = tf.linalg.triangular_solve(
                self.mass_L, tf.reshape(p, [-1, 1]), lower=True)
            return 0.5 * tf.reduce_sum(tf.square(y))
        if self.mass_diag is not None:
            return 0.5 * tf.reduce_sum(tf.square(p) / self.mass_diag)
        if local_m is not None:
            return 0.5 * tf.reduce_sum(tf.square(p) / local_m)
        return 0.5 * tf.reduce_sum(tf.square(p))

    def _mass_inv_p(
        self, p: tf.Tensor, local_m: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Compute M⁻¹ p for the position update θ += ε M⁻¹ p."""
        if self.mass_L is not None:        # Dense: L⁻ᵀ (L⁻¹ p)
            y = tf.linalg.triangular_solve(
                self.mass_L, tf.reshape(p, [-1, 1]), lower=True)
            x = tf.linalg.triangular_solve(
                tf.transpose(self.mass_L), y, lower=False)
            return tf.reshape(x, [-1])
        if self.mass_diag is not None:     # Diagonal: p / m
            return p / self.mass_diag
        if local_m is not None:            # Riemannian
            return p / local_m
        return p                           # Identity

    # ------------------------------------------------------------------
    # Single HMC proposal
    # ------------------------------------------------------------------

    def _hmc_step(
        self,
        current_theta: tf.Tensor,
        current_lp:    tf.Tensor,
        current_grad:  tf.Tensor,
        observations:  tf.Tensor,
        step_size:     float,
        num_leapfrog:  int,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, bool, float]:
        """
        One HMC proposal with the active mass matrix.

        For the Riemannian scheme, M(θ) is built from the squared gradient
        at the current position and held fixed throughout the trajectory
        (quasi-Riemannian explicit leapfrog — avoids implicit fixed-point
        iterations while still reflecting local geometry).

        Returns
        -------
        (new_theta, new_lp, new_grad, accepted: bool, alpha: float)
        """
        d = tf.shape(current_theta)[0]

        # ── CCPM setup ────────────────────────────────────────────────────
        # Draw one seed and one particle cloud for the entire HMC step.
        # Every filter call below (current re-evaluation + all leapfrog
        # gradient calls) uses the same seed and the same initial particles,
        # so the stochastic Hamiltonian H(θ, p; ξ) is consistent throughout.
        # This is the Completely Correlated Pseudo-Marginal (CCPM) approach:
        # the noise enters as a fixed auxiliary variable rather than as
        # independent noise at each evaluation.
        proposal_seed       = int(np.random.randint(0, 2**31))
        fixed_initial_state = self.filter_module.initialize_state()

        # Re-evaluate the current position with this step's noise realization.
        # Without this, H_curr uses stale noise from a previous step while
        # H_prop uses proposal_seed noise — making the acceptance ratio
        # unreliable even when the leapfrog trajectory itself is coherent.
        tf.random.set_seed(proposal_seed)
        current_lp_c, current_grad_c = self._compute_log_prob_and_grad(
            current_theta, observations, fixed_initial_state)

        local_m = (tf.square(current_grad_c) + self.riemannian_lambda
                   if self.riemannian else None)

        p = self._sample_momentum(d, local_m)

        q = current_theta
        r = p + 0.5 * step_size * current_grad_c  # half-step with ξ-consistent grad

        for s in range(num_leapfrog):
            q = q + step_size * self._mass_inv_p(r, local_m)
            tf.random.set_seed(proposal_seed)      # same ξ for every leapfrog step
            q_lp, q_grad = self._compute_log_prob_and_grad(
                q, observations, fixed_initial_state)
            if s < num_leapfrog - 1:
                r = r + step_size * q_grad

        r = r + 0.5 * step_size * q_grad          # final half-step

        if tf.math.is_finite(q_lp):
            H_curr    = -current_lp_c + self._kinetic_energy(p, local_m)
            H_prop    = -q_lp         + self._kinetic_energy(r, local_m)
            log_alpha = float(tf.minimum(0.0, H_curr - H_prop))
            alpha     = float(np.exp(log_alpha))
            # Use numpy RNG so the acceptance draw is independent of the TF
            # RNG stream used for filter noise and momentum sampling.
            accepted  = np.log(np.random.uniform()) < log_alpha
        else:
            alpha    = 0.0
            accepted = False

        if accepted:
            return q, q_lp, q_grad, True, alpha
        # On rejection, return the ξ-consistent current values so the next
        # step doesn't inherit a stale likelihood estimate.
        return current_theta, current_lp_c, current_grad_c, False, alpha

    # ------------------------------------------------------------------
    # Full fixed chain (override)
    # ------------------------------------------------------------------

    def run_chain(
        self,
        observations:       tf.Tensor,
        init_theta:         tf.Tensor,
        num_iterations:     int,
        burn_in:            int   = 0,
        step_size:          float = 0.01,
        num_leapfrog_steps: int   = 5,
        step_size_jitter:   float = 0.2,
    ) -> Dict:
        """
        step_size_jitter : fraction by which ε is randomly perturbed each
            iteration, drawn from Uniform(1−j, 1+j) × step_size.  Breaks
            trajectory resonance without violating detailed balance (the
            jittered ε is used consistently within the MH proposal).
            Set to 0.0 to use a fixed step size.
        """
        start         = time.time()
        current_theta = init_theta
        current_lp, current_grad = self._compute_log_prob_and_grad(
            current_theta, observations)

        samples_ta   = tf.TensorArray(DTYPE,   size=num_iterations, clear_after_read=False)
        log_probs_ta = tf.TensorArray(DTYPE,   size=num_iterations, clear_after_read=False)
        accepted_ta  = tf.TensorArray(tf.bool, size=num_iterations, clear_after_read=False)
        accept_count = 0

        for i in range(num_iterations):
            if step_size_jitter > 0.0:
                eps_i = step_size * np.random.uniform(
                    1.0 - step_size_jitter, 1.0 + step_size_jitter)
            else:
                eps_i = step_size
            current_theta, current_lp, current_grad, ok, _ = self._hmc_step(
                current_theta, current_lp, current_grad,
                observations, eps_i, num_leapfrog_steps)
            if ok:
                accept_count += 1
            samples_ta   = samples_ta.write(i, current_theta)
            log_probs_ta = log_probs_ta.write(i, current_lp)
            accepted_ta  = accepted_ta.write(i, tf.constant(ok))

            if (i + 1) % 25 == 0:
                print(f"    iter {i+1:>4}/{num_iterations}  "
                      f"acc={accept_count/(i+1):.2f}  "
                      f"lp={current_lp.numpy():.1f}  "
                      f"ε={eps_i:.4f}")

        return {
            'samples':         samples_ta.stack()[burn_in:],
            'log_probs':       log_probs_ta.stack()[burn_in:],
            'accepted':        accepted_ta.stack(),
            'acceptance_rate': accept_count / num_iterations,
            'time':            time.time() - start,
        }

    # ------------------------------------------------------------------
    # Windowed warmup + frozen sampling chain
    # ------------------------------------------------------------------

    def run_warmup_and_chain(
        self,
        observations:      tf.Tensor,
        init_theta:        tf.Tensor,
        mass_scheme:       str,
        num_warmup:        int   = 300,
        num_samples:       int   = 200,
        init_step_size:    float = 0.01,
        num_leapfrog:      int   = 5,
        target_acc:        float = 0.65,
        riemannian_lambda: float = 0.01,
        min_step_size:     float = 1e-4,
        step_size_jitter:  float = 0.2,
    ) -> Dict:
        """
        Windowed warmup (identity → mass update → expanding windows) followed
        by a frozen-parameter sampling chain.

        Window schedule (fractions of num_warmup):
          W0 (25%): identity M, dual-avg step size only
          W1  (8%): first Welford snapshot, update M
          W2 (22%): update Welford + dual-avg
          W3 (45%): same (final expansion)
        After warmup: freeze ε̄ and M, run num_samples chain iterations.

        Parameters
        ----------
        mass_scheme : one of 'identity', 'diagonal', 'dense', 'riemannian'

        Returns
        -------
        dict with keys: samples, log_probs, accepted, acceptance_rate, time,
                        step_size_trace, acc_rate_trace, final_step_size,
                        welford_var, welford_cov
        """
        d         = init_theta.shape[0]
        dual_avg  = DualAveraging(init_step_size, target_acc,
                                  min_step_size=min_step_size)
        welford_v = WelfordVariance(d)
        welford_c = WelfordCovariance(d)

        w0 = max(10, int(num_warmup * 0.25))
        w1 = max(5,  int(num_warmup * 0.08))
        w2 = max(5,  int(num_warmup * 0.22))
        w3 = max(5,  num_warmup - w0 - w1 - w2)
        windows = [w0, w1, w2, w3]

        current_theta = init_theta
        current_lp, current_grad = self._compute_log_prob_and_grad(
            current_theta, observations)

        step_size       = init_step_size
        step_size_trace = []
        acc_rate_trace  = []

        self.set_mass_identity()

        for w_idx, w_size in enumerate(windows):
            # Update mass at start of each window after W0, then reset both
            # Welford accumulators and dual averaging so each window adapts
            # only from its own samples (mirrors Stan's windowed warmup).
            if w_idx > 0:
                if mass_scheme == 'diagonal' and welford_v.count >= 2:
                    self.set_mass_diagonal(
                        tf.constant(welford_v.mass_diag, dtype=DTYPE))
                elif mass_scheme == 'dense' and welford_c.count >= 2:
                    try:
                        self.set_mass_dense(
                            tf.constant(welford_c.mass_chol_L(), dtype=DTYPE))
                    except np.linalg.LinAlgError:
                        self.set_mass_diagonal(
                            tf.constant(welford_v.mass_diag, dtype=DTYPE))
                elif mass_scheme == 'riemannian':
                    self.set_mass_riemannian(riemannian_lambda)

                # Reset Welford so transient samples don't pollute next window.
                welford_v = WelfordVariance(d)
                welford_c = WelfordCovariance(d)
                # Reset dual averaging with the current smoothed step size as
                # the new init so it re-adapts ε to the updated mass matrix.
                dual_avg = DualAveraging(step_size, target_acc,
                                          min_step_size=min_step_size)

            scheme_tag = (
                f"diag={self.mass_diag.numpy().round(3)}"
                if self.mass_diag is not None else
                f"dense(L_diag={self.mass_L.numpy().diagonal().round(3)})"
                if self.mass_L is not None else
                "riemannian(local)" if self.riemannian else "identity"
            )
            print(f"  [W{w_idx}] {w_size:>4} iters  ε={step_size:.4f}  M={scheme_tag}")

            acc_w = 0
            for step_i in range(w_size):
                current_theta, current_lp, current_grad, ok, alpha = self._hmc_step(
                    current_theta, current_lp, current_grad,
                    observations, step_size, num_leapfrog)
                if ok:
                    acc_w += 1
                welford_v.update(current_theta.numpy())
                welford_c.update(current_theta.numpy())
                step_size = dual_avg.update(alpha)
                step_size_trace.append(step_size)
                acc_rate_trace.append(float(ok))

                if (step_i + 1) % 10 == 0 or step_i == w_size - 1:
                    print(f"    step {step_i+1:>3}/{w_size}  "
                          f"ε={step_size:.4f}  "
                          f"lp={current_lp.numpy():.1f}  "
                          f"acc={acc_w/(step_i+1):.2f}")

            print(f"         → acc={acc_w / w_size:.2f}")

        # Freeze mass and step size
        final_eps = dual_avg.final_step_size
        if mass_scheme == 'diagonal' and welford_v.count >= 2:
            self.set_mass_diagonal(
                tf.constant(welford_v.mass_diag, dtype=DTYPE))
        elif mass_scheme == 'dense' and welford_c.count >= 2:
            try:
                self.set_mass_dense(
                    tf.constant(welford_c.mass_chol_L(), dtype=DTYPE))
            except np.linalg.LinAlgError:
                print("  Warning: final Cholesky failed — using diagonal M")
                self.set_mass_diagonal(
                    tf.constant(welford_v.mass_diag, dtype=DTYPE))
        elif mass_scheme == 'riemannian':
            self.set_mass_riemannian(riemannian_lambda)

        jitter_pct = f"  jitter=±{step_size_jitter*100:.0f}%" if step_size_jitter > 0 else ""
        print(f"\n  [Chain] {num_samples} samples  frozen ε={final_eps:.4f}{jitter_pct}")
        results = self.run_chain(
            observations       = observations,
            init_theta         = current_theta,
            num_iterations     = num_samples,
            burn_in            = 0,
            step_size          = final_eps,
            num_leapfrog_steps = num_leapfrog,
            step_size_jitter   = step_size_jitter,
        )

        results['step_size_trace'] = np.array(step_size_trace)
        results['acc_rate_trace']  = np.array(acc_rate_trace)
        results['final_step_size'] = final_eps
        results['welford_var']     = welford_v.variance
        results['welford_cov']     = welford_c.covariance
        return results
