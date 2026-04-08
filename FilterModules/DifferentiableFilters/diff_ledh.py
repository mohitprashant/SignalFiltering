import tensorflow as tf
from typing import Tuple

from FilterModules.ParticleFilters.ledh_particle import PFPF_LEDHFilter

DTYPE = tf.float32


class DiffSinkhornLEDHFilter(PFPF_LEDHFilter):
    """
    Differentiable PFPF-LEDH Filter with Sinkhorn Optimal Transport Resampling.

    Inherits the full LEDH particle flow mechanics from PFPF_LEDHFilter
    (local per-particle linearization, auxiliary EKF tracking, Liouville
    log-det Jacobian accumulation) and replaces the hard categorical
    resampling step with a Sinkhorn OT barycentric projection.

    Why this enables differentiability
    ------------------------------------
    The standard PFPF_LEDHFilter uses tf.cond-gated multinomial resampling:
    the gradient is undefined through index-selection, and the ESS threshold
    introduces a discrete branching point.  Sinkhorn OT instead computes a
    soft transport plan P ∈ R^{N×N} via entropy-regularized dual iterations
    and moves each target particle to the barycentric average of source
    particles weighted by P[:,j].  Both the plan and the resulting positions
    are smooth, differentiable functions of the input log-weights and
    particle locations.

    Additionally, resampling is applied unconditionally at every timestep
    (the ESS gate is removed), so the gradient path through update() is
    always identical and never short-circuits to the no-op branch.

    Architecture
    ------------
    DiffSinkhornLEDHFilter
        └── PFPF_LEDHFilter
                └── PFPF_EDHFilter
                        ├── ParticleFilter   (weight mgmt, base resample)
                        └── ExactDaumHuangFilter  (flow eqs, aux EKF)

    Sinkhorn OT methods are defined directly on this class (not via a
    mixin) to avoid MRO ambiguity with the existing cross-branch hierarchy.

    Parameters
    ----------
    num_particles : int
        Number of particles N.
    num_steps : int
        Number of λ-flow integration steps from 0 to 1.
    resample_threshold_ratio : float
        Kept for API compatibility; ignored — OT resampling is always applied.
    ot_epsilon : float
        Entropy regularisation strength ε for Sinkhorn.  Higher ε → smoother
        (more diffuse) transport plan; lower ε → sharper (approaches hard OT).
    ot_n_iter : int
        Number of stabilised log-domain Sinkhorn iterations.
    """

    def __init__(
        self,
        num_particles:           int   = 100,
        num_steps:               int   = 30,
        resample_threshold_ratio: float = 0.5,
        ot_epsilon:              float = 0.5,
        ot_n_iter:               int   = 20,
        label:                   str   = None,
    ):
        label = label or (
            f"Diff-Sinkhorn LEDH "
            f"(N={num_particles}, Steps={num_steps}, ε={ot_epsilon:.2f})"
        )
        super().__init__(
            num_particles=num_particles,
            num_steps=num_steps,
            resample_threshold_ratio=resample_threshold_ratio,
            label=label,
        )
        self.ot_epsilon = tf.constant(ot_epsilon, dtype=DTYPE)
        self.ot_n_iter  = ot_n_iter

    # ------------------------------------------------------------------
    # Sinkhorn OT engine
    # ------------------------------------------------------------------

    def sinkhorn_potentials(
        self,
        log_a: tf.Tensor,   # (N,) log source weights
        log_b: tf.Tensor,   # (N,) log target weights (uniform: -log N)
        C:     tf.Tensor,   # (N, N) squared-Euclidean cost matrix
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Stabilised log-domain Sinkhorn iterations.

        Solves the dual of the entropy-regularised OT problem
            min_P  <C, P> − ε H(P)
            s.t.   P 1 = a,  Pᵀ 1 = b
        via alternating projections in log-space with averaging damping
        (Sinkhorn–Knopp with log-sum-exp stabilisation).

        Returns
        -------
        f, g : (N,) dual potentials satisfying
               P_ij = exp((f_i + g_j − C_ij) / ε) · a_i · b_j
        """
        N   = self.num_particles
        f   = tf.zeros((N,), dtype=DTYPE)
        g   = tf.zeros((N,), dtype=DTYPE)
        eps = self.ot_epsilon

        for _ in range(self.ot_n_iter):
            # f-update: f_i ← -ε log Σ_j exp((g_j - C_ij)/ε + log b_j)
            tmp_f = log_b[tf.newaxis, :] + (g[tf.newaxis, :] - C) / eps
            f     = 0.5 * (f + (-eps * tf.reduce_logsumexp(tmp_f, axis=1)))

            # g-update: g_j ← -ε log Σ_i exp((f_i - C_ij)/ε + log a_i)
            tmp_g = log_a[:, tf.newaxis] + (f[:, tf.newaxis] - C) / eps
            g     = 0.5 * (g + (-eps * tf.reduce_logsumexp(tmp_g, axis=0)))

        return f, g

    def resample(
        self,
        particles:   tf.Tensor,   # (N, nx)
        log_weights: tf.Tensor,   # (N,)
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Differentiable resampling via Sinkhorn OT barycentric projection.

        Maps the current weighted particle distribution (a, x_{1:N}) to a
        new set of N equally-weighted particles by solving the OT problem
        from a = normalised weights to b = uniform(1/N), then applying
        the barycentric projection:

            x̃_j = N · Σ_i P_{ij} · x_i

        The resulting positions x̃ are differentiable w.r.t. both the
        input log_weights and the input particle positions.

        Overrides: ParticleFilter.resample  (hard categorical sampling)
        """
        N           = self.num_particles
        log_weights = tf.maximum(log_weights, -1e9)                         # clip extreme underflows
        log_w_norm  = log_weights - tf.reduce_logsumexp(log_weights)        # (N,) normalised log-weights
        log_b       = tf.fill([N], -tf.math.log(tf.cast(N, DTYPE)))         # (N,) uniform target

        diff = particles[:, tf.newaxis, :] - particles[tf.newaxis, :, :]   # (N, N, nx)
        C    = tf.reduce_sum(diff ** 2, axis=-1)                            # (N, N) cost matrix

        f, g = self.sinkhorn_potentials(log_w_norm, log_b, C)

        # log P_{ij} = (f_i + g_j - C_{ij}) / ε + log a_i + log b_j
        log_P = (
            f[:, tf.newaxis] + g[tf.newaxis, :] - C
        ) / self.ot_epsilon + log_w_norm[:, tf.newaxis] + log_b[tf.newaxis, :]

        P            = tf.exp(log_P)                                        # (N, N) transport plan
        new_particles = tf.cast(N, DTYPE) * tf.linalg.matmul(              # barycentric projection
            P, particles, transpose_a=True
        )
        return new_particles, log_b                                         # return uniform log-weights

    # ------------------------------------------------------------------
    # Update: unconditional Sinkhorn resampling every step
    # ------------------------------------------------------------------

    @tf.function(jit_compile=True)
    def update(
        self,
        state_pred:  Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        observation: tf.Tensor,
    ) -> Tuple[Tuple, tf.Tensor, tf.Tensor]:
        """
        Differentiable measurement update.

        Mirrors PFPF_EDHFilter.update exactly, except:
          1. Sinkhorn OT resampling (self.resample) is applied
             unconditionally — the ESS-gated tf.cond is removed.
          2. No discrete branching point exists in the computational graph,
             so gradients flow cleanly through the resampling step.

        Step summary
        ------------
        1. LEDH particle migration: move particles along the local
           exact Daum-Huang flow towards the observation.
        2. Weight update: add log-likelihood and log-det Jacobian.
        3. Sinkhorn OT resampling (always).
        4. Auxiliary EKF update for the next prediction step.

        Returns
        -------
        updated_state : (particles, log_weights, m_upd, P_upd)
        x_est         : weighted-mean point estimate, shape (nx,)
        metrics       : [ess, avg_flow_cond, ekf_S_cond, ekf_P_cond,
                         step_log_likelihood]
                        (same layout as PFPF_EDHFilter for run_filter compatibility)
        """
        particles, log_weights, m_pred, P_pred = state_pred
        y_curr = tf.reshape(observation, [self.ny])

        # ── 1. LEDH particle migration ────────────────────────────────
        # Dispatches to PFPF_LEDHFilter.particle_migration (local=True),
        # which calls PFPF_EDHFilter.particle_migration with local=True.
        particles, total_cond, log_det_J = self.particle_migration(
            particles, m_pred, P_pred, y_curr, log_weights
        )

        # ── 2. Weight update ──────────────────────────────────────────
        pred_obs_mean = self.h_func(particles)
        obs_t         = tf.reshape(observation, [1, self.ny])
        diff          = obs_t - pred_obs_mean
        log_lik       = self.obs_noise_dist.log_prob(diff)
        log_weights  += log_lik + log_det_J

        max_log_w          = tf.reduce_max(log_weights)
        step_log_likelihood = max_log_w + tf.math.log(
            tf.reduce_mean(tf.exp(log_weights - max_log_w))
        )

        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm     = tf.exp(log_w_norm)
        ess        = 1.0 / (tf.reduce_sum(w_norm ** 2) + 1e-12)

        # ── 3. Unconditional Sinkhorn OT resampling ───────────────────
        particles, log_weights = self.resample(particles, log_weights)

        # ── 4. Point estimate ─────────────────────────────────────────
        log_w_norm_final = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm_final     = tf.exp(log_w_norm_final)
        x_est            = tf.reduce_sum(w_norm_final[:, tf.newaxis] * particles, axis=0)

        # ── 5. Auxiliary EKF update ───────────────────────────────────
        ekf_updated_state, _, ekf_metrics = self.ekf.update((m_pred, P_pred), observation)
        m_upd, P_upd = ekf_updated_state

        avg_cond = total_cond / tf.cast(self.steps, DTYPE)
        metrics  = tf.concat(
            [[ess], [avg_cond], ekf_metrics, [step_log_likelihood]], axis=0
        )
        return (particles, log_weights, m_upd, P_upd), x_est, metrics
