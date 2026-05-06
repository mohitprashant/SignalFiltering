import tensorflow as tf
from typing import Tuple

from FilterModules.DifferentiableFilters.diff_ledh import DiffSinkhornLEDHFilter
from FilterModules.NeuralFilter.Components.neural_sinkhorn_ot import NeuralSinkhornPotentialNet

DTYPE = tf.float32


class DeepONetSinkhornLEDHFilter(DiffSinkhornLEDHFilter):
    """
    Neural Operator Sinkhorn-LEDH Particle Filter.

    Combines LEDH particle flow with a DeepONet operator that *learns*
    the Sinkhorn OT transport conditioned on the current observation and
    SSM parameters, replacing the fixed iterative dual algorithm with a
    single neural-network forward pass.

    Design rationale
    ----------------
    DiffSinkhornLEDHFilter runs K iterations of the fixed Sinkhorn-Knopp
    algorithm at every time step.  The dual potentials (f*, g*) that the
    algorithm converges to depend on the particle cloud, the observation,
    and implicitly on the SSM parameters (through the particle weights and
    positions produced by the LEDH flow).  By training a DeepONet to
    predict f* directly we:
      1. Replace O(N^2 · K) Sinkhorn work with a single O(N · B) network pass.
      2. Make the transport plan an explicit function of θ_SSM, so that when
         HMC proposes a new θ the plan adapts immediately via set_theta().
      3. Maintain full differentiability through the resampling step (no
         hard sampling, all operations remain smooth matrix computations).

    Neural operator structure
    -------------------------
    NeuralSinkhornPotentialNet (DeepONet) maps
        (x_particles, y_obs, θ_SSM) → f ∈ R^N
    where
        f_i = Σ_k  c_k(y, θ)  ·  phi_k(x_i)
        ─ Trunk phi_k : R^{nx} → R  (ScalarTrunkModule)
        ─ Branch: MLP over [y ∥ θ] → coefficients c ∈ R^K

    One Sinkhorn g-update then enforces the row-marginal constraint:
        g_j = −ε · log Σ_i exp((f_i − C_{ij}) / ε + log a_i)
    after which the standard barycentric-projection resampling is applied.

    HMC integration
    ---------------
    The filter stores theta_ssm as a tf.Variable.  Call set_theta(theta)
    after each HMC accepted sample; the next filter.update() will
    automatically use the updated θ when computing the transport plan.

    Inheritance chain
    -----------------
    DeepONetSinkhornLEDHFilter
        └── DiffSinkhornLEDHFilter
                └── PFPF_LEDHFilter
                        └── PFPF_EDHFilter
                                ├── ParticleFilter
                                └── ExactDaumHuangFilter

    Parameters
    ----------
    num_particles            : Number of particles N.
    num_steps                : LEDH flow integration steps.
    resample_threshold_ratio : Kept for API compat; ignored (always resample).
    ot_epsilon               : Sinkhorn entropy regularisation ε.
    ot_n_iter                : Fixed Sinkhorn iterations used in pretraining.
    num_basis                : Number of DeepONet trunk basis functions K.
    embed_dim                : Embedding dim for trunk and branch modules.
    theta_dim                : Dimension of SSM parameter vector θ.
    lr                       : Adam learning rate for pretraining.
    label                    : Optional display label.
    """

    def __init__(
        self,
        num_particles:            int   = 100,
        num_steps:                int   = 30,
        resample_threshold_ratio: float = 0.5,
        ot_epsilon:               float = 0.5,
        ot_n_iter:                int   = 20,
        num_basis:                int   = 16,
        embed_dim:                int   = 32,
        theta_dim:                int   = 1,
        lr:                       float = 1e-3,
        label:                    str   = None,
    ):
        label = label or (
            f"DeepONet-Sinkhorn LEDH "
            f"(N={num_particles}, Steps={num_steps}, ε={ot_epsilon:.2f}, K={num_basis})"
        )
        super().__init__(
            num_particles=num_particles,
            num_steps=num_steps,
            resample_threshold_ratio=resample_threshold_ratio,
            ot_epsilon=ot_epsilon,
            ot_n_iter=ot_n_iter,
            label=label,
        )
        self.num_basis = num_basis
        self.embed_dim = embed_dim
        self.theta_dim = theta_dim
        self.lr        = lr

        # Built in load_ssm after nx / ny are resolved
        self.neural_ot_net = None
        self.optimizer     = None

        # tf.Variables for in-graph conditioning (allocated in load_ssm)
        self.theta_ssm    = None   # (theta_dim,)  — updated by set_theta()
        self._current_obs = None   # (ny,)         — updated inside update()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def load_ssm(self, ssm_model) -> None:
        """
        Load SSM components; build the network only on the first call.

        Subsequent calls (e.g. from an HMC inner loop that rebuilds the SSM
        at each iteration) update only the SSM-derived functions (f_func,
        h_func, Q, R, noise distributions, auxiliary EKF) without
        reinitialising the neural network or its trained weights.
        This preserves pretrained weights across HMC iterations.

        Call set_theta(theta) after each parameter change to update the
        conditioning variable used inside the neural transport map.

        Args:
            ssm_model: Any SSM instance compatible with the parent filter.
        """
        first_call = (self.neural_ot_net is None)
        super().load_ssm(ssm_model)

        if not first_call:
            return   # SSM functions updated; network preserved.

        # ── First call only: build network and allocate tf.Variables ──
        self.neural_ot_net = NeuralSinkhornPotentialNet(
            nx=self.nx,
            ny=self.ny,
            theta_dim=self.theta_dim,
            num_basis=self.num_basis,
            embed_dim=self.embed_dim,
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.lr, clipnorm=1.0
        )

        # tf.Variables allow mutation inside @tf.function(jit_compile=True)
        self.theta_ssm    = tf.Variable(
            tf.zeros([self.theta_dim], dtype=DTYPE),
            trainable=False,
            name="theta_ssm",
        )
        self._current_obs = tf.Variable(
            tf.zeros([self.ny], dtype=DTYPE),
            trainable=False,
            name="current_obs",
        )

        # Warm-up: build all sub-layers so variable shapes are fixed
        dummy_x  = tf.zeros((1, self.nx), dtype=DTYPE)
        dummy_y  = tf.zeros((self.ny,),   dtype=DTYPE)
        dummy_th = tf.zeros((self.theta_dim,), dtype=DTYPE)
        self.neural_ot_net(dummy_x, dummy_y, dummy_th)

        # Adapt the context normalisation layer with representative samples
        self._adapt_context_norm(n_samples=512)

    def _adapt_context_norm(self, n_samples: int = 512) -> None:
        """Adapt the branch's input-normalisation layer with random (y, θ) samples."""
        dummy_y  = tf.random.normal((n_samples, self.ny),        dtype=DTYPE)
        dummy_th = tf.random.normal((n_samples, self.theta_dim), dtype=DTYPE)
        context  = tf.concat([dummy_y, dummy_th], axis=1)
        self.neural_ot_net.context_norm.adapt(context)

    # ------------------------------------------------------------------
    # Runtime parameter update (called by HMC wrapper)
    # ------------------------------------------------------------------

    def set_theta(self, theta: tf.Tensor) -> None:
        """
        Update the SSM parameter conditioning tensor.

        Called after each HMC accepted proposal.  The next call to update()
        will read the new θ from the tf.Variable, so the neural transport plan
        automatically reflects the updated SSM parameters.

        Args:
            theta: (theta_dim,) float32 tensor of SSM parameters.
        """
        self.theta_ssm.assign(
            tf.cast(tf.reshape(theta, [self.theta_dim]), DTYPE)
        )

    # ------------------------------------------------------------------
    # Neural Sinkhorn resampling (replaces parent's fixed Sinkhorn)
    # ------------------------------------------------------------------

    def resample(
        self,
        particles:   tf.Tensor,   # (N, nx)
        log_weights: tf.Tensor,   # (N,)
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Differentiable resampling via learned Sinkhorn f-potentials.

        The DeepONet predicts f-potentials conditioned on (self._current_obs,
        self.theta_ssm).  One Sinkhorn g-update then enforces the source
        marginal constraint, and the resulting transport plan drives a
        differentiable barycentric projection.

        Step sequence
        -------------
        1. Neural f-prediction: f = net(particles, y_obs, θ)   O(N · K)
        2. One g-update (row-marginal correction):
               g_j = −ε · log Σ_i exp((f_i − C_{ij})/ε + log a_i)
        3. Transport plan:
               log P_{ij} = (f_i + g_j − C_{ij})/ε + log a_i + log b_j
        4. Barycentric projection:
               x̃_j = N · Σ_i P_{ij} · x_i

        Returns transported particles with uniform log-weights.

        Note: self._current_obs and self.theta_ssm are tf.Variables that are
        assigned inside update() before this method is called.
        """
        N     = self.num_particles
        eps   = self.ot_epsilon
        log_b = tf.fill([N], -tf.math.log(tf.cast(N, DTYPE)))   # uniform target

        log_weights = tf.maximum(log_weights, -1e9)
        log_w_norm  = log_weights - tf.reduce_logsumexp(log_weights)   # (N,)

        # Squared-Euclidean cost matrix  (N, N)
        diff = particles[:, tf.newaxis, :] - particles[tf.newaxis, :, :]   # (N, N, nx)
        C    = tf.reduce_sum(diff ** 2, axis=-1)                           # (N, N)

        # ── 1. Neural f-potentials ─────────────────────────────────────
        f = self.neural_ot_net(
            particles, self._current_obs, self.theta_ssm
        )   # (N,)

        # ── 2. One Sinkhorn g-update (enforces row-marginals = source weights)
        #   g_j = −ε · logsumexp_i[(f_i − C_{ij})/ε + log a_i]
        tmp_g = log_w_norm[:, tf.newaxis] + (f[:, tf.newaxis] - C) / eps  # (N, N)
        g     = -eps * tf.reduce_logsumexp(tmp_g, axis=0)                 # (N,)

        # ── 3. Log transport plan ─────────────────────────────────────
        #   log P_{ij} = (f_i + g_j − C_{ij})/ε + log a_i + log b_j
        log_P = (
            f[:, tf.newaxis] + g[tf.newaxis, :] - C
        ) / eps + log_w_norm[:, tf.newaxis] + log_b[tf.newaxis, :]        # (N, N)

        P = tf.exp(log_P)                                                  # (N, N)

        # ── 4. Barycentric projection  x̃_j = N · Σ_i P_{ij} · x_i ────
        new_particles = tf.cast(N, DTYPE) * tf.linalg.matmul(
            P, particles, transpose_a=True
        )                                                                  # (N, nx)

        return new_particles, log_b   # uniform log-weights after resampling

    # ------------------------------------------------------------------
    # Update: cache observation, then run LEDH + neural Sinkhorn
    # ------------------------------------------------------------------

    @tf.function
    def update(
        self,
        state_pred:  Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        observation: tf.Tensor,
    ) -> Tuple[Tuple, tf.Tensor, tf.Tensor]:
        """
        Differentiable measurement update with neural Sinkhorn resampling.

        Replicates DiffSinkhornLEDHFilter.update() with one addition:
        self._current_obs is assigned the current observation before
        self.resample() is called, so the neural operator can condition on y_t.

        Step sequence
        -------------
        1. Cache observation  →  self._current_obs (tf.Variable.assign, XLA-safe)
        2. LEDH particle migration  (inherited from PFPF_LEDHFilter)
        3. Weight update: log w += log p(y|x) + log |det J_flow|
        4. Neural Sinkhorn resampling  (self.resample → neural_sinkhorn_resample)
        5. Point estimate (weighted mean)
        6. Auxiliary EKF update for the prediction covariance

        Returns
        -------
        updated_state : (particles, log_weights, m_upd, P_upd)
        x_est         : (nx,) weighted-mean point estimate
        metrics       : [ess, avg_flow_cond, ekf_S_cond, ekf_P_cond,
                         step_log_likelihood]
        """
        particles, log_weights, m_pred, P_pred = state_pred
        y_curr = tf.reshape(observation, [self.ny])

        # ── 1. Cache current observation for the neural resampling step ──
        self._current_obs.assign(y_curr)

        # ── 2. LEDH particle migration ────────────────────────────────
        particles, total_cond, log_det_J = self.particle_migration(
            particles, m_pred, P_pred, y_curr, log_weights
        )

        # ── 3. Weight update ──────────────────────────────────────────
        pred_obs_mean = self.h_func(particles)
        obs_t         = tf.reshape(observation, [1, self.ny])
        diff          = obs_t - pred_obs_mean
        log_lik       = self.obs_noise_dist.log_prob(diff)
        log_weights  += log_lik + log_det_J

        max_log_w           = tf.reduce_max(log_weights)
        step_log_likelihood = max_log_w + tf.math.log(
            tf.reduce_mean(tf.exp(log_weights - max_log_w))
        )

        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm     = tf.exp(log_w_norm)
        ess        = 1.0 / (tf.reduce_sum(w_norm ** 2) + 1e-12)

        # ── 4. Neural Sinkhorn OT resampling (always, unconditional) ──
        # self.resample() dispatches here, reads self._current_obs and
        # self.theta_ssm (both tf.Variables set before this call).
        particles, log_weights = self.resample(particles, log_weights)

        # ── 5. Point estimate ─────────────────────────────────────────
        log_w_norm_final = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm_final     = tf.exp(log_w_norm_final)
        x_est            = tf.reduce_sum(
            w_norm_final[:, tf.newaxis] * particles, axis=0
        )

        # ── 6. Auxiliary EKF update ───────────────────────────────────
        ekf_updated_state, _, ekf_metrics = self.ekf.update(
            (m_pred, P_pred), observation
        )
        m_upd, P_upd = ekf_updated_state

        avg_cond = total_cond / tf.cast(self.steps, DTYPE)
        metrics  = tf.concat(
            [[ess], [avg_cond], ekf_metrics, [step_log_likelihood]], axis=0
        )
        return (particles, log_weights, m_upd, P_upd), x_est, metrics

    # ------------------------------------------------------------------
    # Pretraining
    # ------------------------------------------------------------------

    def pretrain(
        self,
        steps:           int = 2000,
        batch_size:      int = None,
        n_theta_samples: int = 8,
    ) -> None:
        """
        Supervised pretraining of NeuralSinkhornPotentialNet.

        For each gradient step the network is trained to reproduce the
        f-potentials produced by the fixed Sinkhorn algorithm (parent class),
        across diverse (y_obs, θ_SSM) contexts.

        Training objective
        ------------------
        L = E_{x, y, θ} [ ||f_net(x, y, θ) − f*_Sinkhorn(x, w)|| ^2 ]

        where f*_Sinkhorn is the converged f-potential from ot_n_iter fixed
        Sinkhorn iterations.  This teaches the network to internalise the
        Sinkhorn map across the full range of SSM parameters seen by HMC.

        Args:
            steps          : Number of Adam gradient updates.
            batch_size     : Particle cloud size per step.  Defaults to
                             self.num_particles so the fixed Sinkhorn helper
                             (which uses self.num_particles) can be called.
            n_theta_samples: Different θ vectors per step to diversify context.
        """
        if batch_size is None:
            batch_size = self.num_particles

        print(f"Pretraining NeuralSinkhornPotentialNet — {steps} steps …")

        # Adapt context normalisation with representative (y, θ) samples
        sample_x  = self.process_noise_dist.sample(512)
        sample_y  = self.h_func(sample_x) + self.obs_noise_dist.sample(512)
        sample_th = tf.random.normal((512, self.theta_dim), dtype=DTYPE)
        context   = tf.concat([sample_y, sample_th], axis=1)
        self.neural_ot_net.context_norm.adapt(context)

        log_b = tf.fill(
            [batch_size], -tf.math.log(tf.cast(batch_size, DTYPE))
        )

        # Temporarily adjust num_particles if batch_size differs, then restore
        _orig_N = self.num_particles
        if batch_size != _orig_N:
            self.num_particles = batch_size

        for step in range(steps):
            # Simulate a particle cloud and observations
            x_true = (
                self.f_func(
                    tf.random.normal((batch_size, self.nx), stddev=2.5, dtype=DTYPE)
                )
                + self.process_noise_dist.sample(batch_size)
            )
            y_obs = self.h_func(x_true) + self.obs_noise_dist.sample(batch_size)

            # Simulate non-uniform log-weights (realistic post-update distribution)
            raw_w      = tf.random.normal((batch_size,), dtype=DTYPE)
            log_w_norm = raw_w - tf.reduce_logsumexp(raw_w)

            # Ground-truth f-potentials from fixed Sinkhorn (treated as a target,
            # not differentiated — tape only tracks neural_ot_net weights)
            diff_p   = x_true[:, tf.newaxis, :] - x_true[tf.newaxis, :, :]
            C        = tf.reduce_sum(diff_p ** 2, axis=-1)
            f_target, _ = self.sinkhorn_potentials(log_w_norm, log_b, C)
            f_target = tf.stop_gradient(f_target)

            # Sample diverse θ contexts for this step
            theta_batch = tf.random.normal((n_theta_samples, self.theta_dim), dtype=DTYPE)

            accumulated_grads = None
            total_loss        = tf.constant(0.0, dtype=DTYPE)

            for j in range(n_theta_samples):
                theta_j = theta_batch[j]
                y_j     = y_obs[0]   # representative observation for this cloud

                with tf.GradientTape() as tape:
                    f_pred = self.neural_ot_net(x_true, y_j, theta_j)
                    loss   = tf.reduce_mean(tf.square(f_pred - f_target))

                grads = tape.gradient(loss, self.neural_ot_net.trainable_variables)

                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    accumulated_grads = [
                        (g1 + g2) if (g1 is not None and g2 is not None)
                        else (g1 if g1 is not None else g2)
                        for g1, g2 in zip(accumulated_grads, grads)
                    ]
                total_loss += loss

            if accumulated_grads is not None:
                # Average gradients over θ samples
                avg_grads = [
                    g / tf.cast(n_theta_samples, DTYPE) if g is not None else g
                    for g in accumulated_grads
                ]
                self.optimizer.apply_gradients(
                    zip(avg_grads, self.neural_ot_net.trainable_variables)
                )

            if step % 500 == 0:
                avg_loss = total_loss / tf.cast(n_theta_samples, DTYPE)
                print(f"  Step {step:>5}: f-potential MSE = {avg_loss.numpy():.6f}")

        # Restore original particle count if it was temporarily changed
        self.num_particles = _orig_N

    # ------------------------------------------------------------------
    # Override marginal log-likelihood accumulator for HMC compatibility
    # ------------------------------------------------------------------

    @tf.function
    def _compiled_marginal_log_likelihood(
        self,
        observations:  tf.Tensor,
        initial_state,
    ) -> tf.Tensor:
        """
        Accumulate marginal log-likelihood over observations for HMC.

        The parent DiffSinkhornLEDHFilter returns 5-element metrics:
            [ess, avg_flow_cond, ekf_S_cond, ekf_P_cond, step_log_likelihood]
        The base-class implementation incorrectly uses index [1].  This
        override reads the last element (step_log_likelihood), which is
        correct for both this filter and the parent.

        @tf.function (without jit_compile) converts the Python for-loop to a
        tf.while_loop, dispatching the entire T-step pass as a single GPU graph
        rather than T separate eager calls. Gradients flow through while_loop_grad.
        """
        T        = tf.shape(observations)[0]
        state    = initial_state
        total_ll = tf.constant(0.0, dtype=DTYPE)

        for t in tf.range(T):
            state_pred = self.predict(state)
            state, _, step_metrics = self.update(state_pred, observations[t])
            total_ll  += step_metrics[-1]   # step_log_likelihood (last element)

        return total_ll
