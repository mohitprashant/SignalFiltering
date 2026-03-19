import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import tracemalloc
from typing import Any, Dict, Optional, Tuple

from FilterModules.ParticleFilters.particle import ParticleFilter
from FilterModules.ParticleFlow.edh_flow import ExactDaumHuangFilter

tfd = tfp.distributions
DTYPE = tf.float32


class PFPF_EDHFilter(ParticleFilter, ExactDaumHuangFilter):
    """
    Invertible Particle Flow Particle Filter (PF-PF) with Exact Daum-Huang Flow.
    Inherits from both ParticleFilter (for weight management and resampling) 
    and ExactDaumHuangFilter (for flow equations and EKF tracking).
    """
    def __init__(self, num_particles=100, num_steps=30, resample_threshold_ratio=0.5, label=None):
        """
        Initializes the PF-PF EDH Filter.

        Args:
            num_particles (int): The number of particles (N) to use for state estimation. Defaults to 100.
            num_steps (int): The number of integration steps for the lambda flow from 0 to 1. Defaults to 30.
            resample_threshold_ratio (float): The fraction of N below which adaptive resampling is triggered. 
                For example, 0.5 triggers resampling when the Effective Sample Size (ESS) drops below N/2.
            label (str, optional): A custom string label for tracking metrics. If None, auto-generated.
        """
        label = label or f"PF-PF EDH (N={num_particles}, Steps={num_steps})"
        ParticleFilter.__init__(self, num_particles=num_particles, resample_threshold_ratio=resample_threshold_ratio, label=label)
        ExactDaumHuangFilter.__init__(self, num_particles=num_particles, num_steps=num_steps, label=label)


    def load_ssm(self, ssm_model) -> None:
        """
        Loads the State Space Model (SSM) into both parent components.

        Initializes the deterministic functions, noise covariance matrices, and 
        tensor dimensions required by the auxiliary EKF and the Particle Filter.

        Args:
            ssm_model (SSM): An instance of a state space model implementing `filter_components`.
        """
        ExactDaumHuangFilter.load_ssm(self, ssm_model)
        ParticleFilter.load_ssm(self, ssm_model)


    def initialize_state(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Initializes the joint state required for the PF-PF at t=0.

        Returns: 
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]: A 4-tuple containing:
                - particles: Tensor of shape `(num_particles, nx)` representing initial states.
                - log_weights: Tensor of shape `(num_particles,)` with uniform log weights.
                - m_ekf: Tensor of shape `(nx,)` representing the initial auxiliary EKF mean.
                - P_ekf: Tensor of shape `(nx, nx)` representing the initial auxiliary EKF covariance.
        """
        particles, log_weights = ParticleFilter.initialize_state(self)
        _, m_ekf, P_ekf = ExactDaumHuangFilter.initialize_state(self)
        return (particles, log_weights, m_ekf, P_ekf)


    @tf.function(jit_compile=True)
    def predict(self, state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Propagates both the auxiliary EKF moments and the particles forward in time.

        Args:
            state (Tuple): The current state 4-tuple `(particles, log_weights, m_ekf, P_ekf)`.

        Returns:
            Tuple: The predicted state 4-tuple for the next timestep prior to the update step.
        """
        particles, log_weights, m_ekf, P_ekf = state
        m_pred, P_pred = self.ekf.predict((m_ekf, P_ekf))
        particles_pred, log_weights = ParticleFilter.predict(self, (particles, log_weights))
        return (particles_pred, log_weights, m_pred, P_pred)


    @tf.function(jit_compile=True)
    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Executes the measurement update step, encompassing particle migration, weight updates, and resampling.

        1. Migrates particles via EDH continuous flow towards the observation.
        2. Computes the likelihood of the observation and adjusts weights using the flow Jacobian determinant.
        3. Computes the Effective Sample Size (ESS) and triggers resampling if it falls below the threshold.
        4. Updates the auxiliary EKF with the observation.

        Args:
            state_pred (Tuple): The predicted state 4-tuple from the `predict` step.
            observation (tf.Tensor): The actual measurement obtained at the current timestep.

        Returns:
            Tuple: A 3-tuple containing:
                - updated_state: The new state 4-tuple after migration and potential resampling.
                - x_est: A 1D tensor representing the weighted mean point estimate of the state.
                - metrics: A 1D tensor of tracked metrics: `[ess, avg_flow_cond, ekf_S_cond, ekf_P_cond]`.
        """
        particles, log_weights, m_pred, P_pred = state_pred
        y_curr = tf.reshape(observation, [self.ny])

        particles, total_cond, log_det_J = self.particle_migration(particles, m_pred, P_pred, y_curr, log_weights)

        pred_obs_mean = self.h_func(particles)                    # Weight Update
        obs_t = tf.reshape(observation, [1, self.ny])
        diff = obs_t - pred_obs_mean
        log_lik = self.obs_noise_dist.log_prob(diff)
        log_weights += log_lik + log_det_J
        
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)         # ESS and Conditonal Resampling
        w_norm = tf.exp(log_w_norm)
        ess = 1.0 / (tf.reduce_sum(w_norm**2) + 1e-12)
        
        particles, log_weights = tf.cond(
            ess < self.ess_threshold, 
            lambda: self.resample(particles, log_weights), 
            lambda: (particles, log_weights)
        )
        
        log_w_norm_final = log_weights - tf.reduce_logsumexp(log_weights)            # Final point estimate extraction
        w_norm_final = tf.exp(log_w_norm_final)
        x_est = tf.reduce_sum(w_norm_final[:, tf.newaxis] * particles, axis=0)
        
        ekf_updated_state, _, ekf_metrics = self.ekf.update((m_pred, P_pred), observation)   # EKF Update 
        m_upd, P_upd = ekf_updated_state
        
        avg_cond = total_cond / tf.cast(self.steps, self.dtype)
        metrics = tf.concat([[ess], [avg_cond], ekf_metrics], axis=0)
        return (particles, log_weights, m_upd, P_upd), x_est, metrics


    def particle_migration(self, particles, m_pred, P_pred, y_curr, log_weights, local=False):
        """Overriding the EDH particle migration."""
        H = self.ekf.get_jacobian(self.ekf.h_func, m_pred)
        HP = tf.matmul(H, P_pred)
        HPH = tf.matmul(HP, H, transpose_b=True)
        P_HT = tf.matmul(P_pred, H, transpose_b=True)
        R_inv_y = tf.linalg.matvec(self.R_inv, y_curr)
        P_HT_R_inv_y = tf.linalg.matvec(P_HT, R_inv_y)
        
        total_cond = tf.constant(0.0, dtype=self.dtype)
        log_det_J = tf.constant(0.0, dtype=self.dtype)
        
        for i in range(self.steps):                               # Particle Flow Migration
            lam = self.lambdas[i]
            dlam = self.dlambdas[i]
            
            log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)             # Linearize at the weighted mean of the particles
            w_norm = tf.exp(log_w_norm)

            if(local):                                                              # LEDH
                x_bar = particles
            else:                                                                       # EDH
                x_bar = tf.reduce_sum(particles * w_norm[:, tf.newaxis], axis=0)
            
            A, b, cond = self.compute_Ab_and_cond(                                    # Use ParticleFlow for exact flow params
                P_HT=P_HT, 
                H=H, 
                HPH=HPH, 
                R=self.R, 
                P_HT_R_inv_y=P_HT_R_inv_y, 
                x_lin=x_bar, 
                lam=lam, 
                eye_nx=self.eye_nx
            )
            
            total_cond += cond
            drift = tf.transpose(tf.matmul(A, tf.transpose(particles))) + b
            particles = particles + dlam * drift
            log_det_J += dlam * tf.linalg.trace(A)                    # Liouville's theorem, d(log|J|)/dlam = Trace(A). 

        return particles, total_cond, log_det_J


    @tf.function(jit_compile=True)
    def _compiled_loop_edh(self, observations: tf.Tensor, initial_state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Overrides base loops to correctly map and store the expanded 4-tuple state, 
        and specifically tracks the full particle tensor across time for OMAT metric calculations.

        Args:
            observations (tf.Tensor): Sequence of observations, shape `(T, ny)`.
            initial_state (Tuple): The 4-tuple state at t=0.

        Returns:
            Tuple: Stacked `TensorArray`s containing point estimates, step metrics, and raw particle clouds over time.
        """
        return ExactDaumHuangFilter._compiled_loop_edh(self, observations, initial_state)


    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Executes the full PF-PF EDH pipeline over a sequence of observations.

        Pre-processes observations, initializes state, records time and peak memory, and 
        calls the optimized XLA loop. Calculates post-run performance metrics.

        Args:
            observations (tf.Tensor): A tensor of shape `(T, ny)` containing the observation sequence.
            true_states (Optional[tf.Tensor]): Ground truth states of shape `(T, nx)` used to calculate 
                RMSE and OMAT. If None, these metrics evaluate to 0.0.

        Returns:
            Dict[str, Any]: Comprehensive performance metrics dictionary containing:
                - 'label': String label.
                - 'rmse': Root Mean Square Error (if `true_states` provided).
                - 'omat': Weighted Euclidean Error metric calculated on full particle clouds.
                - 'time': Total computation runtime in seconds.
                - 'mem': Peak memory consumption in bytes.
                - 'estimates': Shape `(T, nx)` tracked point estimates.
                - 'ess_avg': Average Effective Sample Size across all timesteps.
                - 'avg_flow_cond', 'avg_ekf_S_cond', 'avg_ekf_P_cond': Matrix condition number metrics.
                - 'particles', 'steps', 'threshold_ratio': Filter configuration parameters.
        """
        processed_obs = self.preprocess_obs(observations)
        state = self.initialize_state()
        tracemalloc.start()
        start_time = time.time()
        
        estimates_tensor, step_metrics_tensor, particles_tensor = self._compiled_loop_edh(processed_obs, state)
        
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time = time.time() - start_time
        rmse = 0.0
        omat_avg = 0.0
        
        if(true_states is not None):
            rmse = np.sqrt(np.mean((true_states.numpy() - estimates_tensor.numpy())**2))
            true_states_expanded = tf.expand_dims(true_states, axis=1)
            weights = tf.fill([self.num_particles], 1.0 / self.num_particles)
            omat_vals = self.compute_omat(particles_tensor, weights, true_states_expanded)
            omat_avg = float(np.mean(omat_vals))

        results = {
            'label': self.label,
            'rmse': rmse,
            'omat': omat_avg,
            'time': total_time,
            'mem': peak_mem,
            'estimates': estimates_tensor,
            'ess_avg': np.mean(step_metrics_tensor.numpy()[:, 0]),
            'avg_flow_cond': np.mean(step_metrics_tensor.numpy()[:, 1]),
            'avg_ekf_S_cond': np.mean(step_metrics_tensor.numpy()[:, 2]),
            'avg_ekf_P_cond': np.mean(step_metrics_tensor.numpy()[:, 3]),
            'particles': self.num_particles,
            'steps': self.steps,
            'threshold_ratio': self.ratio
        }
        
        self.metrics = results
        return results
    
        