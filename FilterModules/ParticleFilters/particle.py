import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Any, Dict, Optional, Tuple

from FilterModules.filter_base import BaseFilter
from StateSpaceModels.ssm_base import SSM

tfd = tfp.distributions
DTYPE = tf.float32


class ParticleFilter(BaseFilter):
    """
    Particle Filter (Sequential Monte Carlo).
    Inherits from BaseFilter to utilize standardized execution loops, optimized with XLA.
    """
    def __init__(self, num_particles=1000, resample_threshold_ratio=0.5, label=None):
        """
        Initializes the Particle Filter.

        Args:
            num_particles (int): The number of particles (N) to use for state estimation. Defaults to 1000.
            resample_threshold_ratio (float): The fraction of N below which adaptive resampling is triggered. 
                For example, 0.5 triggers resampling when the Effective Sample Size (ESS) drops below N/2.
            label (str): A custom string label for tracking metrics.
        """
        label = label or f"PF (N={num_particles}, Th={resample_threshold_ratio:.1f})"
        super().__init__(label=label)
        self.dtype = DTYPE
        self.num_particles = num_particles
        self.ratio = resample_threshold_ratio
        self.ess_threshold = tf.constant(num_particles * resample_threshold_ratio, dtype=self.dtype)


    def load_ssm(self, ssm_model: SSM) -> None:
        """
        Loads the State Space Model (SSM) and initializes the process and observation noise distributions.

        Extracts the dimensions, transition/measurement functions, and noise covariance matrices 
        from the provided SSM. It also adds a small jitter (1e-6) to the diagonals of the covariance 
        matrices to ensure stable Cholesky decompositions during sampling.

        Args:
            ssm_model (SSM): An instance of a state space model implementing the `filter_components` property.
        """
        comps = ssm_model.filter_components()
        self.nx, self.ny = comps["nx"], comps["ny"]
        self.f_func, self.h_func = comps["f_func"], comps["h_func"]
        self.Q, self.R, self.P_init = comps["Q"], comps["R"], comps["P_init"]
        self.preprocess_obs = comps["preprocess_obs"]

        self.process_noise_dist = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.nx, dtype=self.dtype), 
            covariance_matrix=self.Q + tf.eye(self.nx, dtype=self.dtype) * 1e-6
        )
        self.obs_noise_dist = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.ny, dtype=self.dtype), 
            covariance_matrix=self.R + tf.eye(self.ny, dtype=self.dtype) * 1e-6
        )


    def initialize_state(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initializes the particle filter state at t=0.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
                - particles: A tensor of shape `(num_particles, nx)` representing the initial states.
                - log_weights: A 1D tensor of shape `(num_particles,)` containing uniform log weights (-log(N)).
        """
        init_dist = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.nx, dtype=self.dtype), 
            covariance_matrix=self.P_init + tf.eye(self.nx, dtype=self.dtype) * 1e-6
        )
        particles = init_dist.sample(self.num_particles)
        log_weights = tf.fill([self.num_particles], -tf.math.log(float(self.num_particles)))
        return (particles, log_weights)


    def resample(self, particles: tf.Tensor, log_weights: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs systematic resampling to eliminate low-weight particles and multiply high-weight ones.

        Args:
            particles (tf.Tensor): The current tensor of particles, shape `(num_particles, nx)`.
            log_weights (tf.Tensor): Unnormalized log weights corresponding to the particles, shape `(num_particles,)`.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
                - resampled_particles: The newly selected particles.
                - new_log_weights: Reset uniform log weights (-log(N)) for the new particles.
        """
        log_weights_norm = log_weights - tf.reduce_logsumexp(log_weights)
        indices = tf.random.categorical(tf.expand_dims(log_weights_norm, 0), self.num_particles)
        resampled_particles = tf.gather(particles, tf.reshape(indices, [-1]))
        new_log_weights = tf.fill([self.num_particles], -tf.math.log(float(self.num_particles)))
        return resampled_particles, new_log_weights


    @tf.function(jit_compile=True)
    def predict(self, state: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get prediction.

        Propagates all particles forward in time using the deterministic state transition function `f_func`, 
        and adds sampled stochastic process noise.

        Args:
            state (Tuple[tf.Tensor, tf.Tensor]): The current `(particles, log_weights)` tuple.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The predicted state tuple `(particles_pred, log_weights)` for the next timestep.
        """
        particles, log_weights = state
        pred_mean = self.f_func(particles)
        noise = self.process_noise_dist.sample(self.num_particles)
        particles_pred = pred_mean + noise
        return (particles_pred, log_weights)


    @tf.function(jit_compile=True)
    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        The update step.

        Evaluates the likelihood of the current observation given the predicted particles, updates the 
        particle weights, calculates the Effective Sample Size (ESS), and triggers `tf.cond` resampling 
        if the ESS drops below the configured threshold.

        Args:
            state_pred (Tuple[tf.Tensor, tf.Tensor]): The predicted `(particles, log_weights)` from the `predict` step.
            observation (tf.Tensor): The actual measurement obtained at the current timestep.

        Returns:
            Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]: A tuple containing:
                - updated_state: The new `(particles, log_weights)` tuple after the update and potential resampling.
                - x_est: A 1D tensor representing the weighted mean point estimate of the state.
                - metrics: A 1D tensor containing the calculated ESS for this timestep.
        """
        particles, log_weights = state_pred
        pred_obs_mean = self.h_func(particles)
        obs_t = tf.reshape(observation, [1, self.ny])
        diff = obs_t - pred_obs_mean
        
        log_weights += self.obs_noise_dist.log_prob(diff)
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)                    # ESS Calculation
        w_norm = tf.exp(log_w_norm)
        ess = 1.0 / (tf.reduce_sum(w_norm**2) + 1e-12)
        
        particles, log_weights = tf.cond(                                              # Adaptive Resampling, using tf.cond for XLA graph
            ess < self.ess_threshold, 
            lambda: self.resample(particles, log_weights), 
            lambda: (particles, log_weights)
        )
        
        log_w_norm_final = log_weights - tf.reduce_logsumexp(log_weights)              # Output Extraction
        w_norm_final = tf.exp(log_w_norm_final)
        x_est = tf.reduce_sum(w_norm_final[:, tf.newaxis] * particles, axis=0)
        
        metrics = tf.stack([ess])                                                       # Add ESS to metrics
        return (particles, log_weights), x_est, metrics


    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Executes the Particle Filter over a sequence of observations.

        Pre-processes the observations and defers to the `BaseFilter` to manage the underlying 
        JIT-compiled loop, memory tracking and time profiling.

        Args:
            observations (tf.Tensor): A tensor of shape `(T, ny)` containing the observation sequence.
            true_states (Optional[tf.Tensor]): An optional tensor of ground truth states used to calculate RMSE.

        Returns:
            Dict[str, Any]: A dictionary containing comprehensive performance metrics including:
                - 'label': Filter label.
                - 'rmse': Root Mean Square Error (if `true_states` were provided).
                - 'time': Total runtime in seconds.
                - 'mem': Peak memory usage in bytes.
                - 'estimates': A stacked tensor of shape `(T, nx)` containing the estimated states.
                - 'ess_avg': The average Effective Sample Size across all timesteps.
                - 'particles': The configured number of particles.
                - 'threshold_ratio': The configured resampling threshold ratio.
        """
        processed_obs = self.preprocess_obs(observations)
        results = super().run_filter(processed_obs, true_states)
        
        if('step_metrics' in results):
            results['ess_avg'] = np.mean(results['step_metrics'][:, 0])
            
        results['particles'] = self.num_particles
        results['threshold_ratio'] = self.ratio
        return results