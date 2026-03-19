import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Any, Dict, Optional, Tuple

from FilterModules.filter_base import BaseFilter
from FilterModules.kernel_flow_base import ParticleFlowKernel
from StateSpaceModels.ssm_base import SSM

tfd = tfp.distributions
DTYPE = tf.float32


class KernelizedParticleFlowFilter(BaseFilter):
    """
    Kernelized Particle Flow Filter (PFF) implementation.
    Inherits from BaseFilter to utilize standardized execution loops.
    """
    def __init__(self, num_particles=30, kernel_type='matrix', alpha=0.05, num_steps=20, dt=0.05, label=None):
        """
        Initializes the Kernelized Particle Flow Filter.

        Args:
            num_particles (int): The number of particles in the ensemble.
            kernel_type (str): The type of kernel to use for drift calculation.
            alpha (float): Scaling factor for the kernel bandwidth. Controls the 
                interaction radius between particles.
            num_steps (int): Number of Euler integration steps across pseudo-time.
            dt (float): Integration step size in pseudo-time.
            label (str): A custom string label for tracking metrics.
        """
        label = label or f"KPFF ({kernel_type}, N={num_particles})"
        super().__init__(label=label)
        
        self.dtype = DTYPE
        self.num_particles = num_particles
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.num_steps = num_steps
        self.dt = tf.constant(dt, dtype=self.dtype)

        self.kernel = ParticleFlowKernel(alpha=alpha, kernel_type=kernel_type, num_particles=num_particles)


    def load_ssm(self, ssm_model: SSM) -> None:
        """
        Loads components from the underlying State Space Model (SSM).

        Extracts the state dimensions, dynamic functions, covariance matrices, 
        and observation preprocessing logic. It also precomputes the inverse 
        of the observation noise matrix and initializes the process noise distribution.

        Args:
            ssm_model (SSM): An instance of a State Space Model.
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
        self.R_inv = tf.linalg.inv(self.R + tf.eye(self.ny, dtype=self.dtype) * 1e-6)


    def initialize_state(self) -> tf.Tensor:
        """
        Samples the initial particle cloud from the prior distribution at t=0.

        Returns:
            tf.Tensor: A tensor of shape (num_particles, nx) representing 
            the initial ensemble of particles.
        """
        init_dist = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.nx, dtype=self.dtype), 
            covariance_matrix=self.P_init + tf.eye(self.nx, dtype=self.dtype) * 1e-6
        )
        particles = init_dist.sample(self.num_particles)
        return particles


    @tf.function(jit_compile=True)
    def predict(self, state: tf.Tensor) -> tf.Tensor:
        """
        Prediction Step. Propagates particles forward in time.

        Applies the deterministic transition function f_func and injects 
        stochastic process noise to all particles.

        Args:
            state (tf.Tensor): The current particle cloud tensor of shape.

        Returns:
            tf.Tensor: The predicted particle states for the next timestep.
        """
        particles = state
        pred_mean = self.f_func(particles)
        noise = self.process_noise_dist.sample(self.num_particles)
        return pred_mean + noise


    @tf.function(jit_compile=True)
    def update(self, state_pred: tf.Tensor, observation: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Update Step. Integrates the particles across pseudo-time to assimilate the observation.

        Calculates the exact gradient of the log-likelihood and prior, then computes 
        a smooth drift term using the ParticleFlowKernel. The particles are migrated 
        incrementally via Euler integration using the calculated drift and step size dt.

        Args:
            state_pred (tf.Tensor): The predicted particle cloud of shape (num_particles, nx).
            observation (tf.Tensor): The current observation vector of shape (ny,).

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A 3-tuple containing:
                - x: The posterior particle cloud tensor after flow migration.
                - x_est: A 1D tensor representing the mean point estimate of the state.
                - metrics: A 1D tensor of tracked metrics, notably the particle `spread`.
        """
        x = state_pred
        y = tf.reshape(observation, [self.ny])
        mean_prior = tf.reduce_mean(x, axis=0)                        # Approximate Jacobian H at prior mean for observation grad stability
        mean_prior_exp = tf.expand_dims(mean_prior, 0)
        
        with tf.GradientTape() as tape:
            tape.watch(mean_prior_exp)
            h_mean = self.h_func(mean_prior_exp)
        H_bar = tf.squeeze(tape.batch_jacobian(h_mean, mean_prior_exp), axis=0)

        B_diag = tfp.stats.variance(x, sample_axis=0) + 1e-3                         # Particle variance for the kernel bandwidth
        D = tf.expand_dims(B_diag, 0)
        
        for _ in range(self.num_steps):
            h_x = self.h_func(x)
            innov = tf.expand_dims(y, 0) - h_x
            grad_lik = tf.matmul(tf.matmul(innov, self.R_inv), H_bar)                # Gradient of Log-Likelihood: H^T R^-1 (y - h(x))
            mean_curr = tf.reduce_mean(x, axis=0, keepdims=True)                      # Gradient of Prior: -inv(B) (x - mean)
            grad_pri = -(x - mean_curr) / (B_diag + 1e-6)
            grad_log_p = grad_lik + grad_pri                                          # Combine gradients and kernel for drift approx
            drift = self.kernel.compute_drift(x, grad_log_p, B_diag)
            x = x + self.dt * (D * drift)
            
        x_est = tf.reduce_mean(x, axis=0)
        spread = tf.sqrt(tf.reduce_mean(B_diag))
        metrics = tf.stack([spread])
        return x, x_est, metrics


    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Executes the filter sequence over multiple timesteps and extends tracking metrics.

        Args:
            observations (tf.Tensor): The sequence of observations, shape (T, ny).
            true_states (Optional[tf.Tensor]): Ground truth states used to calculate RMSE.

        Returns:
            Dict[str, Any]: A dictionary containing filter metrics including:
                - 'label', 'rmse', 'time', 'mem', 'estimates'
                - 'avg_spread': The average particle spread across all steps.
                - 'particles', 'kernel_type': Model hyperparameters.
        """
        processed_obs = self.preprocess_obs(observations)
        results = super().run_filter(processed_obs, true_states)
        
        if('step_metrics' in results):
            results['avg_spread'] = np.mean(results['step_metrics'][:, 0])
            
        results['particles'] = self.num_particles
        results['kernel_type'] = self.kernel_type
        return results