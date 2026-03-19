import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import tracemalloc
from typing import Any, Dict, Optional, Tuple

from FilterModules.filter_base import BaseFilter
from FilterModules.KalmanFilters.extend_kalman import ExtendedKalmanFilter
from StateSpaceModels.ssm_base import SSM
from FilterModules.flow_base import ParticleFlow

tfd = tfp.distributions
DTYPE = tf.float32


class ExactDaumHuangFilter(BaseFilter, ParticleFlow):
    """
    Deterministic Exact Daum-Huang (EDH) Particle Flow Filter using the EKF for update and predictions.
    """
    def __init__(self, num_particles=100, num_steps=30, label=None):
        """
        Initializes the EDH Filter.

        Args:
            num_particles (int): The number of particles used to represent the state distribution. Defaults to 100.
            num_steps (int): The number of integration steps for the lambda flow from 0 to 1. Defaults to 30.
            label (str): A custom label for metric tracking.
        """
        label = label or f"EDH (N={num_particles}, Steps={num_steps})"
        BaseFilter.__init__(self, label=label)
        
        self.dtype = DTYPE
        self.num_particles = num_particles
        self.steps = num_steps
        self.ekf = ExtendedKalmanFilter(label="Auxiliary_EKF", joseph_form=True)
        lambdas = np.linspace(0, 1, num_steps + 1)**2
        self.lambdas = tf.constant(lambdas, dtype=self.dtype)
        self.dlambdas = tf.constant(np.diff(lambdas), dtype=self.dtype)


    def load_ssm(self, ssm_model: SSM) -> None:
        """
        Loads the State Space Model (SSM) into the internal EKF and initializes flow components.

        Args:
            ssm_model (SSM): The dynamical system model object to be tracked.
        """
        self.ekf.load_ssm(ssm_model)
        
        self.nx = self.ekf.nx
        self.ny = self.ekf.ny
        self.Q = self.ekf.Q
        self.R = self.ekf.R
        self.preprocess_obs = self.ekf.preprocess_obs
        self.eye_nx = self.ekf.eye_nx

        self.process_noise_dist = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.nx, dtype=self.dtype), 
            covariance_matrix=self.Q + tf.eye(self.nx, dtype=self.dtype) * 1e-6
        )
        
        self.R_inv = tf.linalg.inv(self.R + tf.eye(self.ny, dtype=self.dtype) * 1e-6)


    def initialize_state(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Initializes the particles and the internal EKF state at time t=0.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: A tuple containing:
                - particles: Tensor of shape (num_particles, nx) representing the initial cloud.
                - m_ekf: Tensor of shape (nx,) representing the initial EKF mean.
                - P_ekf: Tensor of shape (nx, nx) representing the initial EKF covariance.
        """
        m_ekf, P_ekf = self.ekf.initialize_state()
        
        init_dist = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros(self.nx, dtype=self.dtype), 
            covariance_matrix=self.ekf.P_init + self.eye_nx * 1e-6
        )
        particles = init_dist.sample(self.num_particles)
        
        return (particles, m_ekf, P_ekf)


    @tf.function(jit_compile=True)
    def predict(self, state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Prediction Step.
        
        Propagates both the internal EKF moments and the particle cloud forward in time.

        Args:
            state (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): The current state tuple containing 
                (particles, m_ekf, P_ekf).

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: The predicted state tuple for the next timestep.
        """
        particles, m_ekf, P_ekf = state
        m_pred, P_pred = self.ekf.predict((m_ekf, P_ekf))
        pred_means = self.ekf.f_func(particles)
        noise = self.process_noise_dist.sample(self.num_particles)
        particles_pred = pred_means + noise
        return (particles_pred, m_pred, P_pred)


    @tf.function(jit_compile=True)
    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Update Step.
        
        Migrates particles via the continuous Exact Daum-Huang flow equations over `num_steps`,
        and updates the auxiliary EKF covariance using standard extended Kalman equations.

        Args:
            state_pred (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): The predicted state tuple from the predict step.
            observation (tf.Tensor): The actual measurement obtained at the current timestep.

        Returns:
            Tuple[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]: A tuple containing:
                - updated_state: The new state tuple (particles, m_upd, P_upd).
                - est: A tensor representing the point estimate (mean of the migrated particles).
                - metrics: A 1D tensor tracking the condition numbers of the flow and EKF steps.
        """
        particles, m_pred, P_pred = state_pred
        y_curr = tf.reshape(observation, [self.ny])

        particles, total_cond = self.particle_migration(particles, m_pred, P_pred, y_curr)
            
        ekf_updated_state, _, ekf_metrics = self.ekf.update((m_pred, P_pred), observation)    # EKF update
        m_upd, P_upd = ekf_updated_state
        est = tf.reduce_mean(particles, axis=0)
        avg_cond = total_cond / tf.cast(self.steps, self.dtype)
        metrics = tf.concat([[avg_cond], ekf_metrics], axis=0)
        return (particles, m_upd, P_upd), est, metrics


    def particle_migration(self, particles, m_pred, P_pred, y_curr, local=False):
        """
        Particle Migration Loop (Optimized).
        """
        H = self.ekf.get_jacobian(self.ekf.h_func, m_pred)    # Precompute constants for XLA optimization
        HP = tf.matmul(H, P_pred)
        HPH = tf.matmul(HP, H, transpose_b=True)
        P_HT = tf.matmul(P_pred, H, transpose_b=True)
        R_inv_y = tf.linalg.matvec(self.R_inv, y_curr)
        P_HT_R_inv_y = tf.linalg.matvec(P_HT, R_inv_y)

        total_cond = tf.constant(0.0, dtype=self.dtype)
        
        for i in range(self.steps):                          # Migration loop
            lam = self.lambdas[i]
            dlam = self.dlambdas[i]
            
            if(local):                                       # LEDH
                x_bar = particles
            else:                                            # EDH
                x_bar = tf.reduce_mean(particles, axis=0)

            A, b, cond = self.compute_Ab_and_cond(
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

        return particles, total_cond


    @staticmethod
    def compute_omat(particles, weights, true_state):
        """
        Computes OMAT metric (Weighted Euclidean Error).
        Vectorized to process a full timeline tensor simultaneously.
        
        Args:
            particles (tf.Tensor): Particle cloud (T, N, K).
            weights (tf.Tensor): Weights (N,).
            true_state (tf.Tensor): Ground truth (T, 1, K).
            
        Returns:
            np.ndarray: OMAT values for the timeline.
        """
        diff = tf.norm(particles - true_state, axis=-1)
        return tf.reduce_sum(diff * weights, axis=-1).numpy()


    @tf.function(jit_compile=True)
    def _compiled_loop_edh(self, observations: tf.Tensor, initial_state: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Override the generic BaseFilter loop to track the full multi-dimensional 
        particle cloud across time, which is required for the OMAT calculation.
        
        Args:
            observations (tf.Tensor): The sequence of observations over time, shape (T, ny).
            initial_state (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): The starting state tuple at t=0.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Stacked TensorArrays containing the estimates, 
            step metrics, and the full particle history respectively.
        """
        T = tf.shape(observations)[0]
        estimates_ta = tf.TensorArray(dtype=self.dtype, size=T, clear_after_read=False)
        metrics_ta = tf.TensorArray(dtype=self.dtype, size=T, clear_after_read=False)
        particles_ta = tf.TensorArray(dtype=self.dtype, size=T, clear_after_read=False)
        state = initial_state

        for t in tf.range(T):
            state_pred = self.predict(state)
            state, estimate, step_metrics = self.update(state_pred, observations[t])
            estimates_ta = estimates_ta.write(t, estimate)
            metrics_ta = metrics_ta.write(t, step_metrics)
            particles_ta = particles_ta.write(t, state[0])

        return estimates_ta.stack(), metrics_ta.stack(), particles_ta.stack()


    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Executes the EDH loop, overrides BaseFilter to track particles and compute OMAT.
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
            'avg_flow_cond': np.mean(step_metrics_tensor.numpy()[:, 0]),
            'avg_ekf_S_cond': np.mean(step_metrics_tensor.numpy()[:, 1]),
            'avg_ekf_P_cond': np.mean(step_metrics_tensor.numpy()[:, 2]),
            'particles': self.num_particles,
            'steps': self.steps
        }
        
        self.metrics = results
        return results