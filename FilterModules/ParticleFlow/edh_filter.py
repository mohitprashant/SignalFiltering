import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flow_base import ParticleFlow

tfd = tfp.distributions


class EDHFilter(ParticleFlow):
    """
    Deterministic Exact Daum-Huang Filter.
    """
    def __init__(self, model, num_particles=100, num_steps=30):
        """
        Initializes the EDH Filter.

        Args:
            model (Lorenz96Model): System model.
            num_particles (int): Number of particles.
            num_steps (int): Number of flow integration steps (lambda steps).
        """
        self.model = model
        self.N = num_particles
        self.steps = num_steps
        # Quadratic lambda schedule for numerical stability near lambda=0
        self.lambdas = tf.constant(np.linspace(0, 1, num_steps+1)**2, dtype=tf.float32)
        self.dlambdas = tf.constant(np.diff(np.linspace(0, 1, num_steps+1)**2), dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def run_step(self, particles, m_ekf, P_ekf, y_curr):
        """
        Executes one time step of the EDH filter.

        Args:
            particles (tf.Tensor): Current particles (N, K).
            m_ekf (tf.Tensor): Current EKF mean.
            P_ekf (tf.Tensor): Current EKF covariance.
            y_curr (tf.Tensor): Current observation.

        Returns:
            tuple: (particles, m_upd, P_upd, est, ess, avg_cond)
        """
        m_pred = self.model.rk4_step(m_ekf)                                        # Make prediction
        F = tf.squeeze(self.model.get_jacobian(tf.expand_dims(m_ekf, 0)))
        P_pred = F @ P_ekf @ tf.transpose(F) + self.model.Q
        
        particles = self.model.transition(particles)
        total_cond = 0.0
        
        for i in range(self.steps):                                                # Particle migration
            lam = self.lambdas[i]
            dlam = self.dlambdas[i]
            
            x_bar = tf.reduce_mean(particles, axis=0)                              # EDH: Global Linearization at Mean
            A, b, cond = self.compute_Ab_and_cond(P_pred, y_curr, x_bar, lam)
            total_cond += cond
            
            drift = tf.transpose(A @ tf.transpose(particles)) + b
            particles = particles + dlam * drift
            
        y_res = y_curr - m_pred
        S = P_pred + self.model.R
        K_gain = P_pred @ tf.linalg.inv(S)
        m_upd = m_pred + tf.linalg.matvec(K_gain, y_res)
        P_upd = (tf.eye(self.model.K) - K_gain) @ P_pred
        
        est = tf.reduce_mean(particles, axis=0)
        return particles, m_upd, P_upd, est, float(self.N), total_cond / self.steps