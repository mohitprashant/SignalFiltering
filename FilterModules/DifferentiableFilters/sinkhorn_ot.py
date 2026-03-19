import tensorflow as tf

from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter

DTYPE = tf.float32


class SinkhornParticleFilter(SoftResamplingParticleFilter):
    """
    Differentiable Particle Filter using Sinkhorn (Entropy-Regularized OT) Resampling.
    """
    def __init__(self, num_particles=100, epsilon=0.5, n_iter=20, label=None):
        """
        Initializes the Sinkhorn Particle Filter.

        Args:
            num_particles (int): Number of particles (N).
            epsilon (float): Entropy regularization strength. Higher 
                values lead to smoother, more diffused transport.
            n_iter (int): Number of Sinkhorn iterations for convergence.
            label (str): Label for tracking.
        """
        label = label or f"Sinkhorn PF (N={num_particles}, eps={epsilon:.2f})"
        super().__init__(num_particles=num_particles, soft_alpha=1.0, label=label) # Dummy soft_alpha=1.0 since overriding resample()
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=self.dtype)
        self.n_iter = n_iter


    def sinkhorn_potentials(self, log_a: tf.Tensor, log_b: tf.Tensor, C: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Computes Sinkhorn potentials (f, g) using stabilized log-domain updates.
        
        Solves dual problem of entropy-regularized optimal transport to find 
        scaling factors that transform the cost matrix into a doubly stochastic transport plan.

        Args:
            log_a (tf.Tensor): Log-source weights (normalized particle weights).
            log_b (tf.Tensor): Log-target weights (usually uniform -log N).
            C (tf.Tensor): Cost matrix (squared Euclidean distance between particles).

        Returns:
            tuple: (f, g) Sinkhorn dual potentials.
        """
        N = self.num_particles
        f = tf.zeros((N,), dtype=self.dtype)
        g = tf.zeros((N,), dtype=self.dtype)
        eps = self.epsilon
        
        for _ in range(self.n_iter):                                            
            tmp_f = log_b[tf.newaxis, :] + (g[tf.newaxis, :] - C) / eps      # Get primal     
            f_update = -eps * tf.reduce_logsumexp(tmp_f, axis=1)
            f = 0.5 * (f + f_update)
            
            tmp_g = log_a[:, tf.newaxis] + (f[:, tf.newaxis] - C) / eps       # Get dual
            g_update = -eps * tf.reduce_logsumexp(tmp_g, axis=0)
            g = 0.5 * (g + g_update)
            
        return f, g


    def resample(self, particles: tf.Tensor, log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Performs Soft Resampling via Optimal Transport.
        Overrides the mixture-based soft resampling in the parent class.
        
        Maps the current weighted particle set to a new set of particles with 
        uniform weights by applying the barycentric projection derived from the 
        optimal transport plan P.

        Args:
            particles (tf.Tensor): Current particle states of shape (N, nx).
            log_weights (tf.Tensor): Current unnormalized log-weights.

        Returns:
            tuple: (new_particles, new_log_weights)
        """
        N = self.num_particles
        log_weights = tf.maximum(log_weights, -1e9) 
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)            # Normalize weights
        log_b = tf.fill([N], -tf.math.log(float(N)))                           # Target weights
        diff = particles[:, tf.newaxis, :] - particles[tf.newaxis, :, :]       # Cost Matrix
        C = tf.reduce_sum(diff**2, axis=-1)

        f, g = self.sinkhorn_potentials(log_w_norm, log_b, C)
        
        log_P = (f[:, tf.newaxis] + g[tf.newaxis, :] - C) / self.epsilon + log_w_norm[:, tf.newaxis] + log_b[tf.newaxis, :]      
        P = tf.exp(log_P)                                                         # Transport Matrix in log domain
        
        # P = tf.debugging.check_numerics(P, "Transport Matrix P contains NaNs")
        new_particles = float(N) * tf.linalg.matmul(P, particles, transpose_a=True)     # Apply Transport
        new_log_weights = log_b
        return new_particles, new_log_weights