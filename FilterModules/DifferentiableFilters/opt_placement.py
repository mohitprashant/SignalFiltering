import tensorflow as tf

from FilterModules.ParticleFilters.particle import ParticleFilter

DTYPE = tf.float32


class OptimalPlacementParticleFilter(ParticleFilter):
    """
    Differentiable Particle Filter using Optimal Placement (OPR) Resampling.
    Inherits from ParticleFilter to utilize XLA-compiled loops and standardized SSM loading.
    """

    def __init__(self, num_particles=100, label=None):
        """
        Initializes the OPR Particle Filter.

        Args:
            num_particles (int): Number of particles (N).
            label (str): Optional custom string label for tracking metrics.
        """
        label = label or f"OPR PF (N={num_particles})"
        super().__init__(num_particles=num_particles, resample_threshold_ratio=2.0, label=label)


    def inverse_cdf_transform(self, x_sorted: tf.Tensor, w_sorted: tf.Tensor) -> tf.Tensor:
        """
        Maps sorted particles to new positions using a differentiable inverse CDF.

        This function approximates the inverse of the empirical CDF through linear 
        interpolation. Extrapolates tails using an exponential distribution assumption.
        Upgraded to seamlessly broadcast over multi-dimensional state spaces (N, nx).

        Args:
            x_sorted (tf.Tensor): Particle states sorted in ascending order, shape (N, nx).
            w_sorted (tf.Tensor): Corresponding normalized weights, shape (N,).

        Returns:
            tf.Tensor: New particles 'placed' at uniform quantiles (1/2N, 3/2N, ...).
        """
        N = self.num_particles
        w_sorted = tf.maximum(w_sorted, 1e-9)
        w_sorted = w_sorted / tf.reduce_sum(w_sorted)
        w_cumsum = tf.cumsum(w_sorted)
        cdf_vals = w_cumsum - w_sorted / 2.0
        
        u_targets = (2.0 * tf.range(1, N + 1, dtype=self.dtype) - 1.0) / (2.0 * float(N))      # Target uniform quantiles
        u_targets_ext = u_targets[:, tf.newaxis]                                            # Shape: (N, 1)
        w_1, x_1 = w_sorted[0], x_sorted[0]
        w_N, x_N = w_sorted[-1], x_sorted[-1]
        
        left_mask = u_targets <= (w_1 / 2.0)
        right_mask = u_targets >= (1.0 - w_N / 2.0)
        
        x_left = x_1[tf.newaxis, :] + tf.math.log(2.0 * u_targets_ext / w_1 + 1e-9)         # Exponential tail approximations
        val_right = tf.maximum(2.0 * (1.0 - u_targets_ext) / w_N, 1e-9)
        x_right = x_N[tf.newaxis, :] - tf.math.log(val_right)
        
        indices = tf.clip_by_value(tf.searchsorted(cdf_vals, u_targets), 1, N - 1)          # Linear Interpolation Step
        
        x_lo = tf.gather(x_sorted, indices - 1)                                             
        x_hi = tf.gather(x_sorted, indices)                                               
        c_lo = tf.gather(cdf_vals, indices - 1)[:, tf.newaxis]
        c_hi = tf.gather(cdf_vals, indices)[:, tf.newaxis]
        
        slope = (x_hi - x_lo) / tf.maximum(c_hi - c_lo, 1e-9)
        x_interp = x_lo + (u_targets_ext - c_lo) * slope
        
        x_final = tf.where(right_mask[:, tf.newaxis], x_right, x_interp)                     # Apply boundary conditions
        x_final = tf.where(left_mask[:, tf.newaxis], x_left, x_final)
        return x_final


    def resample(self, particles: tf.Tensor, log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Deterministic, differentiable resampling via Optimal Placement.

        Args:
            particles (tf.Tensor): Current particle states of shape (N, nx).
            log_weights (tf.Tensor): Current unnormalized log-weights.

        Returns:
            tuple: 
                - new_particles (tf.Tensor): Repositioned particles.
                - new_log_weights (tf.Tensor): Reset uniform weights.
        """
        N = self.num_particles
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        weights = tf.exp(log_w_norm)
        perm = tf.argsort(particles[:, 0])                                        # Sort particles to define the CDF                                          
        x_sorted = tf.gather(particles, perm)
        w_sorted = tf.gather(weights, perm)
        new_particles = self.inverse_cdf_transform(x_sorted, w_sorted)            # Apply Inverse CDF Transform
        new_log_weights = tf.fill([N], -tf.math.log(float(N)))
        return new_particles, new_log_weights