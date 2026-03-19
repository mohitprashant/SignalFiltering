import tensorflow as tf
from FilterModules.ParticleFilters.particle import ParticleFilter

DTYPE = tf.float32



class SoftResamplingParticleFilter(ParticleFilter):
    """
    Particle Filter using Soft Resampling.
    """
    def __init__(self, num_particles=100, soft_alpha=0.5, label=None):
        """
        Initializes the Soft Resampling Particle Filter.

        Args:
            num_particles (int): Number of particles (N).
            soft_alpha (float): The mixture parameter (0 <= α <= 1). 
                a=1 is standard multinomial resampling; 
                a=0 is sampling from a uniform distribution.
            label (str): Optional custom string label for tracking metrics.
        """
        label = label or f"Soft PF (N={num_particles}, a={soft_alpha:.2f})"

        super().__init__(num_particles=num_particles, resample_threshold_ratio=2.0, label=label)
        self.soft_alpha = tf.convert_to_tensor(soft_alpha, dtype=self.dtype)


    def resample(self, particles: tf.Tensor, log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Performs the Soft Resampling step with importance weight correction.

        This method generates a new particle set by sampling indices according 
        to a 'soft' proposal distribution and then updates the weights to 
        account for the discrepancy between the proposal and the target 
        filtering distribution.

        Args:
            particles (tf.Tensor): Current particle states.
            log_weights (tf.Tensor): Unnormalized log-weights.

        Returns:
            tuple:
                - new_particles (tf.Tensor): Resampled particle states.
                - new_log_weights (tf.Tensor): Corrected log-weights.
        """
        N = self.num_particles
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)    # Normalize weights
        W_t = tf.exp(log_w_norm)
        
        uniform_w = 1.0 / float(N)
        W_tilde = self.soft_alpha * W_t + (1.0 - self.soft_alpha) * uniform_w         # Create Soft Mixture Weights
        W_tilde = tf.reshape(W_tilde, [N])
        
        log_W_tilde = tf.math.log(W_tilde + 1e-9)                                      # Resample indices based on W_tilde
        indices = tf.random.categorical(log_W_tilde[tf.newaxis, :], N)[0]
        new_particles = tf.gather(particles, indices)
        
        W_t_sel = tf.gather(W_t, indices)                                               # Importance Weight Correction
        W_tilde_sel = tf.gather(W_tilde, indices)
        
        new_log_weights = tf.math.log(W_t_sel / (W_tilde_sel + 1e-9) + 1e-9)
        new_log_weights = tf.reshape(new_log_weights, [N])
        
        new_log_weights = new_log_weights - tf.reduce_logsumexp(new_log_weights)        # Renormalize new weights to ensure stability in future steps
        return new_particles, new_log_weights