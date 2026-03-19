import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
DTYPE = tf.float32


class ParticleFlowKernel:
    """
    Computes the kernelized drift term for the Particle Flow Filter.
    """
    def __init__(self, alpha: float, kernel_type: str, num_particles: int):
        self.alpha = tf.constant(alpha, dtype=DTYPE)
        self.type = kernel_type
        self.num_particles = num_particles


    def compute_drift(self, particles: tf.Tensor, grad_log_p: tf.Tensor, B_diag: tf.Tensor) -> tf.Tensor:
        """
        Computes the kernelized drift approximation vector.

        Args:
            particles (tf.Tensor): Current particle states. Shape (N, nx).
            grad_log_p (tf.Tensor): Gradient of the log-homotopy (likelihood + prior). Shape (N, nx).
            B_diag (tf.Tensor): Diagonal of the particle covariance matrix. Shape (nx,).

        Returns:
            tf.Tensor: Computed drift vector for each particle. Shape (N, nx).
        """
        N_float = tf.cast(self.num_particles, DTYPE)
        diff = tf.expand_dims(particles, 1) - tf.expand_dims(particles, 0)
        diff_sq = tf.square(diff)                                      

        if(self.type == 'matrix'):
            scale = 2.0 * self.alpha * (B_diag + 1e-5)                      # (nx,)
            scale = tf.reshape(scale, [1, 1, -1])                           # (1, 1, nx)
            K_vals = tf.exp(-diff_sq / scale)                               # (N, N, nx)
            
            grad_exp = tf.expand_dims(grad_log_p, 0)                        # (1, N, nx)
            term1 = tf.reduce_sum(K_vals * grad_exp, axis=1)                    # (N, nx)
            
            div_K = K_vals * (diff / (scale * 0.5))                          # (N, N, nx)
            term2 = tf.reduce_sum(div_K, axis=1)                             # (N, nx)
            
            return (term1 + term2) / N_float
        
        else:                                                               # scalar
            dists = tf.reduce_sum(diff_sq, axis=-1)
            mask = tf.eye(self.num_particles, dtype=DTYPE) * 1e30
            dists_masked = dists + mask
            
            median_dist = tfp.stats.percentile(dists_masked, 50.0)
            scale_scalar = median_dist / (2.0 * tf.math.log(2.0))
            K_scalar = tf.exp(-dists / scale_scalar)
            
            grad_exp = tf.expand_dims(grad_log_p, 0)      
            K_broadcast = tf.expand_dims(K_scalar, -1)             
            term1 = tf.reduce_sum(K_broadcast * grad_exp, axis=1)  
            
            factor = diff / (scale_scalar * 0.5)                 
            div_K = K_broadcast * factor                      
            term2 = tf.reduce_sum(div_K, axis=1)
            
            return (term1 + term2) / N_float