import tensorflow as tf
from typing import Tuple
from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter

DTYPE = tf.float32



class CNFParticleFilter(SoftResamplingParticleFilter):
    """
    Conditional Normalizing Flow Particle Filter (CNF-PF).
    """
    def __init__(self, num_particles=100, soft_alpha=0.5, label=None):
        """
        Initializes the CNF Particle Filter with neural networks for flow parameters.

        Args:
            num_particles (int): Number of particles (N).
            soft_alpha (float): Mixture parameter for soft resampling (0 <= α <= 1).
            label (str): Optional custom string label for tracking metrics.
        """
        label = label or f"CNF-PF (N={num_particles}, a={soft_alpha:.2f})"
        super().__init__(num_particles=num_particles, soft_alpha=soft_alpha, label=label)
        self.shift_net = None
        self.scale_net = None


    def load_ssm(self, ssm_model):
        """
        Loads the SSM and dynamically constructs the Neural Flow networks based 
        on the state and observation dimensions (nx, ny).
        """
        super().load_ssm(ssm_model)
        
        # Shift Network: Maps observation (ny) to State Shift (nx)
        self.shift_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='tanh', input_shape=(self.ny,)),
            tf.keras.layers.Dense(self.nx, kernel_initializer='zeros') 
        ])
        
        # Scale Network: Maps observation (ny) to State Log-Scale (nx)
        self.scale_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='tanh', input_shape=(self.ny,)),
            tf.keras.layers.Dense(self.nx, kernel_initializer='zeros') 
        ])


    @tf.function(jit_compile=True)
    def predict(self, state: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Predict Step for CNF. 
        Must retain the previous particles because the 
        Conditional Flow applies a transformation after the proposal sampling and 
        needs both states to evaluate the process noise likelihood.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: (particles_prev, hat_particles, log_weights)
        """
        particles_prev, log_weights = state
        pred_mean = self.f_func(particles_prev)
        noise = self.process_noise_dist.sample(self.num_particles)
        hat_particles = pred_mean + noise
        return (particles_prev, hat_particles, log_weights)

    
    def _apply_flow_transformation(self, hat_particles: tf.Tensor, obs_in: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies the neural flow transformation conditioned on the observation.
        """
        shift = self.shift_net(obs_in)
        log_scale = self.scale_net(obs_in)
        scale = tf.exp(log_scale)
        particles_new = hat_particles * scale + shift
        return particles_new, log_scale


    def _compute_log_weight_update(self, particles_prev: tf.Tensor, hat_particles: tf.Tensor, particles_new: tf.Tensor, obs_in: tf.Tensor, log_scale: tf.Tensor) -> tf.Tensor:
        """
        Evaluates dynamic and observation likelihoods to compute the log importance weight update.
        """
        pred_mean = self.f_func(particles_prev)
        log_p_dyn_new = self.process_noise_dist.log_prob(particles_new - pred_mean)        # log p(x_new | x_prev)
        log_p_dyn_hat = self.process_noise_dist.log_prob(hat_particles - pred_mean)        # log q(x_hat | x_prev) - proposal evaluate
        pred_obs_mean = self.h_func(particles_new)                                         # log p(y | x_new)

        diff = obs_in - pred_obs_mean
        log_lik = self.obs_noise_dist.log_prob(diff)
        log_det_J = tf.reduce_sum(log_scale, axis=-1)
        return log_p_dyn_new - log_p_dyn_hat + log_lik + log_det_J


    @tf.function(jit_compile=True)
    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Update Step for CNF.
        """
        particles_prev, hat_particles, log_weights = state_pred
        obs_in = tf.reshape(observation, [1, self.ny])

        particles_new, log_scale = self._apply_flow_transformation(hat_particles, obs_in)
        log_weight_update = self._compute_log_weight_update(particles_prev, hat_particles, particles_new, obs_in, log_scale)
        log_weights += log_weight_update

        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm = tf.exp(log_w_norm)
        ess = 1.0 / (tf.reduce_sum(w_norm**2) + 1e-12)
        
        particles, log_weights = tf.cond(
            ess < self.ess_threshold, 
            lambda: self.resample(particles_new, log_weights), 
            lambda: (particles_new, log_weights)
        )
        
        log_w_norm_final = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm_final = tf.exp(log_w_norm_final)
        x_est = tf.reduce_sum(w_norm_final[:, tf.newaxis] * particles, axis=0)
        metrics = tf.stack([ess])
        return (particles, log_weights), x_est, metrics