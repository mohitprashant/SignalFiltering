import tensorflow as tf
from typing import Tuple

from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter
from FilterModules.NeuralFilter.Components.gradnet import CondGradNet

DTYPE = tf.float32


class GradNetParticleFilter(SoftResamplingParticleFilter):
    """
    Particle Filter using a Pre-trained Gradient Network for Transport.
    """
    def __init__(self, num_particles=100, soft_alpha=0.5, lr=0.002, num_modules=4, label=None):
        """
        Initializes the GradNet Particle Filter.

        Args:
            num_particles (int): Number of particles (N).
            soft_alpha (float): Mixture parameter for soft resampling. 
            lr (float): Learning rate for the Adam optimizer during pretraining. 
            num_modules (int): Number of Conditional GNM layers to stack in the 
                transport network.
            label (str): Custom string label for metric tracking.
        """
        label = label or f"GradNet PF (N={num_particles})"
        super().__init__(num_particles=num_particles, soft_alpha=soft_alpha, label=label)
        self.lr = lr
        self.num_modules = num_modules
        self.net = None
        self.optimizer = None


    def load_ssm(self, ssm_model):
        """
        Loads the State Space Model (SSM) and dynamically constructs the gradient network.

        Initializes the CondGradNet and the optimizer, matching the network's 
        input and output dimensions to the state (nx) and observation (ny) 
        dimensions of the provided SSM.

        Args:
            ssm_model: The State Space Model object containing transition and 
                observation dynamics.
        """
        super().load_ssm(ssm_model)
        self.net = CondGradNet(nx=self.nx, num_modules=self.num_modules)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.net([tf.zeros((1, self.nx), dtype=self.dtype), tf.zeros((1, self.ny), dtype=self.dtype)])


    def pretrain(self, steps=2000, batch_size=64):
        """
        Pre-trains the transport network in a supervised manner using simulated data.

        Simulates ground truth trajectories and observations from the loaded SSM. 
        The network is trained to map noisy proposal particles back to the true 
        underlying state, conditioned on the observation.

        Args:
            steps (int, optional): Number of gradient descent steps to perform. Defaults to 2000.
            batch_size (int, optional): Number of samples per training batch. Defaults to 64.
        """
        print(f"Pretraining GradNetOT - {steps} steps...")
        dummy_noise = self.process_noise_dist.sample(1000)
        dummy_x_true = self.f_func(tf.zeros((1000, self.nx), dtype=self.dtype)) + dummy_noise
        dummy_y = self.h_func(dummy_x_true) + self.obs_noise_dist.sample(1000)
        self.net.y_norm.adapt(dummy_y)
        
        for step in range(steps):
            x_prev_true = tf.random.normal((batch_size, self.nx), mean=0.0, stddev=2.5, dtype=self.dtype) 
            x_true = self.f_func(x_prev_true) + self.process_noise_dist.sample(batch_size)
            y_obs = self.h_func(x_true) + self.obs_noise_dist.sample(batch_size)

            x_prev_est = x_prev_true + tf.random.normal(shape=tf.shape(x_prev_true), stddev=0.5, dtype=self.dtype)
            x_prop = self.f_func(x_prev_est) + self.process_noise_dist.sample(batch_size)
            
            with tf.GradientTape() as tape:
                x_mapped = self.net([x_prop, y_obs], return_jacobian=False)
                loss = tf.reduce_mean(tf.square(x_mapped - x_true))
                
            grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
            
            if(step % 500 == 0):
                print(f"Step {step}: MSE Loss = {loss.numpy():.4f}")


    @tf.function(jit_compile=True)
    def predict(self, state: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Executes the prediction step of the particle filter.

        Proposes new particles by passing the previous particles through the 
        SSM's transition dynamics and adding process noise.

        Args:
            state (Tuple[tf.Tensor, tf.Tensor]): A tuple of (particles, log_weights) 
                from the previous time step.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: (previous_particles, proposed_particles, log_weights)
        """
        particles_prev, log_weights = state
        pred_mean = self.f_func(particles_prev)
        pred_mean = tf.clip_by_value(pred_mean, -1e4, 1e4)             # Clip check to prevent rk4 from exploding
        
        noise = self.process_noise_dist.sample(self.num_particles)
        hat_particles = pred_mean + noise
        hat_particles = tf.clip_by_value(hat_particles, -1e4, 1e4)
        return (particles_prev, hat_particles, log_weights)


    def _apply_transport_map(self, hat_particles: tf.Tensor, obs_in: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Applies the neural transport map and calculates the analytical Jacobian log-determinant.

        Args:
            hat_particles (tf.Tensor): Proposed particles from the predict step.
            obs_in (tf.Tensor): Current observation vector.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: (mapped_particles, log_det_jacobian)
        """
        obs_rep = tf.repeat(obs_in, self.num_particles, axis=0)
        particles_new, jac = self.net([hat_particles, obs_rep], return_jacobian=True)
        particles_new = tf.clip_by_value(particles_new, -1e4, 1e4)
        jac = 0.5 * (jac + tf.linalg.matrix_transpose(jac))
        jac += tf.eye(self.nx, batch_shape=[self.num_particles], dtype=self.dtype) * 1e-3       # Add jitter to prevent exploding

        L = tf.linalg.cholesky(jac)
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L) + 1e-12), axis=-1)
        log_det = tf.where(tf.math.is_nan(log_det), tf.zeros_like(log_det), log_det)                       # Prevent explode
        return particles_new, log_det


    def _compute_log_weight_update(self, particles_prev: tf.Tensor, hat_particles: tf.Tensor, particles_new: tf.Tensor, obs_in: tf.Tensor, log_det: tf.Tensor) -> tf.Tensor:
        """
        Calculates the importance weight update via the Change of Variables formula.

        Evaluates the transition and observation likelihoods for the transformed 
        particles.

        Args:
            particles_prev (tf.Tensor): Particles from the previous time step.
            hat_particles (tf.Tensor): Untransformed proposal particles.
            particles_new (tf.Tensor): Transported particles.
            obs_in (tf.Tensor): Current observation.
            log_det (tf.Tensor): Log-determinant of the transport map's Jacobian.

        Returns:
            tf.Tensor: Safe, bounded log-weight updates for each particle.
        """
        pred_mean = self.f_func(particles_prev)
        pred_mean = tf.clip_by_value(pred_mean, -1e4, 1e4)

        obs_rep = tf.repeat(obs_in, self.num_particles, axis=0)
        log_p_new = self.process_noise_dist.log_prob(particles_new - pred_mean)
        log_p_hat = self.process_noise_dist.log_prob(hat_particles - pred_mean)
        
        pred_obs_mean = self.h_func(particles_new)
        pred_obs_mean = tf.clip_by_value(pred_obs_mean, -1e4, 1e4)
        diff = obs_rep - pred_obs_mean
        log_lik = self.obs_noise_dist.log_prob(diff)

        if(len(log_p_new.shape) > 1):
            log_p_new = tf.reduce_sum(log_p_new, axis=-1)
            log_p_hat = tf.reduce_sum(log_p_hat, axis=-1)
        if(len(log_lik.shape) > 1):
            log_lik = tf.reduce_sum(log_lik, axis=-1)

        log_p_new = tf.maximum(log_p_new, -1e5)                 # Prevent explode
        log_p_hat = tf.maximum(log_p_hat, -1e5)
        log_lik = tf.maximum(log_lik, -1e5)
        
        weight_update = log_p_new - log_p_hat + log_lik + log_det
        weight_update = tf.where(tf.math.is_nan(weight_update), tf.zeros_like(weight_update), weight_update)
        return tf.clip_by_value(weight_update, -100.0, 100.0)


    @tf.function(jit_compile=True)
    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Applies the transport map, updates particle weights using Change of Variables, 
        performs conditional soft-resampling, and extracts the state estimate.

        Args:
            state_pred (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): The predicted state 
                containing (previous_particles, proposed_particles, log_weights).
            observation (tf.Tensor): The current observation vector.

        Returns:
            Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]: A tuple containing:
                - The updated state (particles, log_weights).
                - The state estimate (weighted mean).
                - Auxiliary metrics (like ESS).
        """
        particles_prev, hat_particles, log_weights = state_pred
        obs_in = tf.reshape(observation, [1, self.ny])
        
        particles_new, log_det = self._apply_transport_map(hat_particles, obs_in)
        log_weight_update = self._compute_log_weight_update(particles_prev, hat_particles, particles_new, obs_in, log_det)
        log_weights += log_weight_update
        
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm = tf.exp(log_w_norm)
        ess = 1.0 / (tf.reduce_sum(w_norm**2) + 1e-12)
        
        particles, log_weights = tf.cond(                                 # Deprecated, resampling is force triggered by parent
            ess < self.ess_threshold, 
            lambda: self.resample(particles_new, log_weights), 
            lambda: (particles_new, log_weights)
        )
        
        log_w_norm_final = log_weights - tf.reduce_logsumexp(log_weights)
        w_norm_final = tf.exp(log_w_norm_final)

        w_norm_final = tf.where(
            tf.math.is_nan(w_norm_final), 
            tf.fill(tf.shape(w_norm_final), 1.0 / self.num_particles), 
            w_norm_final
        )

        x_est = tf.reduce_sum(w_norm_final[:, tf.newaxis] * particles, axis=0)
        metrics = tf.stack([ess])
        return (particles, log_weights), x_est, metrics