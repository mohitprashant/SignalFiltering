import tensorflow as tf

from FilterModules.NeuralFilter.gradnet_filter import GradNetParticleFilter
from FilterModules.NeuralFilter.Components.deeponet import DeepONetGradNet

DTYPE = tf.float32


class DeepONetParticleFilter(GradNetParticleFilter):
    """
    Particle Filter using a DeepONet for Transport.
    """
    def __init__(self, num_particles=100, soft_alpha=0.5, lr=0.002, num_basis=16, embed_dim=32, label=None):
        label = label or f"DeepONet PF (N={num_particles})"
        super().__init__(num_particles=num_particles, soft_alpha=soft_alpha, lr=lr, num_modules=num_basis, label=label)
        self.num_basis = num_basis
        self.embed_dim = embed_dim


    def load_ssm(self, ssm_model):
        """
        Overrides the GradNet initialization to load ssm.
        """
        super(GradNetParticleFilter, self).load_ssm(ssm_model)
        self.net = DeepONetGradNet(nx=self.nx, num_basis=self.num_basis, embed_dim=self.embed_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0)
        self.net([tf.zeros((1, self.nx), dtype=self.dtype), tf.zeros((1, self.ny), dtype=self.dtype)])
        
        
    def set_particle_count(self, new_N: int):
        """
        Updates the number of particles used by the filter at runtime.
        """
        self.num_particles = new_N
        self.ess_threshold = tf.convert_to_tensor(new_N * 2.0, dtype=self.dtype)