import tensorflow as tf
from typing import Tuple

from FilterModules.ParticleFlow.edh_flow import ExactDaumHuangFilter

DTYPE = tf.float32


class LocalizedExactDaumHuangFilter(ExactDaumHuangFilter):
    """
    Deterministic Localized Exact Daum-Huang (LEDH) Particle Flow Filter.
    
    Inherits from the ExactDaumHuangFilter.
    Computes the flow parameters (A, b) locally for each individual particle 
    (Local Linearization) rather than globally at the mean of the particle cloud.
    """
    def __init__(self, num_particles=100, num_steps=30, label=None):
        """
        Initializes the LEDH Filter.

        Args:
            num_particles (int): The number of particles used to represent the state distribution.
            num_steps (int): The number of integration steps for the lambda flow from 0 to 1.
            label (str): A custom label for metric tracking.
        """
        label = label or f"LEDH (N={num_particles}, Steps={num_steps})"
        super().__init__(num_particles=num_particles, num_steps=num_steps, label=label)


    def particle_migration(self, particles, m_pred, P_pred, y_curr, local=True):
        """
        Particle Migration Loop (Optimized). Overrides EDH to a minor extent. Very minor extent.
        """
        return super().particle_migration(particles, m_pred, P_pred, y_curr, local)

