import tensorflow as tf
from typing import Tuple

from FilterModules.ParticleFilters.edh_particle import PFPF_EDHFilter

DTYPE = tf.float32


class PFPF_LEDHFilter(PFPF_EDHFilter):
    """
    Deterministic Localized Exact Daum-Huang (LEDH) Particle Flow Particle Filter.
    
    Inherits from the PFPF_EDHFilter.
    Computes the flow parameters (A, b) locally for each individual particle 
    (Local Linearization) rather than globally at the mean of the particle cloud.
    """
    def __init__(self, num_particles=100, num_steps=30, resample_threshold_ratio=0.5, label=None):
        """
        Initializes the PF-PF LEDH Filter.

        Args:
            num_particles (int): The number of particles (N) to use for state estimation. Defaults to 100.
            num_steps (int): The number of integration steps for the lambda flow from 0 to 1. Defaults to 30.
            resample_threshold_ratio (float): The fraction of N below which adaptive resampling is triggered. 
                For example, 0.5 triggers resampling when the Effective Sample Size (ESS) drops below N/2.
            label (str, optional): A custom string label for tracking metrics. If None, auto-generated.
        """
        label = label or f"PF-PF LEDH (N={num_particles}, Steps={num_steps})"
        super().__init__(num_particles=num_particles, num_steps=num_steps, resample_threshold_ratio=resample_threshold_ratio, label=label)


    def particle_migration(self, particles, m_pred, P_pred, y_curr, log_weights, local=True):
        """
        Particle Migration Loop (Optimized). Overrides PFPF_EDH to a minor extent. Very minor extent.
        """
        return super().particle_migration(particles, m_pred, P_pred, y_curr, log_weights, local)

