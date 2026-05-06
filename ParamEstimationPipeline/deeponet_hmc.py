"""
ParamEstimationPipeline/deeponet_hmc.py

DeepONet-aware HMC wrapper.

DeepONetHMC extends the base HMC class with two additions required by
DeepONetSinkhornLEDHFilter:

  1. filter.set_theta(theta) is called after filter.load_ssm(ssm) at each
     leapfrog step so the neural transport plan conditions on the current
     parameter proposal.

  2. filter._compiled_marginal_log_likelihood is used instead of
     _compiled_loop, because the DeepONet filter returns a 5-element
     metrics vector and the base-class loop indexes [1] (wrong for this
     filter); the override indexes [-1] (step_log_likelihood).
"""

import numpy as np
import tensorflow as tf
from typing import Callable, Tuple

from ParamEstimationPipeline.hmc_pipeline import HMC

DTYPE = tf.float32


class DeepONetHMC(HMC):
    """
    HMC subclass for DeepONetSinkhornLEDHFilter.

    Overrides _compute_log_prob_and_grad to call set_theta after load_ssm
    and to route through _compiled_marginal_log_likelihood.
    """

    def _compute_log_prob_and_grad(
        self,
        theta:         tf.Tensor,
        observations:  tf.Tensor,
        initial_state  = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        initial_state: if provided, skip initialize_state() and reuse the
        supplied particle cloud. Pass a fixed cloud sampled once per HMC
        proposal so all leapfrog steps integrate the same Hamiltonian.
        """
        with tf.GradientTape() as tape:
            tape.watch(theta)
            prior_lp = self.prior_log_prob_fn(theta)
            if not tf.math.is_finite(prior_lp):
                return tf.constant(-np.inf, dtype=DTYPE), tf.zeros_like(theta)
            ssm_model = self.ssm_builder(theta)
            self.filter_module.load_ssm(ssm_model)
            if hasattr(self.filter_module, 'set_theta'):
                self.filter_module.set_theta(theta)
            if initial_state is None:
                initial_state = self.filter_module.initialize_state()
            log_likelihood = self.filter_module._compiled_marginal_log_likelihood(
                observations, initial_state
            )
            total_log_prob = log_likelihood + prior_lp

        grad = tape.gradient(total_log_prob, theta)
        if grad is None or tf.reduce_any(tf.math.is_nan(grad)):
            grad = tf.zeros_like(theta)
        return total_log_prob, grad
