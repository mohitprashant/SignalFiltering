import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from typing import Callable, Any, Dict, List, Tuple

from StateSpaceModels.ssm_base import SSM
from FilterModules.filter_base import BaseFilter

DTYPE = tf.float32
tfd = tfp.distributions


class HMC:
    """
    Hamiltonian Monte Carlo (HMC) for parameter estimation.
    Designed to accept any Differentiable Filter (e.g., SoftResamplingParticleFilter) 
    and any SSM. Computes gradients through the filter loop using tf.GradientTape.
    """
    def __init__(self, 
                 ssm_builder: Callable[[tf.Tensor], SSM], 
                 filter_module: BaseFilter, 
                 prior_log_prob_fn: Callable[[tf.Tensor], tf.Tensor]):
        """
        Initializes the HMC algorithm.
        
        Args:
            ssm_builder: A function taking a parameter tensor (theta) and returning an updated SSM instance.
            filter_module: An instantiated differentiable filter object.
            prior_log_prob_fn: Function returning the log prior probability of theta.
        """
        self.ssm_builder = ssm_builder
        self.filter_module = filter_module
        self.prior_log_prob_fn = prior_log_prob_fn


    def _compute_log_prob_and_grad(self, theta: tf.Tensor, observations: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]: #type: ignore
        """
        Computes the log posterior and its gradient with respect to the parameters.
        """
        with tf.GradientTape() as tape:
            tape.watch(theta)
            prior_lp = self.prior_log_prob_fn(theta)

            if(not tf.math.is_finite(prior_lp)):                               # Prevent explode
                return tf.constant(-np.inf, dtype=DTYPE), tf.zeros_like(theta)
            
            ssm_model = self.ssm_builder(theta)
            self.filter_module.load_ssm(ssm_model)
            initial_state = self.filter_module.initialize_state()
            
            # Leveraging the specific log-likelihood accumulation loop to prevent Tape memory explosion
            log_likelihood = self.filter_module._compiled_marginal_log_likelihood(observations, initial_state)
            total_log_prob = log_likelihood + prior_lp
            
        grad = tape.gradient(total_log_prob, theta)
        
        if(grad is None or tf.reduce_any(tf.math.is_nan(grad))):        # Explosion guard
            grad = tf.zeros_like(theta)
            
        return total_log_prob, grad


    def run_chain(self, 
                  observations: tf.Tensor, 
                  init_theta: tf.Tensor, 
                  num_iterations: int,
                  burn_in: int = 100,
                  step_size: float = 0.01,
                  num_leapfrog_steps: int = 10) -> Dict[str, Any]:
        """
        Executes the HMC chain.
        """
        print(f"Starting HMC Chain for {num_iterations} iterations...")
        start_time = time.time()
        
        current_theta = init_theta
        current_log_prob, current_grad = self._compute_log_prob_and_grad(current_theta, observations)
        
        theta_dim = tf.shape(init_theta)[0]
        samples_ta = tf.TensorArray(dtype=DTYPE, size=num_iterations, clear_after_read=False)
        log_probs_ta = tf.TensorArray(dtype=DTYPE, size=num_iterations, clear_after_read=False)
        accepted_ta = tf.TensorArray(dtype=tf.bool, size=num_iterations, clear_after_read=False)
        
        accept_count = 0
        
        for i in range(num_iterations):
            current_momentum = tf.random.normal(shape=[theta_dim], dtype=DTYPE)    # Sample momentum p ~ N(0, I)

            proposed_theta = current_theta                                            # Leapfrog Integration
            proposed_momentum = current_momentum + 0.5 * step_size * current_grad
            
            for step in range(num_leapfrog_steps):
                proposed_theta = proposed_theta + step_size * proposed_momentum
                proposed_log_prob, proposed_grad = self._compute_log_prob_and_grad(proposed_theta, observations)
                
                if(step != num_leapfrog_steps - 1):
                    proposed_momentum = proposed_momentum + step_size * proposed_grad
                    
            proposed_momentum = proposed_momentum + 0.5 * step_size * proposed_grad
            
            
            if(tf.math.is_finite(proposed_log_prob)):                             # Acceptance probability
                current_kinetic = 0.5 * tf.reduce_sum(current_momentum ** 2)
                proposed_kinetic = 0.5 * tf.reduce_sum(proposed_momentum ** 2)
                log_alpha = (proposed_log_prob - proposed_kinetic) - (current_log_prob - current_kinetic)
                log_u = tf.math.log(tf.random.uniform([], dtype=DTYPE))
                is_accepted = log_u < log_alpha
            else:
                is_accepted = tf.constant(False)

            if(is_accepted):
                current_theta = proposed_theta
                current_log_prob = proposed_log_prob
                current_grad = proposed_grad
                accept_count += 1
                
            samples_ta = samples_ta.write(i, current_theta)       # Record Trace
            log_probs_ta = log_probs_ta.write(i, current_log_prob)
            accepted_ta = accepted_ta.write(i, is_accepted)
            
            if((i + 1) % 10 == 0):
                print(f"Iteration {i+1}/{num_iterations} | Acc Rate: {accept_count/(i+1):.3f} | LL: {current_log_prob.numpy():.2f}")

        total_time = time.time() - start_time
        
        all_samples = samples_ta.stack()
        all_log_probs = log_probs_ta.stack()
        
        return {
            'samples': all_samples[burn_in:],
            'log_probs': all_log_probs[burn_in:],
            'accepted': accepted_ta.stack(),
            'acceptance_rate': accept_count / num_iterations,
            'time': total_time
        }
