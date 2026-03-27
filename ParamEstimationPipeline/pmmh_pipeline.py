import time
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Any, Dict
from StateSpaceModels.ssm_base import SSM
from FilterModules.filter_base import BaseFilter

DTYPE = tf.float32
tfd = tfp.distributions

class PMMH:
    """
    Particle Marginal Metropolis-Hastings (PMMH) for parameter estimation.
    Designed to accept any filter inheriting from BaseFilter and any SSM.
    """
    def __init__(self, 
                 ssm_builder: Callable[[tf.Tensor], SSM], 
                 filter_module: BaseFilter, 
                 prior_log_prob_fn: Callable[[tf.Tensor], tf.Tensor], 
                 proposal_dist_fn: Callable[[tf.Tensor], tfd.Distribution]):  # type: ignore
        """
        Initializes the PMMH algorithm.
        
        Args:
            ssm_builder: A function taking a parameter tensor (theta) and returning an updated SSM instance.
            filter_module: An instantiated filter object (e.g., ParticleFilter).
            prior_log_prob_fn: Function returning the log prior probability of theta.
            proposal_dist_fn: Function taking current theta and returning a TFP Distribution for the proposal.
        """
        self.ssm_builder = ssm_builder
        self.filter_module = filter_module
        self.prior_log_prob_fn = prior_log_prob_fn
        self.proposal_dist_fn = proposal_dist_fn


    def _compute_marginal_likelihood(self, theta: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
        """
        Updates the SSM with the proposed parameters and runs the XLA-compiled filter loop 
        to get an unbiased estimate of the log marginal likelihood.
        """
        ssm_model = self.ssm_builder(theta)        # Update SSM with the new theta parameters
        self.filter_module.load_ssm(ssm_model)
        initial_state = self.filter_module.initialize_state()
        _, step_metrics = self.filter_module._compiled_loop(observations, initial_state)
        log_likelihood = tf.reduce_sum(step_metrics[:, 1]) 
        return log_likelihood


    def run_chain(self, 
                  observations: tf.Tensor, 
                  init_theta: tf.Tensor, 
                  num_iterations: int,
                  burn_in: int = 0) -> Dict[str, Any]:
        """
        Executes the PMMH MCMC chain.
        
        Args:
            observations: Tensor of observations shape (T, ny).
            init_theta: Starting parameters.
            num_iterations: Number of MCMC steps.
            burn_in: Number of initial samples to discard.
            
        Returns:
            Dictionary containing samples, log_probs, and acceptance rate.
        """
        print(f"Starting PMMH Chain for {num_iterations} iterations...")
        start_time = time.time()
        
        current_theta = init_theta
        current_prior = self.prior_log_prob_fn(current_theta)
        current_ll = self._compute_marginal_likelihood(current_theta, observations)

        theta_dim = tf.shape(init_theta)[0]
        samples_ta = tf.TensorArray(dtype=DTYPE, size=num_iterations, clear_after_read=False)
        log_probs_ta = tf.TensorArray(dtype=DTYPE, size=num_iterations, clear_after_read=False)
        accepted_ta = tf.TensorArray(dtype=tf.bool, size=num_iterations, clear_after_read=False)
        
        accept_count = 0
        
        for i in range(num_iterations):
            proposal_dist = self.proposal_dist_fn(current_theta)    # Propose new theta
            proposed_theta = proposal_dist.sample()
            proposed_prior = self.prior_log_prob_fn(proposed_theta)
            
            # Optimization: If prior is invalid (-inf), automatically reject before running the expensive filter
            if tf.math.is_finite(proposed_prior):
                proposed_ll = self._compute_marginal_likelihood(proposed_theta, observations)
                reverse_proposal_dist = self.proposal_dist_fn(proposed_theta)       # Calculate acceptance probability
                log_q_reverse = reverse_proposal_dist.log_prob(current_theta)
                log_q_forward = proposal_dist.log_prob(proposed_theta)
                
                # MH Ratio log alpha = [log p(y|*) + log p(*) + log q(old|*)] - [log p(y|old) + log p(old) + log q(*|old)]
                log_alpha = (proposed_ll + proposed_prior + log_q_reverse) - \
                            (current_ll + current_prior + log_q_forward)
                
                log_u = tf.math.log(tf.random.uniform([], dtype=DTYPE))
                is_accepted = log_u < log_alpha
            else:
                is_accepted = tf.constant(False)
                proposed_ll = current_ll
            
            if(is_accepted):
                current_theta = proposed_theta
                current_prior = proposed_prior
                current_ll = proposed_ll
                accept_count += 1
                
            samples_ta = samples_ta.write(i, current_theta)                        # Record trace
            log_probs_ta = log_probs_ta.write(i, current_ll + current_prior)
            accepted_ta = accepted_ta.write(i, is_accepted)
            
            if((i + 1) % 10 == 0):
                print(f"Iteration {i+1}/{num_iterations} | Acc Rate: {accept_count/(i+1):.3f} | LL: {current_ll.numpy():.2f}")

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