import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from typing import Callable, List

from StateSpaceModels.ssm_base import SSM
from ParamEstimationPipeline.pmmh_pipeline import PMMH 
from FilterModules.filter_base import BaseFilter

DTYPE = tf.float32
tfd = tfp.distributions


def run_pmmh_experiment(
    ssm_builder: Callable[[tf.Tensor], SSM],
    filter_module: BaseFilter,
    prior_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    proposal_dist_fn: Callable[[tf.Tensor], tfp.distributions.Distribution],   #type: ignore
    true_theta: tf.Tensor,
    init_theta: tf.Tensor,
    param_labels: List[str],
    T: int = 1000,
    num_iterations: int = 2000,
    burn_in: int = 100
):
    print(f"--- Generating synthetic data (T={T}) ---")
    true_ssm = ssm_builder(true_theta)
    true_states, observations = true_ssm.simulate(T)
    
    preprocess_fn = true_ssm.filter_components().get("preprocess_obs", lambda y: y)
    processed_observations = preprocess_fn(observations)
    
    print(f"\n--- Initializing PMMH with Filter: {filter_module.label} ---")
    pmmh_sampler = PMMH(
        ssm_builder=ssm_builder,
        filter_module=filter_module,
        prior_log_prob_fn=prior_log_prob_fn,
        proposal_dist_fn=proposal_dist_fn
    )
    
    
    results = pmmh_sampler.run_chain(                           # Run the standard unmodified chain
        observations=processed_observations,
        init_theta=init_theta,
        num_iterations=num_iterations,
        burn_in=burn_in
    )
    
    # ==========================================
    # POST-RUN MCMC DIAGNOSTICS (ESS & R-Hat)
    # ==========================================
    samples = results['samples']
    samples_np = samples.numpy()
    num_params = samples.shape[1]

    ess_values = tfp.mcmc.effective_sample_size(samples).numpy()
    
    half_n = samples_np.shape[0] // 2                     # R-hat : Slices the post-burn-in trace into two halves to check for stationarity
    chain_1 = samples_np[:half_n, :]
    chain_2 = samples_np[half_n:2*half_n, :]
    
    w1 = np.var(chain_1, axis=0, ddof=1)
    w2 = np.var(chain_2, axis=0, ddof=1)
    W = (w1 + w2) / 2.0
    
    mean1 = np.mean(chain_1, axis=0)
    mean2 = np.mean(chain_2, axis=0)
    overall_mean = (mean1 + mean2) / 2.0
    
    B = half_n * ((mean1 - overall_mean)**2 + (mean2 - overall_mean)**2) # Between-chain variance
    V_hat = ((half_n - 1) / half_n) * W + (1.0 / half_n) * B
    
    r_hat_values = np.sqrt(V_hat / (W + 1e-8)) # Final Split R-hat
    
    total_time = results['time']
    time_per_iter = (total_time / num_iterations) * 1000 
    iters_per_sec = num_iterations / total_time
    
    print("\n" + "="*65)
    print(" PMMH FINAL ESTIMATION RESULTS ")
    print("="*65)
    print(f"Total Iterations:   {num_iterations} (Burn-in: {burn_in})")
    print(f"Total Time:         {total_time:.2f} seconds ({iters_per_sec:.1f} it/s)")
    print(f"Acceptance Rate:    {results['acceptance_rate']:.2%}")
    print("-" * 65)
    print(f"{'Parameter':<25} | {'True':<6} | {'Est':<6} | {'ESS':<5} | {'R^':<5}")
    print("-" * 65)
    
    for i in range(num_params):
        print(f"{param_labels[i]:<25} | {true_theta[i].numpy():.3f}  | {np.mean(samples_np[:, i]):.3f}  | {ess_values[i]:>5.0f} | {r_hat_values[i]:>4.2f}")
        
    accepted_trace = results['accepted'].numpy().astype(float)         # Acceptance rate
    window = min(50, len(accepted_trace)) 
    rolling_acc = np.convolve(accepted_trace, np.ones(window)/window, mode='valid')

    # ==========================================
    # Plotting
    # ==========================================
    fig, axs = plt.subplots(num_params + 1, 1, figsize=(10, 2.5 * (num_params + 1)))
    colors = plt.cm.tab10.colors 
    
    for i in range(num_params):
        c = colors[i % len(colors)]
        axs[i].plot(samples_np[:, i], color=c, alpha=0.7)
        axs[i].axhline(true_theta[i].numpy(), color='red', linestyle='--', linewidth=2, label='True Value')
        axs[i].set_title(f"Trace: {param_labels[i]}  |  ESS: {ess_values[i]:.0f}  |  R^: {r_hat_values[i]:.2f}")
        axs[i].legend(loc="upper right")
        axs[i].grid(True, alpha=0.3)
        
    ax_acc = axs[-1]
    x_axis_rolling = np.arange(window - 1, len(accepted_trace))
    ax_acc.plot(x_axis_rolling, rolling_acc, color='purple', alpha=0.8, label="Rolling Acc. Rate")
    ax_acc.axhline(0.25, color='gray', linestyle='--', linewidth=2, label='Ideal RWMH (25%)')
    ax_acc.set_title(f"Rolling Acceptance Rate (Window = {window} steps)")
    ax_acc.set_ylim(0, 1)
    ax_acc.set_xlabel("MCMC Iteration (Post Burn-in)")
    ax_acc.legend(loc="upper right")
    ax_acc.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    from StateSpaceModels.stochastic_vol import StochasticVolatilityModel 
    from FilterModules.ParticleFilters.particle import ParticleFilter
    from FilterModules.ParticleFilters.ledh_particle import PFPF_LEDHFilter
    from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter
    
    def sv_builder(theta: tf.Tensor) -> SSM:
        return StochasticVolatilityModel(alpha=theta[0], sigma=theta[1], beta=theta[2])

    def sv_prior(theta: tf.Tensor) -> tf.Tensor:
        alpha, sigma, beta = theta[0], theta[1], theta[2]
        if alpha <= 0.0 or alpha >= 1.0 or sigma <= 0.0 or beta <= 0.0:
            return tf.constant(-np.inf, dtype=DTYPE)
        p_alpha = tfd.Normal(loc=0.9, scale=0.1).log_prob(alpha)
        p_sigma = tfd.Gamma(concentration=2.0, rate=2.0).log_prob(sigma)
        p_beta  = tfd.Gamma(concentration=2.0, rate=2.0).log_prob(beta)
        return p_alpha + p_sigma + p_beta

    def sv_proposal(theta: tf.Tensor) -> tfp.distributions.Distribution: #type: ignore
        step_sizes = tf.constant([0.01, 0.01, 0.01], dtype=DTYPE) 
        return tfd.MultivariateNormalDiag(loc=theta, scale_diag=step_sizes)

    true_theta_sv = tf.constant([0.91, 1.0, 0.5], dtype=DTYPE)
    init_theta_sv = tf.constant([0.5, 0.5, 0.5], dtype=DTYPE)
    param_labels_sv = ['alpha', 'sigma', 'beta']
    
    # pf_filter = ParticleFilter(num_particles=500, resample_threshold_ratio=1.0, label="PF_SV")
    # ledh_pfpf = PFPF_LEDHFilter(num_particles=100, resample_threshold_ratio=1.0, label="LEHD_PF_SV")
    soft_res = SoftResamplingParticleFilter(num_particles=500, soft_alpha=0.8, label="SoftRes_PF_SV")

    run_pmmh_experiment(
        ssm_builder=sv_builder,
        filter_module=soft_res,
        prior_log_prob_fn=sv_prior,
        proposal_dist_fn=sv_proposal,
        true_theta=true_theta_sv,
        init_theta=init_theta_sv,
        param_labels=param_labels_sv,
        T=1000,
        num_iterations=2000,
        burn_in=100
    )