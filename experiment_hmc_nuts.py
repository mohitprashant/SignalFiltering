"""
experiment_nuts.py

Runs NUTSAdaptiveHMC on the univariate Stochastic Volatility model.
Structured to mirror experiment_hmc.py so results are directly comparable.

Parameters (constrained space, same as experiment_hmc.py):
    theta = [alpha, sigma, beta]
    alpha ∈ (0, 1)  — AR persistence
    sigma > 0       — state noise std
    beta  > 0       — obs scale
"""

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from typing import Callable, Any, Dict, List

from ParamEstimationPipeline.nuts_adaptive_hmc import NUTSAdaptiveHMC
from StateSpaceModels.ssm_base import SSM
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from FilterModules.filter_base import BaseFilter
from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter

DTYPE = tf.float32
tfd   = tfp.distributions


def run_nuts_experiment(
    ssm_builder:        Callable[[tf.Tensor], SSM],
    filter_module:      BaseFilter,
    prior_log_prob_fn:  Callable[[tf.Tensor], tf.Tensor],
    true_theta:         tf.Tensor,
    init_theta:         tf.Tensor,
    param_labels:       List[str],
    T:                  int   = 200,
    num_iterations:     int   = 300,
    burn_in:            int   = 100,
    step_size:          float = 0.01,
    target_accept_rate: float = 0.65,
    max_tree_depth:     int   = 6,
):
    print(f"--- Generating synthetic data (T={T}) ---")
    true_ssm = ssm_builder(true_theta)
    true_states, observations = true_ssm.simulate(T)

    preprocess_fn        = true_ssm.filter_components().get("preprocess_obs", lambda y: y)
    processed_obs        = preprocess_fn(observations)

    print(f"\n--- Initialising NUTS (ε₀={step_size}, max_depth={max_tree_depth},"
          f" target_α={target_accept_rate}) ---")
    sampler = NUTSAdaptiveHMC(
        ssm_builder=ssm_builder,
        filter_module=filter_module,
        prior_log_prob_fn=prior_log_prob_fn,
        target_accept_rate=target_accept_rate,
        max_tree_depth=max_tree_depth,
    )

    results = sampler.run_chain(
        observations=processed_obs,
        init_theta=init_theta,
        num_iterations=num_iterations,
        burn_in=burn_in,
        step_size=step_size,
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    samples    = results['samples']          # post-warmup only
    samples_np = samples.numpy()
    num_params = samples_np.shape[1]

    ess_values = tfp.mcmc.effective_sample_size(samples).numpy()

    # Split-R̂ from two halves of the chain
    half_n       = samples_np.shape[0] // 2
    chain_1      = samples_np[:half_n]
    chain_2      = samples_np[half_n : 2 * half_n]
    W            = (np.var(chain_1, axis=0, ddof=1) + np.var(chain_2, axis=0, ddof=1)) / 2.0
    B            = half_n * ((np.mean(chain_1, axis=0) - np.mean(chain_2, axis=0)) ** 2) / 2.0
    r_hat_values = np.sqrt(((half_n - 1) / half_n * W + B / half_n) / (W + 1e-8))

    total_time    = results['time']
    iters_per_sec = num_iterations / total_time

    print("\n" + "=" * 65)
    print(" NUTS FINAL ESTIMATION RESULTS ")
    print("=" * 65)
    print(f"Total Iterations:   {num_iterations}  (Warmup: {burn_in})")
    print(f"Total Time:         {total_time:.2f} s  ({iters_per_sec:.1f} it/s)")
    print(f"Final step size ε:  {results['final_step_size']:.5f}")
    print(f"Mean tree depth:    {np.mean(results['tree_depths']):.2f}")
    print("-" * 65)
    print(f"{'Parameter':<20} | {'True':>6} | {'Est':>6} | {'ESS':>6} | {'R^':>5}")
    print("-" * 65)
    for i in range(num_params):
        print(
            f"{param_labels[i]:<20} | {true_theta[i].numpy():>6.3f}"
            f" | {np.mean(samples_np[:, i]):>6.3f}"
            f" | {ess_values[i]:>6.0f}"
            f" | {r_hat_values[i]:>5.2f}"
        )

    # ── Plotting ──────────────────────────────────────────────────────────────
    step_sizes  = results['step_sizes']
    tree_depths = results['tree_depths']
    colors      = plt.cm.tab10.colors

    n_rows = num_params + 3          # param traces + step-size + tree-depth + rolling ε
    fig, axs = plt.subplots(n_rows, 1, figsize=(11, 2.8 * n_rows))

    # — Trace plots -----------------------------------------------------------
    for i in range(num_params):
        c = colors[i % len(colors)]
        axs[i].plot(samples_np[:, i], color=c, alpha=0.7, linewidth=0.8)
        axs[i].axhline(
            true_theta[i].numpy(), color='red', linestyle='--',
            linewidth=2, label='True'
        )
        axs[i].set_title(
            f"Trace: {param_labels[i]}  |  ESS: {ess_values[i]:.0f}"
            f"  |  R̂: {r_hat_values[i]:.2f}"
        )
        axs[i].legend(loc='upper right', fontsize=8)
        axs[i].grid(True, alpha=0.3)

    # — Step-size trajectory --------------------------------------------------
    ax_ss = axs[num_params]
    ax_ss.plot(step_sizes, color='steelblue', linewidth=1.2)
    ax_ss.axvline(burn_in, color='gray', linestyle='--', linewidth=1.5,
                  label=f'Warmup ends ({burn_in})')
    ax_ss.set_title("Dual-Averaging Step Size ε")
    ax_ss.set_ylabel("ε")
    ax_ss.legend(fontsize=8)
    ax_ss.grid(True, alpha=0.3)

    # — Tree-depth trajectory -------------------------------------------------
    ax_td = axs[num_params + 1]
    ax_td.plot(tree_depths, color='darkorange', linewidth=0.8, alpha=0.8)
    ax_td.axhline(np.mean(tree_depths), color='black', linestyle='--',
                  linewidth=1.3, label=f'Mean = {np.mean(tree_depths):.1f}')
    ax_td.set_title("NUTS Tree Depth per Iteration")
    ax_td.set_ylabel("Depth")
    ax_td.legend(fontsize=8)
    ax_td.grid(True, alpha=0.3)

    # — Rolling acceptance proxy (chain-moved rate) ---------------------------
    ax_acc = axs[num_params + 2]
    moved       = np.any(samples_np[1:] != samples_np[:-1], axis=1).astype(float)
    window      = min(50, len(moved))
    rolling_mov = np.convolve(moved, np.ones(window) / window, mode='valid')
    x_roll      = np.arange(window - 1, len(moved))
    ax_acc.plot(x_roll, rolling_mov, color='purple', alpha=0.8,
                label=f'Rolling move rate (w={window})')
    ax_acc.axhline(target_accept_rate, color='gray', linestyle='--',
                   linewidth=1.5, label=f'Target α={target_accept_rate}')
    ax_acc.set_ylim(0, 1)
    ax_acc.set_title("Rolling Chain-Move Rate (Post-Warmup)")
    ax_acc.set_xlabel("Post-Warmup Iteration")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    # plt.savefig("nuts_sv_results.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    def sv_builder(theta: tf.Tensor) -> SSM:
        return StochasticVolatilityModel(
            alpha=theta[0], sigma=theta[1], beta=theta[2], static_diff=True
        )

    def sv_prior(theta: tf.Tensor) -> tf.Tensor:
        alpha, sigma, beta = theta[0], theta[1], theta[2]
        if alpha <= 0.0 or alpha >= 1.0 or sigma <= 0.0 or beta <= 0.0:
            return tf.constant(-np.inf, dtype=DTYPE)
        lp_alpha = tfd.Normal(loc=0.9, scale=0.1).log_prob(alpha)
        lp_sigma = tfd.Gamma(concentration=2.0, rate=2.0).log_prob(sigma)
        lp_beta  = tfd.Gamma(concentration=2.0, rate=2.0).log_prob(beta)
        return lp_alpha + lp_sigma + lp_beta

    true_theta_sv = tf.constant([0.91, 1.0, 0.5], dtype=DTYPE)
    init_theta_sv = tf.constant([0.5,  0.5, 0.5], dtype=DTYPE)
    param_labels_sv = ['alpha', 'sigma', 'beta']

    soft_res = SoftResamplingParticleFilter(
        num_particles=100, soft_alpha=0.5, label="SoftRes_PF_SV"
    )

    run_nuts_experiment(
        ssm_builder=sv_builder,
        filter_module=soft_res,
        prior_log_prob_fn=sv_prior,
        true_theta=true_theta_sv,
        init_theta=init_theta_sv,
        param_labels=param_labels_sv,
        T=200,
        num_iterations=300,
        burn_in=50,
        step_size=0.01,
        target_accept_rate=0.3,
        max_tree_depth=6,
    )
