"""
experiment_nuts_mass_matrix.py

Compares four HMC mass matrix preconditioning schemes using NUTS
(No-U-Turn Sampler) as the proposal mechanism on the Stochastic
Volatility SSM.

Structure mirrors experiment_hmc_mass_matrix.py exactly; the only
difference is that NUTSMassMatrixHMC replaces MassMatrixHMC and the
number of leapfrog steps is determined automatically per iteration.

Parameters operate in UNCONSTRAINED space
-----------------------------------------
  logit(α) ∈ ℝ,  log(σ) ∈ ℝ,  log(β) ∈ ℝ
This removes hard parameter boundaries so NUTS can take large steps
without hitting walls near α = 0.91 ≈ boundary 1.
"""

import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Callable, Dict, List, Tuple

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter
from ParamEstimationPipeline.nuts_hmc import NUTSMassMatrixHMC

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU(s) available: {[g.name for g in gpus]}")
else:
    print("No GPU found — running on CPU")

DTYPE = tf.float32
tfd   = tfp.distributions
np.random.seed(42)
tf.random.set_seed(42)


# ======================================================================
# DIAGNOSTICS
# ======================================================================

def compute_diagnostics(
    results:      Dict,
    true_theta:   tf.Tensor,
    param_labels: List[str],
) -> Dict:
    """ESS (%), split-chain R-hat, RMSE, acceptance rate."""
    samples    = results['samples']
    samples_np = samples.numpy()
    S, d       = samples_np.shape

    ess     = tfp.mcmc.effective_sample_size(samples).numpy()
    ess_pct = ess / S * 100.0

    half = S // 2
    c1, c2 = samples_np[:half], samples_np[half:2*half]
    W      = (np.var(c1, axis=0, ddof=1) + np.var(c2, axis=0, ddof=1)) / 2.0
    B      = half * ((c1.mean(0) - (c1.mean(0)+c2.mean(0))/2)**2
                   + (c2.mean(0) - (c1.mean(0)+c2.mean(0))/2)**2)
    V_hat  = ((half - 1) / half) * W + B / half
    r_hat  = np.sqrt(V_hat / np.maximum(W, 1e-10))

    post_mean = samples_np.mean(axis=0)
    rmse      = float(np.sqrt(np.mean((post_mean - true_theta.numpy())**2)))

    return {
        'ess':             ess_pct,
        'r_hat':           r_hat,
        'rmse':            rmse,
        'post_mean':       post_mean,
        'acceptance_rate': results['acceptance_rate'],
        'time':            results['time'],
        'final_step_size': results.get('final_step_size', None),
    }


# ======================================================================
# COMPARISON RUNNER
# ======================================================================

SCHEME_COLORS = {
    'identity':   '#e07b39',
    'diagonal':   '#4e9fc7',
    'dense':      '#3aaa72',
    'riemannian': '#9b59b6',
}
SCHEME_LABELS = {
    'identity':   'Identity  M = I',
    'diagonal':   'Diagonal  M = diag(1/σ̂²)',
    'dense':      'Dense     M ≈ Σ̂⁻¹',
    'riemannian': 'Riemannian  M(θ) = diag(|∇U|² + λ)',
}


def compare_mass_schemes(
    ssm_builder:       Callable,
    filter_module:     SoftResamplingParticleFilter,
    prior_log_prob_fn: Callable,
    observations:      tf.Tensor,
    init_theta:        tf.Tensor,
    true_theta:        tf.Tensor,
    param_labels:      List[str],
    schemes:           Tuple[str, ...] = ('identity', 'diagonal', 'dense', 'riemannian'),
    num_warmup:        int   = 300,
    num_samples:       int   = 300,
    init_step_size:    float = 0.1,
    max_tree_depth:    int   = 4,
    target_acc:        float = 0.65,
    min_step_size:     float = 1e-3,
    riemannian_lambda: float = 0.01,
) -> Dict[str, Dict]:
    all_results = {}

    for scheme in schemes:
        print(f"\n{'='*65}")
        print(f"  SCHEME: {SCHEME_LABELS[scheme]}")
        print(f"{'='*65}")

        nuts = NUTSMassMatrixHMC(
            ssm_builder, filter_module, prior_log_prob_fn,
            max_tree_depth=max_tree_depth,
        )
        raw = nuts.run_warmup_and_chain(
            observations      = observations,
            init_theta        = init_theta,
            mass_scheme       = scheme,
            num_warmup        = num_warmup,
            num_samples       = num_samples,
            init_step_size    = init_step_size,
            num_leapfrog      = 1,          # ignored by NUTS
            target_acc        = target_acc,
            min_step_size     = min_step_size,
            riemannian_lambda = riemannian_lambda,
            step_size_jitter  = 0.0,        # NUTS varies L instead
        )

        diag = compute_diagnostics(raw, true_theta, param_labels)

        print(f"\n  ── Results ──────────────────────────────────────────")
        print(f"  Acceptance rate : {diag['acceptance_rate']:.2%}")
        print(f"  Final step size : {diag['final_step_size']:.4f}")
        print(f"  RMSE (post mean): {diag['rmse']:.4f}")
        for i, lbl in enumerate(param_labels):
            print(f"  {lbl:<15}: true={true_theta[i].numpy():.3f}  "
                  f"est={diag['post_mean'][i]:.3f}  "
                  f"ESS={diag['ess'][i]:.1f}%  "
                  f"R̂={diag['r_hat'][i]:.3f}")

        all_results[scheme] = {'raw': raw, 'diag': diag}

    return all_results


# ======================================================================
# VISUALISATION
# ======================================================================

def plot_comparison(
    all_results:  Dict[str, Dict],
    true_theta:   tf.Tensor,
    param_labels: List[str],
    schemes:      Tuple[str, ...],
) -> None:
    d        = len(param_labels)
    n_scheme = len(schemes)
    true_np  = true_theta.numpy()

    fig  = plt.figure(figsize=(5 * n_scheme, 4 * (d + 3)))
    outer = gridspec.GridSpec(d + 3, n_scheme, figure=fig,
                              hspace=0.55, wspace=0.35)

    for p_idx, lbl in enumerate(param_labels):
        for s_idx, scheme in enumerate(schemes):
            ax      = fig.add_subplot(outer[p_idx, s_idx])
            samples = all_results[scheme]['raw']['samples'].numpy()[:, p_idx]
            color   = SCHEME_COLORS[scheme]
            ax.plot(samples, color=color, alpha=0.7, lw=0.8)
            ax.axhline(true_np[p_idx], color='red', lw=1.5, ls='--', label='True')
            run_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
            ax.plot(run_mean, color='k', lw=1.2, ls=':', label='Running mean')
            ess = all_results[scheme]['diag']['ess'][p_idx]
            rh  = all_results[scheme]['diag']['r_hat'][p_idx]
            ax.set_title(f"{SCHEME_LABELS[scheme][:22]}\n"
                         f"{lbl}  ESS={ess:.1f}%  R̂={rh:.2f}", fontsize=8)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.2)
            if s_idx == 0:
                ax.set_ylabel(lbl, fontsize=9)

    for s_idx, scheme in enumerate(schemes):
        ax    = fig.add_subplot(outer[d, s_idx])
        trace = all_results[scheme]['raw'].get('step_size_trace', [])
        if len(trace):
            ax.semilogy(trace, color=SCHEME_COLORS[scheme], lw=1.2, alpha=0.8)
        eps_f = all_results[scheme]['diag']['final_step_size']
        ax.axhline(eps_f, color='k', ls='--', lw=1.2,
                   label=f"frozen ε={eps_f:.4f}")
        ax.set_title("Warmup: step size (dual avg)", fontsize=8)
        ax.set_xlabel("Warmup iteration")
        ax.set_ylabel("ε (log scale)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2, which='both')

    for s_idx, scheme in enumerate(schemes):
        ax     = fig.add_subplot(outer[d + 1, s_idx])
        acc_tr = all_results[scheme]['raw'].get('acc_rate_trace', [])
        if len(acc_tr):
            win  = max(1, len(acc_tr) // 10)
            roll = np.convolve(acc_tr, np.ones(win) / win, mode='valid')
            ax.plot(roll, color=SCHEME_COLORS[scheme], lw=1.2)
        ax.axhline(0.65, color='gray', ls='--', lw=1.2, label='Target 65%')
        ax.set_title(f"Chain acc={all_results[scheme]['diag']['acceptance_rate']:.2%}",
                     fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Rolling acceptance")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    ax_ess = fig.add_subplot(outer[d + 2, :n_scheme // 2])
    ax_rh  = fig.add_subplot(outer[d + 2, n_scheme // 2:])
    x       = np.arange(d)
    width   = 0.8 / n_scheme
    offsets = (np.arange(n_scheme) - n_scheme / 2 + 0.5) * width

    for s_idx, scheme in enumerate(schemes):
        color = SCHEME_COLORS[scheme]
        ax_ess.bar(x + offsets[s_idx], all_results[scheme]['diag']['ess'],
                   width, label=scheme, color=color, alpha=0.8)
        ax_rh.bar(x + offsets[s_idx],  all_results[scheme]['diag']['r_hat'],
                  width, label=scheme, color=color, alpha=0.8)

    ax_ess.set_xticks(x); ax_ess.set_xticklabels(param_labels)
    ax_ess.set_ylabel("ESS (% of samples)")
    ax_ess.set_title("ESS comparison")
    ax_ess.legend(fontsize=7)
    ax_ess.grid(True, alpha=0.2, axis='y')

    ax_rh.set_xticks(x); ax_rh.set_xticklabels(param_labels)
    ax_rh.axhline(1.1, color='red', ls='--', lw=1.2, label='R̂ = 1.1')
    ax_rh.set_ylabel("Split-chain R̂")
    ax_rh.set_title("R-hat comparison  (< 1.1 = converged)")
    ax_rh.legend(fontsize=7)
    ax_rh.grid(True, alpha=0.2, axis='y')

    plt.suptitle("NUTS Mass Matrix Comparison — SoftResampling Particle Filter",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig("nuts_mass_matrix_comparison.png", dpi=130, bbox_inches='tight')
    plt.show()
    print("Figure saved → nuts_mass_matrix_comparison.png")


def print_summary_table(
    all_results:  Dict[str, Dict],
    true_theta:   tf.Tensor,
    param_labels: List[str],
    schemes:      Tuple[str, ...],
) -> None:
    cols   = ['Scheme', 'acc rate', 'ε (final)', 'RMSE',
              *[f'ESS({l[:6]})' for l in param_labels],
              *[f'R̂({l[:6]})'   for l in param_labels]]
    widths = [24] + [10] * (len(cols) - 1)
    sep    = '─' * sum(w + 3 for w in widths)

    print(f"\n{'='*len(sep)}")
    print("  SUMMARY: NUTS Mass Matrix Comparison")
    print(f"{'='*len(sep)}")
    print(' | '.join(c.ljust(w) for c, w in zip(cols, widths)))
    print(sep)

    for scheme in schemes:
        diag = all_results[scheme]['diag']
        row  = [
            scheme,
            f"{diag['acceptance_rate']:.2%}",
            f"{diag['final_step_size']:.4f}",
            f"{diag['rmse']:.4f}",
            *[f"{e:.1f}%" for e in diag['ess']],
            *[f"{r:.3f}"  for r in diag['r_hat']],
        ]
        print(' | '.join(v.ljust(w) for v, w in zip(row, widths)))

    print(sep)


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":

    # ── Experiment parameters ─────────────────────────────────────────
    # N/T = 10 keeps Var[log p̂] = O(0.1) for coherent NUTS gradients.
    T              = 20
    N_PARTICLES    = 200    # N/T = 10
    MAX_TREE_DEPTH = 4
    NUM_WARMUP     = 300
    NUM_SAMPLES    = 300
    INIT_STEP      = 0.1    # larger start works in unconstrained space
    TARGET_ACC     = 0.65
    MIN_STEP       = 1e-4

    SCHEMES = ('identity', 'diagonal', 'dense')

    # ── Constrained SSM / prior ───────────────────────────────────────
    def sv_builder(theta: tf.Tensor):
        return StochasticVolatilityModel(
            alpha=theta[0], sigma=theta[1], beta=theta[2], static_diff=True)

    def sv_prior(theta: tf.Tensor) -> tf.Tensor:
        a, s, b = theta[0], theta[1], theta[2]
        if a <= 0.0 or a >= 1.0 or s <= 0.0 or b <= 0.0:
            return tf.constant(-np.inf, dtype=DTYPE)
        return (tfd.Normal(0.9, 0.1).log_prob(a)
                + tfd.Gamma(2.0, 2.0).log_prob(s)
                + tfd.Gamma(2.0, 2.0).log_prob(b))

    # ── Unconstrained reparametrisation ──────────────────────────────
    # φ = logit α,  ψ = log σ,  χ = log β  →  no hard boundaries.
    # Log-Jacobian: log|J| = log α + log(1−α) + log σ + log β

    def _to_constrained(theta_u: tf.Tensor) -> tf.Tensor:
        return tf.stack([tf.sigmoid(theta_u[0]),
                         tf.exp(theta_u[1]),
                         tf.exp(theta_u[2])])

    def sv_builder_u(theta_u: tf.Tensor):
        return sv_builder(_to_constrained(theta_u))

    def sv_prior_u(theta_u: tf.Tensor) -> tf.Tensor:
        theta_c  = _to_constrained(theta_u)
        prior_lp = sv_prior(theta_c)
        log_jac  = (tf.math.log(theta_c[0]) + tf.math.log(1.0 - theta_c[0])
                    + theta_u[1] + theta_u[2])
        return prior_lp + log_jac

    # ── True / initial theta ──────────────────────────────────────────
    _tc = [0.91, 1.0, 0.5]
    true_theta_c = tf.constant(_tc, dtype=DTYPE)
    true_theta   = tf.constant([
        math.log(_tc[0] / (1.0 - _tc[0])),   # logit(0.91) ≈  2.31
        math.log(_tc[1]),                      # log(1.0)   =   0.00
        math.log(_tc[2]),                      # log(0.5)   ≈ −0.69
    ], dtype=DTYPE)
    param_labels = ['logit(α)', 'log(σ)', 'log(β)']

    _ic = [0.85, 0.8, 0.6]
    init_theta = tf.constant([
        math.log(_ic[0] / (1.0 - _ic[0])),
        math.log(_ic[1]),
        math.log(_ic[2]),
    ], dtype=DTYPE)

    # ── 1. Data ───────────────────────────────────────────────────────
    print(f"--- Generating synthetic data (T={T}) ---")
    _ssm = sv_builder(true_theta_c)
    _, _raw_obs = _ssm.simulate(T)
    _pre   = _ssm.filter_components().get("preprocess_obs", lambda y: y)
    obs_tf = tf.constant(_pre(_raw_obs), dtype=DTYPE)

    # ── 2. Filter ─────────────────────────────────────────────────────
    soft_filter = SoftResamplingParticleFilter(
        num_particles = N_PARTICLES,
        label         = "SoftResampling (SV)",
    )
    soft_filter.load_ssm(sv_builder(true_theta_c))

    # ── 3. Compare ────────────────────────────────────────────────────
    all_results = compare_mass_schemes(
        ssm_builder       = sv_builder_u,
        filter_module     = soft_filter,
        prior_log_prob_fn = sv_prior_u,
        observations      = obs_tf,
        init_theta        = init_theta,
        true_theta        = true_theta,
        param_labels      = param_labels,
        schemes           = SCHEMES,
        num_warmup        = NUM_WARMUP,
        num_samples       = NUM_SAMPLES,
        init_step_size    = INIT_STEP,
        max_tree_depth    = MAX_TREE_DEPTH,
        target_acc        = TARGET_ACC,
        min_step_size     = MIN_STEP,
        riemannian_lambda = 0.01,
    )

    # ── 4. Output ─────────────────────────────────────────────────────
    print_summary_table(all_results, true_theta, param_labels, SCHEMES)
    plot_comparison(all_results, true_theta, param_labels, SCHEMES)
