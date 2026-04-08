"""
experiment_deeponet_estimation.py

Full DeepONetSinkhornLEDH pipeline for SSM parameter estimation.

Compares four estimation settings on SVSSM and MSVSSM-3:
  - DeepONet filter only    — baseline filtering metrics
  - PMMH (via DeepONet)     — gradient-free MCMC
  - HMC  (via DeepONet)     — gradient-based HMC
  - HMC-NUTS (via DeepONet) — adaptive trajectory NUTS

Metrics tracked
---------------
  Filtering  : RMSE, avg per-step ESS, runtime, peak memory, OMAT
  MCMC chains: per-param ESS, per-param R_hat, acceptance rate,
               runtime, RMSE of posterior mean, trace convergence plots

OMAT (Optimal Mass Transport cost)
-----------------------------------
  Computed as the sqrt of the mean optimal assignment cost between the
  T-length estimate sequence and the T-length true state sequence using
  scipy's Hungarian algorithm.  For 1-D data this equals the empirical
  Wasserstein-2 distance between the two empirical distributions.

Notes on DeepONet integration
------------------------------
  * `load_ssm` is idempotent after the first call: the pretrained
    neural_ot_net weights survive SSM parameter updates.
  * `set_theta(theta)` must be called after `load_ssm` at each MCMC
    step so the neural transport plan conditions on the current proposal.
  * DeepONetPMMH overrides _compute_marginal_likelihood to use
    `_compiled_marginal_log_likelihood` (correct [-1] log-lik index)
    rather than _compiled_loop (whose base-class index [1] is wrong
    for this filter's 5-element metrics vector).
"""

import time
import numpy as np
import scipy.optimize
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple, Any

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel
from FilterModules.NeuralFilter.deeponet_sinkhorn_ledh import DeepONetSinkhornLEDHFilter
from ParamEstimationPipeline.pmmh_pipeline import PMMH
from ParamEstimationPipeline.hmc_pipeline import HMC
from ParamEstimationPipeline.nuts_adaptive_hmc import NUTSAdaptiveHMC

DTYPE = tf.float32
tfd   = tfp.distributions

np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel("ERROR")


# ======================================================================
# DEEPONET-AWARE INFERENCE WRAPPERS
# ======================================================================

class _DeepONetInferenceMixin:
    """
    Replaces _compute_log_prob_and_grad so that:
      1. _compiled_marginal_log_likelihood is used (correct [-1] indexing).
      2. set_theta(theta) is called after load_ssm to update conditioning.
    """

    def _compute_log_prob_and_grad(
        self,
        theta:        tf.Tensor,
        observations: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            prior_lp = self.prior_log_prob_fn(theta)
            if not tf.math.is_finite(prior_lp):
                return tf.constant(-np.inf, dtype=DTYPE), tf.zeros_like(theta)

            ssm_model = self.ssm_builder(theta)
            self.filter_module.load_ssm(ssm_model)
            self.filter_module.set_theta(theta)
            initial_state = self.filter_module.initialize_state()
            log_likelihood = self.filter_module._compiled_marginal_log_likelihood(
                observations, initial_state
            )
            total_log_prob = log_likelihood + prior_lp

        grad = tape.gradient(total_log_prob, theta)
        if grad is None or tf.reduce_any(tf.math.is_nan(grad)):
            grad = tf.zeros_like(theta)
        return total_log_prob, grad


class DeepONetHMC(_DeepONetInferenceMixin, HMC):
    """Standard HMC using DeepONetSinkhornLEDHFilter as differentiable likelihood."""
    pass


class DeepONetNUTS(_DeepONetInferenceMixin, NUTSAdaptiveHMC):
    """NUTS with dual-averaging using DeepONetSinkhornLEDHFilter."""
    pass


class DeepONetPMMH(PMMH):
    """
    PMMH using DeepONetSinkhornLEDHFilter as the particle filter.

    Overrides _compute_marginal_likelihood to:
      - Use _compiled_marginal_log_likelihood (step_metrics[-1] = log_likelihood)
        instead of _compiled_loop / step_metrics[:, 1] (wrong index for this filter).
      - Call set_theta after load_ssm so neural transport conditions on theta.
    """

    def _compute_marginal_likelihood(
        self,
        theta:        tf.Tensor,
        observations: tf.Tensor,
    ) -> tf.Tensor:
        ssm_model = self.ssm_builder(theta)
        self.filter_module.load_ssm(ssm_model)
        self.filter_module.set_theta(theta)
        initial_state = self.filter_module.initialize_state()
        return self.filter_module._compiled_marginal_log_likelihood(
            observations, initial_state
        )


# ======================================================================
# WEIGHT TRANSFER UTILITY
# ======================================================================

def _copy_weights(
    src: DeepONetSinkhornLEDHFilter,
    dst: DeepONetSinkhornLEDHFilter,
) -> None:
    """
    Copy all neural_ot_net trainable weights (and normalisation statistics)
    from a pretrained filter to a freshly-initialised filter instance.

    Both src and dst must have had load_ssm() called already so that
    the Keras layers exist and their variable shapes match.
    """
    for sw, dw in zip(src.neural_ot_net.weights, dst.neural_ot_net.weights):
        dw.assign(sw)


# ======================================================================
# METRICS HELPERS
# ======================================================================

def compute_omat(
    estimates_np:   np.ndarray,
    true_states_np: np.ndarray,
) -> float:
    """
    Optimal Mass Transport cost (matched RMSE).

    Computes the Hungarian-optimal assignment between the T-length estimate
    sequence and the T-length true-state sequence, where each time step
    contributes one point mass.  Returns the square root of the mean
    minimum-cost assignment cost (equivalent to W2 in 1D).

    For multivariate states the pairwise cost is squared Euclidean.
    The T×T assignment is O(T^3) via scipy; feasible for T ≤ 500.
    """
    T  = min(len(estimates_np), len(true_states_np))
    nx = estimates_np.shape[-1] if estimates_np.ndim > 1 else 1

    est = estimates_np[:T].reshape(T, nx).astype(np.float64)
    tru = true_states_np[:T].reshape(T, nx).astype(np.float64)

    diff   = est[:, np.newaxis, :] - tru[np.newaxis, :, :]   # (T, T, nx)
    C      = np.sum(diff ** 2, axis=-1)                        # (T, T)
    ri, ci = scipy.optimize.linear_sum_assignment(C)
    return float(np.sqrt(C[ri, ci].sum() / T))


def compute_mcmc_diagnostics(
    samples_np: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split-chain ESS and R_hat for post-warmup MCMC samples.

    Args:
        samples_np: Shape (N_post_warmup, P).

    Returns:
        ess_values  : (P,) effective sample sizes.
        r_hat_values: (P,) Gelman-Rubin split-chain R_hat.
    """
    samples_tf = tf.constant(samples_np, dtype=DTYPE)
    ess_values = tfp.mcmc.effective_sample_size(samples_tf).numpy()

    half_n = samples_np.shape[0] // 2
    c1, c2 = samples_np[:half_n], samples_np[half_n: 2 * half_n]
    W       = (np.var(c1, axis=0, ddof=1) + np.var(c2, axis=0, ddof=1)) / 2.0
    B       = half_n * ((np.mean(c1, axis=0) - np.mean(c2, axis=0)) ** 2) / 2.0
    r_hat   = np.sqrt(((half_n - 1) / half_n * W + B / half_n) / (W + 1e-8))
    return ess_values, r_hat


def run_filter_metrics(
    filter_module: DeepONetSinkhornLEDHFilter,
    ssm_model,
    observations:  tf.Tensor,
    true_states:   tf.Tensor,
) -> Dict[str, Any]:
    """
    Run DeepONet filter and return all performance metrics.

    Returns keys: rmse, ess (avg per-step), runtime (s), memory (MB), omat.

    Implementation notes
    --------------------
    * true_states is reshaped to (T, nx) if 1-D.  simulate() samples
      scalar Normal for SVSSM, yielding shape (T,); edh_particle.run_filter
      calls tf.expand_dims(true_states, axis=1) expecting (T, nx) input so
      that the result is (T, 1, nx), broadcastable with (T, N, nx) particles.
    * raw (unpreprocessed) observations must be passed — edh_particle.run_filter
      calls self.preprocess_obs() internally; passing proc_obs would double-transform.
    * The returned dict uses 'ess_avg' and 'omat' (edh_particle.run_filter keys),
      not 'step_metrics' (base-class key).
    """
    filter_module.load_ssm(ssm_model)

    if len(true_states.shape) == 1:          # (T,) → (T, 1) for univariate models
        true_states = tf.reshape(true_states, [-1, 1])

    # edh_particle.run_filter preprocesses obs internally; pass raw observations
    raw = filter_module.run_filter(observations, true_states)

    return {
        'rmse':    float(raw['rmse']),
        'ess':     float(raw['ess_avg']),    # edh_particle returns 'ess_avg'
        'runtime': float(raw['time']),
        'memory':  raw['mem'] / (1024 ** 2),
        'omat':    float(raw['omat']),       # edh_particle computes OMAT internally
    }


# ======================================================================
# SSM CONFIGURATIONS
# ======================================================================

# ── SVSSM ──────────────────────────────────────────────────────────────
SV_TRUE_THETA  = tf.constant([0.91, 1.00, 0.50], dtype=DTYPE)
SV_INIT_THETA  = tf.constant([0.70, 0.70, 0.70], dtype=DTYPE)
SV_THETA_DIM   = 3
SV_PARAM_NAMES = ["alpha", "sigma", "beta"]


def sv_builder(theta: tf.Tensor) -> StochasticVolatilityModel:
    return StochasticVolatilityModel(
        alpha=float(theta[0]), sigma=float(theta[1]), beta=float(theta[2]),
        static_diff=True,
    )


def sv_prior_log_prob(theta: tf.Tensor) -> tf.Tensor:
    a, s, b = theta[0], theta[1], theta[2]
    if a <= 0.0 or a >= 1.0 or s <= 0.0 or b <= 0.0:
        return tf.constant(-np.inf, dtype=DTYPE)
    return (
        tfd.Normal(loc=0.9, scale=0.1).log_prob(a)
        + tfd.Gamma(concentration=2.0, rate=2.0).log_prob(s)
        + tfd.Gamma(concentration=2.0, rate=2.0).log_prob(b)
    )


def sv_proposal(theta: tf.Tensor) -> tfd.Distribution:     #type: ignore
    return tfd.MultivariateNormalDiag(
        loc=theta,
        scale_diag=tf.constant([0.02, 0.02, 0.02], dtype=DTYPE),
    )


# ── MSVSSM-3 ──────────────────────────────────────────────────────────
MSV_P          = 3
MSV_TRUE_THETA = tf.constant(
    [0.90, 0.90, 0.90,    # phi
     0.50, 0.50, 0.50,    # sigma_eta diagonal
     0.50, 0.50, 0.50],   # beta
    dtype=DTYPE,
)
MSV_INIT_THETA = tf.constant(
    [0.70, 0.70, 0.70,
     0.70, 0.70, 0.70,
     0.70, 0.70, 0.70],
    dtype=DTYPE,
)
MSV_THETA_DIM   = 9
MSV_PARAM_NAMES = (
    [f"phi_{i}"   for i in range(MSV_P)]
    + [f"s_eta_{i}" for i in range(MSV_P)]
    + [f"beta_{i}"  for i in range(MSV_P)]
)


def msv_builder(theta: tf.Tensor) -> MultivariateStochasticVolatilityModel:
    phi       = theta[0:3].numpy().astype(np.float32)
    s_eta     = theta[3:6].numpy().astype(np.float32)
    beta      = theta[6:9].numpy().astype(np.float32)
    sigma_eta = np.diag(s_eta)                          # diagonal covariance
    sigma_eps = np.eye(MSV_P, dtype=np.float32)
    return MultivariateStochasticVolatilityModel(
        p=MSV_P, phi=phi, sigma_eta=sigma_eta,
        sigma_eps=sigma_eps, beta=beta, static_diff=True,
    )


def msv_prior_log_prob(theta: tf.Tensor) -> tf.Tensor:
    phi, s_eta, beta = theta[0:3], theta[3:6], theta[6:9]
    if tf.reduce_any(phi <= 0.0) or tf.reduce_any(phi >= 1.0):
        return tf.constant(-np.inf, dtype=DTYPE)
    if tf.reduce_any(s_eta <= 0.0) or tf.reduce_any(beta <= 0.0):
        return tf.constant(-np.inf, dtype=DTYPE)
    return (
        tf.reduce_sum(tfd.Normal(loc=0.9, scale=0.05).log_prob(phi))
        + tf.reduce_sum(tfd.Gamma(concentration=2.0, rate=4.0).log_prob(s_eta))
        + tf.reduce_sum(tfd.Gamma(concentration=2.0, rate=4.0).log_prob(beta))
    )


def msv_proposal(theta: tf.Tensor) -> tfd.Distribution:  #type: ignore
    return tfd.MultivariateNormalDiag(
        loc=theta,
        scale_diag=tf.constant([0.015] * MSV_THETA_DIM, dtype=DTYPE),
    )


# ======================================================================
# FILTER FACTORY
# ======================================================================

def _make_filter(
    theta_dim:     int,
    num_particles: int   = 100,
    num_steps:     int   = 20,
    label:         str   = "DeepONet",
) -> DeepONetSinkhornLEDHFilter:
    return DeepONetSinkhornLEDHFilter(
        num_particles=num_particles,
        num_steps=num_steps,
        theta_dim=theta_dim,
        label=label,
    )


# ======================================================================
# FULL MODEL EXPERIMENT
# ======================================================================

def run_model_experiment(
    model_name:     str,
    ssm_builder:    Callable,
    prior_log_prob: Callable,
    proposal_fn:    Callable,
    true_theta:     tf.Tensor,
    init_theta:     tf.Tensor,
    param_names:    List[str],
    theta_dim:      int,
    T:              int,
    # Filter
    num_particles:  int   = 100,
    num_steps:      int   = 20,
    pretrain_steps: int   = 1000,
    # MCMC
    num_iterations: int   = 300,
    burn_in:        int   = 100,
    pmmh_step:      float = 0.02,
    hmc_step:       float = 0.01,
    hmc_leapfrog:   int   = 10,
    nuts_step:      float = 0.01,
    nuts_max_depth: int   = 6,
) -> Dict[str, Any]:
    """
    End-to-end experiment for one SSM:

    1. Simulate  T observations from the true SSM.
    2. Pretrain  the DeepONetSinkhornLEDHFilter.
    3. Filter    baseline (RMSE / ESS / runtime / memory / OMAT).
    4. PMMH      chain — gradient-free stochastic likelihood.
    5. HMC       chain — gradient-based, fixed trajectory length.
    6. HMC-NUTS  chain — gradient-based, adaptive trajectory.

    Each MCMC method gets a fresh filter instance with the pretrained
    neural_ot_net weights copied from the reference filter.

    Returns a results dict with 'filter_metrics' and 'mcmc_results'
    keyed by 'pmmh', 'hmc', 'nuts'.
    """
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}  |  T={T}  |  theta_dim={theta_dim}")
    print(f"{'='*70}")

    # ── 1. Simulate ────────────────────────────────────────────────────
    print(f"\n[1/6] Simulating {T} time steps ...")
    true_ssm    = ssm_builder(true_theta)
    true_states, observations = true_ssm.simulate(T)
    preprocess  = true_ssm.filter_components().get("preprocess_obs", lambda y: y)
    proc_obs    = preprocess(observations)

    # ── 2. Build & Pretrain ────────────────────────────────────────────
    print(f"\n[2/6] Building & pretraining filter ({pretrain_steps} steps) ...")
    ref_filt = _make_filter(theta_dim, num_particles, num_steps,
                             label=f"DeepONet-ref-{model_name}")
    ref_filt.load_ssm(true_ssm)
    ref_filt.set_theta(true_theta)
    ref_filt.pretrain(steps=pretrain_steps)

    # ── 3. Filter baseline ─────────────────────────────────────────────
    print(f"\n[3/6] Filter baseline ...")
    # Pass raw observations — run_filter_metrics / edh_particle.run_filter preprocesses internally
    filter_results = run_filter_metrics(ref_filt, true_ssm, observations, true_states)
    print(
        f"  RMSE={filter_results['rmse']:.4f}  "
        f"ESS={filter_results['ess']:.1f}  "
        f"time={filter_results['runtime']:.2f}s  "
        f"mem={filter_results['memory']:.1f}MB  "
        f"OMAT={filter_results['omat']:.4f}"
    )

    mcmc_results: Dict[str, Any] = {}

    # ── 4. PMMH ────────────────────────────────────────────────────────
    print(f"\n[4/6] PMMH  ({num_iterations} iters, burn_in={burn_in}) ...")
    pmmh_filt = _make_filter(theta_dim, num_particles, num_steps,
                              label=f"DeepONet-PMMH-{model_name}")
    pmmh_filt.load_ssm(true_ssm)
    _copy_weights(ref_filt, pmmh_filt)

    pmmh_sampler = DeepONetPMMH(
        ssm_builder=ssm_builder,
        filter_module=pmmh_filt,
        prior_log_prob_fn=prior_log_prob,
        proposal_dist_fn=proposal_fn,
    )
    res = pmmh_sampler.run_chain(
        observations=proc_obs,
        init_theta=init_theta,
        num_iterations=num_iterations,
        burn_in=burn_in,
    )
    samples_np       = res['samples'].numpy()
    ess_c, rhat_c    = compute_mcmc_diagnostics(samples_np)
    post_mean        = np.mean(samples_np, axis=0)
    rmse_post        = float(np.sqrt(np.mean((post_mean - true_theta.numpy()) ** 2)))
    mcmc_results['pmmh'] = {**res, 'samples_np': samples_np,
                             'ess_chain': ess_c, 'r_hat': rhat_c,
                             'post_mean': post_mean, 'rmse_post': rmse_post,
                             'method': 'PMMH'}

    # ── 5. HMC ─────────────────────────────────────────────────────────
    print(f"\n[5a/6] HMC  ({num_iterations} iters, burn_in={burn_in}) ...")
    hmc_filt = _make_filter(theta_dim, num_particles, num_steps,
                             label=f"DeepONet-HMC-{model_name}")
    hmc_filt.load_ssm(true_ssm)
    _copy_weights(ref_filt, hmc_filt)

    hmc_sampler = DeepONetHMC(
        ssm_builder=ssm_builder,
        filter_module=hmc_filt,
        prior_log_prob_fn=prior_log_prob,
    )
    res = hmc_sampler.run_chain(
        observations=proc_obs,
        init_theta=init_theta,
        num_iterations=num_iterations,
        burn_in=burn_in,
        step_size=hmc_step,
        num_leapfrog_steps=hmc_leapfrog,
    )
    samples_np       = res['samples'].numpy()
    ess_c, rhat_c    = compute_mcmc_diagnostics(samples_np)
    post_mean        = np.mean(samples_np, axis=0)
    rmse_post        = float(np.sqrt(np.mean((post_mean - true_theta.numpy()) ** 2)))
    mcmc_results['hmc'] = {**res, 'samples_np': samples_np,
                            'ess_chain': ess_c, 'r_hat': rhat_c,
                            'post_mean': post_mean, 'rmse_post': rmse_post,
                            'method': 'HMC'}

    # ── 6. HMC-NUTS ────────────────────────────────────────────────────
    print(f"\n[5b/6] HMC-NUTS  ({num_iterations} iters, burn_in={burn_in}) ...")
    nuts_filt = _make_filter(theta_dim, num_particles, num_steps,
                              label=f"DeepONet-NUTS-{model_name}")
    nuts_filt.load_ssm(true_ssm)
    _copy_weights(ref_filt, nuts_filt)

    nuts_sampler = DeepONetNUTS(
        ssm_builder=ssm_builder,
        filter_module=nuts_filt,
        prior_log_prob_fn=prior_log_prob,
        target_accept_rate=0.65,
        max_tree_depth=nuts_max_depth,
    )
    res = nuts_sampler.run_chain(
        observations=proc_obs,
        init_theta=init_theta,
        num_iterations=num_iterations,
        burn_in=burn_in,
        step_size=nuts_step,
    )
    samples_np       = res['samples'].numpy()
    ess_c, rhat_c    = compute_mcmc_diagnostics(samples_np)
    post_mean        = np.mean(samples_np, axis=0)
    rmse_post        = float(np.sqrt(np.mean((post_mean - true_theta.numpy()) ** 2)))
    mcmc_results['nuts'] = {**res, 'samples_np': samples_np,
                             'ess_chain': ess_c, 'r_hat': rhat_c,
                             'post_mean': post_mean, 'rmse_post': rmse_post,
                             'method': 'HMC-NUTS'}

    print(f"\n[6/6] Done — {model_name}")
    return {
        'model_name':     model_name,
        'true_theta':     true_theta.numpy(),
        'param_names':    param_names,
        'filter_metrics': filter_results,
        'mcmc_results':   mcmc_results,
        'true_states':    true_states,
        'proc_obs':       proc_obs,
    }


# ======================================================================
# PRINTING
# ======================================================================

def print_filter_table(results: Dict) -> None:
    fm = results['filter_metrics']
    print(f"\n{'─'*58}")
    print(f"  {results['model_name']} — Filter Baseline (DeepONet-Sinkhorn-LEDH)")
    print(f"{'─'*58}")
    print(f"  {'Metric':<14}  {'Value'}")
    print(f"  {'─'*30}")
    print(f"  {'RMSE':<14}  {fm['rmse']:.5f}")
    print(f"  {'Avg ESS':<14}  {fm['ess']:.2f}")
    print(f"  {'Runtime (s)':<14}  {fm['runtime']:.3f}")
    print(f"  {'Peak Mem (MB)':<14}  {fm['memory']:.2f}")
    print(f"  {'OMAT':<14}  {fm['omat']:.5f}")


def print_inference_table(results: Dict) -> None:
    true_theta  = results['true_theta']
    param_names = results['param_names']
    mcmc        = results['mcmc_results']
    methods     = ['pmmh', 'hmc', 'nuts']
    labels      = ['PMMH', 'HMC', 'HMC-NUTS']

    print(f"\n{'─'*72}")
    print(f"  {results['model_name']} — MCMC Estimation (DeepONet-Sinkhorn-LEDH)")
    print(f"{'─'*72}")
    print(f"  {'Method':<12} {'Time(s)':<10} {'Acc%':<8} "
          f"{'min_ESS':<10} {'max_R^':<9} {'theta_RMSE'}")
    print(f"  {'─'*65}")
    for key, label in zip(methods, labels):
        r    = mcmc[key]
        acc  = r['acceptance_rate']
        acc_s = f"{acc:.1%}" if acc is not None else "N/A (NUTS)"
        print(f"  {label:<12} {r['time']:<10.2f} {acc_s:<8} "
              f"{np.min(r['ess_chain']):<10.1f} "
              f"{np.max(r['r_hat']):<9.3f} "
              f"{r['rmse_post']:.5f}")

    # Per-parameter posterior means
    print(f"\n  Posterior means  (True → PMMH | HMC | HMC-NUTS):")
    print(f"  {'─'*65}")
    for i, pname in enumerate(param_names):
        vals = "  |  ".join(f"{mcmc[k]['post_mean'][i]:.3f}" for k in methods)
        print(f"  {pname:<14}  True={true_theta[i]:.3f}   {vals}")

    # ESS
    print(f"\n  Chain ESS per parameter:")
    print(f"  {'─'*65}")
    header = f"  {'Param':<14}" + "".join(f"{lb:<12}" for lb in labels)
    print(header)
    for i, pname in enumerate(param_names):
        row = f"  {pname:<14}" + "".join(
            f"{mcmc[k]['ess_chain'][i]:<12.1f}" for k in methods
        )
        print(row)

    # R-hat
    print(f"\n  R-hat per parameter  (< 1.10 = converged):")
    print(f"  {'─'*65}")
    print(header)
    for i, pname in enumerate(param_names):
        row = f"  {pname:<14}" + "".join(
            f"{mcmc[k]['r_hat'][i]:<12.3f}" for k in methods
        )
        print(row)


# ======================================================================
# PLOTTING
# ======================================================================

_METHOD_KEYS   = ['pmmh', 'hmc', 'nuts']
_METHOD_LABELS = ['PMMH', 'HMC', 'HMC-NUTS']
_METHOD_COLORS = ['#e41a1c', '#377eb8', '#4daf4a']


def plot_trace_diagnostics(results: Dict) -> None:
    """
    Trace plots + ESS bar + R-hat bar for every parameter and every method.
    """
    model_name  = results['model_name']
    true_theta  = results['true_theta']
    param_names = results['param_names']
    mcmc        = results['mcmc_results']
    n_p         = len(param_names)

    n_rows = n_p + 2          # traces + ESS + R-hat
    n_cols = len(_METHOD_KEYS)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 2.8 * n_rows))
    fig.suptitle(
        f"DeepONet-Sinkhorn-LEDH  ·  {model_name}\n"
        f"SSM Parameter Estimation — Trace & Convergence Diagnostics",
        fontsize=13, fontweight='bold',
    )

    for col, (key, label, color) in enumerate(
        zip(_METHOD_KEYS, _METHOD_LABELS, _METHOD_COLORS)
    ):
        r       = mcmc[key]
        samp    = r['samples_np']
        ess     = r['ess_chain']
        rhat    = r['r_hat']

        # ── Trace plots ───────────────────────────────────────────────
        for row, pname in enumerate(param_names):
            ax = axs[row, col]
            ax.plot(samp[:, row], color=color, alpha=0.65, linewidth=0.75)
            ax.axhline(true_theta[row], color='red', ls='--',
                       lw=1.8, label='True θ')
            ax.axhline(np.mean(samp[:, row]), color='black', ls=':',
                       lw=1.3, label='Post. mean')
            ax.set_title(
                f"[{label}] {pname}   ESS={ess[row]:.0f}   R̂={rhat[row]:.3f}",
                fontsize=8,
            )
            if row == 0:
                ax.legend(fontsize=7, loc='upper right')
            ax.set_ylabel(pname, fontsize=8)
            ax.grid(True, alpha=0.25)

        # ── ESS bar ───────────────────────────────────────────────────
        ax_ess = axs[n_p, col]
        bars = ax_ess.bar(range(n_p), ess, color=color, alpha=0.8, edgecolor='white')
        ax_ess.set_xticks(range(n_p))
        ax_ess.set_xticklabels(param_names, rotation=45, fontsize=7)
        ax_ess.set_title(f"[{label}] Chain ESS", fontsize=9)
        ax_ess.axhline(100, color='gray', ls='--', lw=1.2, label='ESS=100')
        for bar, val in zip(bars, ess):
            ax_ess.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(ess) * 0.01,
                        f'{val:.0f}', ha='center', fontsize=6)
        ax_ess.legend(fontsize=7)
        ax_ess.grid(True, alpha=0.25, axis='y')

        # ── R-hat bar ─────────────────────────────────────────────────
        ax_rh = axs[n_p + 1, col]
        rhat_colors = [
            '#d73027' if v > 1.1 else '#4dac26' for v in rhat
        ]
        ax_rh.bar(range(n_p), rhat, color=rhat_colors, alpha=0.85, edgecolor='white')
        ax_rh.set_xticks(range(n_p))
        ax_rh.set_xticklabels(param_names, rotation=45, fontsize=7)
        ax_rh.axhline(1.1, color='red', ls='--', lw=1.5,
                      label='R̂=1.10 threshold')
        ax_rh.axhline(1.0, color='black', ls='-', lw=0.8, alpha=0.5)
        ax_rh.set_title(f"[{label}] R-hat", fontsize=9)
        ax_rh.legend(fontsize=7)
        ax_rh.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.show()


def plot_acceptance_diagnostics(results: Dict) -> None:
    """
    Rolling acceptance (PMMH, HMC) + step-size/tree-depth (NUTS).
    """
    model_name = results['model_name']
    mcmc       = results['mcmc_results']

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"DeepONet-Sinkhorn-LEDH  ·  {model_name}\n"
        f"MCMC Sampling Diagnostics",
        fontsize=12, fontweight='bold',
    )

    for ax, key, label, color in zip(
        axs, _METHOD_KEYS, _METHOD_LABELS, _METHOD_COLORS
    ):
        r = mcmc[key]
        if key in ('pmmh', 'hmc') and r.get('accepted') is not None:
            acc_trace = r['accepted'].numpy().astype(float)
            window    = min(50, len(acc_trace))
            rolling   = np.convolve(acc_trace, np.ones(window) / window, mode='valid')
            x_roll    = np.arange(window - 1, len(acc_trace))
            ax.plot(x_roll, rolling, color=color, alpha=0.85, linewidth=1.2,
                    label=f'Rolling acc (w={window})')
            ideal = 0.25 if key == 'pmmh' else 0.65
            ax.axhline(ideal, color='gray', ls='--', lw=1.5,
                       label=f'Ideal {ideal:.0%}')
            ax.set_title(f"[{label}] Rolling Acceptance Rate", fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Iteration")
            ax.legend(fontsize=8)
        elif key == 'nuts':
            ss = r.get('step_sizes', [])
            td = r.get('tree_depths', [])
            if ss:
                ax2 = ax.twinx()
                ax.plot(ss, color=color, alpha=0.8, lw=1.2, label='Step size ε')
                ax2.plot(td, color='darkorange', alpha=0.6, lw=0.9,
                         ls='--', label='Tree depth')
                ax2.set_ylabel("Tree depth", fontsize=8)
                ax2.legend(loc='upper right', fontsize=7)
                ax.axvline(results.get('burn_in', 100), color='gray',
                           ls='--', lw=1.2)
            ax.set_title(f"[{label}] NUTS — Step Size & Tree Depth", fontsize=9)
            ax.set_xlabel("Iteration")
            ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_filter_metrics_comparison(results_list: List[Dict]) -> None:
    """
    Side-by-side bar charts for filter baseline metrics across all models.
    """
    model_names = [r['model_name'] for r in results_list]
    metrics     = ['rmse', 'ess', 'runtime', 'memory', 'omat']
    m_labels    = ['RMSE', 'Avg ESS', 'Runtime (s)', 'Peak Mem (MB)', 'OMAT (W2)']
    colors      = plt.cm.tab10.colors

    fig, axs = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    fig.suptitle(
        "DeepONet-Sinkhorn-LEDH — Filter Baseline Metrics Comparison",
        fontsize=12, fontweight='bold',
    )
    for ax, metric, mlabel in zip(axs, metrics, m_labels):
        vals = [r['filter_metrics'][metric] for r in results_list]
        bars = ax.bar(model_names, vals,
                      color=[colors[i] for i in range(len(results_list))],
                      alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f'{val:.3f}', ha='center', fontsize=9)
        ax.set_title(mlabel, fontsize=10)
        ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.show()


def plot_posterior_summary(results: Dict) -> None:
    """
    Box-plots of posterior samples per parameter, across methods.
    """
    model_name  = results['model_name']
    true_theta  = results['true_theta']
    param_names = results['param_names']
    mcmc        = results['mcmc_results']
    n_p         = len(param_names)

    fig, axs = plt.subplots(1, n_p, figsize=(max(4 * n_p, 12), 5))
    if n_p == 1:
        axs = [axs]
    fig.suptitle(
        f"DeepONet-Sinkhorn-LEDH  ·  {model_name}\n"
        f"Posterior Distribution per Parameter",
        fontsize=12, fontweight='bold',
    )
    for ax, pname, true_val in zip(axs, param_names, true_theta):
        data = [mcmc[k]['samples_np'][:, param_names.index(pname)]
                for k in _METHOD_KEYS]
        bp = ax.boxplot(
            data, labels=_METHOD_LABELS, patch_artist=True, notch=False,
            medianprops=dict(color='black', lw=2),
        )
        for patch, color in zip(bp['boxes'], _METHOD_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(true_val, color='red', ls='--', lw=2, label='True θ')
        ax.set_title(pname, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    plt.show()


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":

    # ── SVSSM ─────────────────────────────────────────────────────────
    # Memory budget notes (SVSSM, nx=1):
    #   T=100  : halves gradient-tape memory vs T=200 for HMC/NUTS
    #   N=50   : OT cost matrix is N×N; 50² vs 100² = 4× smaller
    #   steps=10: LEDH flow steps compound gradient memory per filter call
    #   leapfrog=5, nuts_depth=4 (max 2⁴=16 evaluations): limits per-iter graph size
    results_sv = run_model_experiment(
        model_name      = "SVSSM",
        ssm_builder     = sv_builder,
        prior_log_prob  = sv_prior_log_prob,
        proposal_fn     = sv_proposal,
        true_theta      = SV_TRUE_THETA,
        init_theta      = SV_INIT_THETA,
        param_names     = SV_PARAM_NAMES,
        theta_dim       = SV_THETA_DIM,
        T               = 100,
        # Filter
        num_particles   = 50,
        num_steps       = 10,
        pretrain_steps  = 500,
        # MCMC
        num_iterations  = 200,
        burn_in         = 50,
        pmmh_step       = 0.02,
        hmc_step        = 0.01,
        hmc_leapfrog    = 5,
        nuts_step       = 0.01,
        nuts_max_depth  = 4,
    )
    print_filter_table(results_sv)
    print_inference_table(results_sv)
    plot_trace_diagnostics(results_sv)
    plot_acceptance_diagnostics(results_sv)
    plot_posterior_summary(results_sv)

    # ── MSVSSM-3 ──────────────────────────────────────────────────────
    # Memory budget notes (MSVSSM-3, nx=3, theta_dim=9):
    #   Gradient tapes are ~9× larger than SVSSM (theta_dim ratio);
    #   keep T, N, steps, and tree depth lower than SVSSM.
    results_msv = run_model_experiment(
        model_name      = "MSVSSM-3",
        ssm_builder     = msv_builder,
        prior_log_prob  = msv_prior_log_prob,
        proposal_fn     = msv_proposal,
        true_theta      = MSV_TRUE_THETA,
        init_theta      = MSV_INIT_THETA,
        param_names     = MSV_PARAM_NAMES,
        theta_dim       = MSV_THETA_DIM,
        T               = 80,
        # Filter
        num_particles   = 50,
        num_steps       = 8,
        pretrain_steps  = 600,
        # MCMC
        num_iterations  = 150,
        burn_in         = 50,
        pmmh_step       = 0.015,
        hmc_step        = 0.005,
        hmc_leapfrog    = 4,
        nuts_step       = 0.005,
        nuts_max_depth  = 3,
    )
    print_filter_table(results_msv)
    print_inference_table(results_msv)
    plot_trace_diagnostics(results_msv)
    plot_acceptance_diagnostics(results_msv)
    plot_posterior_summary(results_msv)

    # ── Cross-model filter comparison ─────────────────────────────────
    plot_filter_metrics_comparison([results_sv, results_msv])
