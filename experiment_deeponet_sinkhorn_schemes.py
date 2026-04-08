"""
Experiment: Theta Sampling Scheme Comparison for DeepONetSinkhornLEDHFilter

Models
------
  SVSSM   : Univariate Stochastic Volatility, theta = [alpha, sigma, beta] (dim=3)
  MSVSSM-3: Multivariate SV p=3,   theta = [phi×3, sigma_eta_diag×3, beta×3] (dim=9)

Sampling schemes compared
-------------------------
  1. Naive            – Gaussian samples drawn independently from the prior.
  2. Space-filling    – Latin Hypercube Sampling (LHS) for uniform coverage.
  3. Posterior-focused– Samples from a short pilot HMC chain using the fixed
                        DiffSinkhornLEDHFilter as likelihood evaluator; these
                        concentrate near the true posterior.

For each (model, scheme) combination the same DeepONetSinkhornLEDHFilter
is pretrained using only the theta samples from that scheme as conditioning
context, then evaluated on held-out test observations.

Metrics: pretraining f-potential MSE, filter RMSE, average ESS, runtime.
"""

import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Callable, List, Tuple, Dict, Any

# ── Try scipy for LHS; fall back to manual implementation ────────────
try:
    from scipy.stats.qmc import LatinHypercube
    _HAS_QMC = True
except ImportError:
    _HAS_QMC = False

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter
from FilterModules.NeuralFilter.deeponet_sinkhorn_ledh import DeepONetSinkhornLEDHFilter

from ParamEstimationPipeline.hmc_pipeline import HMC

DTYPE  = tf.float32
tfd    = tfp.distributions
np.random.seed(0)
tf.random.set_seed(0)
tf.get_logger().setLevel("ERROR")


# ======================================================================
# SSM CONFIGURATION
# ======================================================================

# ── SVSSM ─────────────────────────────────────────────────────────────
SV_TRUE_THETA  = np.array([0.91, 1.00, 0.50], dtype=np.float32)
SV_THETA_DIM   = 3
SV_PARAM_NAMES = ["alpha", "sigma", "beta"]

# Prior bounds (used for LHS grid and prior log-prob)
SV_LB = np.array([0.50, 0.10, 0.10], dtype=np.float32)
SV_UB = np.array([0.99, 3.00, 3.00], dtype=np.float32)

# Prior means/stds for Gaussian (naive) sampling
SV_PRIOR_MU  = np.array([0.90, 1.00, 1.00], dtype=np.float32)
SV_PRIOR_STD = np.array([0.10, 0.50, 0.50], dtype=np.float32)


def sv_builder(theta: tf.Tensor) -> StochasticVolatilityModel:
    return StochasticVolatilityModel(
        alpha=float(theta[0]), sigma=float(theta[1]), beta=float(theta[2]),
        static_diff=True,
    )


def sv_prior_log_prob(theta: tf.Tensor) -> tf.Tensor:
    a, s, b = theta[0], theta[1], theta[2]
    if a <= 0.0 or a >= 1.0 or s <= 0.0 or b <= 0.0:
        return tf.constant(-np.inf, dtype=DTYPE)
    lp  = tfd.Normal(loc=0.9, scale=0.1).log_prob(a)
    lp += tfd.Gamma(concentration=2.0, rate=2.0).log_prob(s)
    lp += tfd.Gamma(concentration=2.0, rate=2.0).log_prob(b)
    return lp


# ── MSVSSM-3 ──────────────────────────────────────────────────────────
# theta = [phi_0..2, sigma_eta_0..2, beta_0..2]  (flat, dim=9)
# sigma_eps is fixed to I_3 for identifiability.
MSV_P    = 3
MSV_TRUE_THETA = np.array(
    [0.90, 0.90, 0.90,   # phi
     0.50, 0.50, 0.50,   # sigma_eta (diagonal)
     0.50, 0.50, 0.50],  # beta
    dtype=np.float32,
)
MSV_THETA_DIM   = 9
MSV_PARAM_NAMES = (
    [f"phi_{i}" for i in range(MSV_P)]
    + [f"s_eta_{i}" for i in range(MSV_P)]
    + [f"beta_{i}" for i in range(MSV_P)]
)

MSV_LB = np.array([0.50]*3 + [0.10]*3 + [0.10]*3, dtype=np.float32)
MSV_UB = np.array([0.99]*3 + [2.00]*3 + [2.00]*3, dtype=np.float32)

MSV_PRIOR_MU  = np.array([0.90]*3 + [0.50]*3 + [0.50]*3, dtype=np.float32)
MSV_PRIOR_STD = np.array([0.05]*3 + [0.30]*3 + [0.30]*3, dtype=np.float32)


def msv_builder(theta: tf.Tensor) -> MultivariateStochasticVolatilityModel:
    phi       = theta[0:3].numpy().astype(np.float32)
    s_eta     = theta[3:6].numpy().astype(np.float32)
    beta      = theta[6:9].numpy().astype(np.float32)
    sigma_eta = np.diag(s_eta)
    sigma_eps = np.eye(MSV_P, dtype=np.float32)
    return MultivariateStochasticVolatilityModel(
        p=MSV_P, phi=phi, sigma_eta=sigma_eta,
        sigma_eps=sigma_eps, beta=beta, static_diff=True,
    )


def msv_prior_log_prob(theta: tf.Tensor) -> tf.Tensor:
    phi   = theta[0:3]
    s_eta = theta[3:6]
    beta  = theta[6:9]
    if tf.reduce_any(phi <= 0.0) or tf.reduce_any(phi >= 1.0):
        return tf.constant(-np.inf, dtype=DTYPE)
    if tf.reduce_any(s_eta <= 0.0) or tf.reduce_any(beta <= 0.0):
        return tf.constant(-np.inf, dtype=DTYPE)
    lp  = tf.reduce_sum(tfd.Normal(loc=0.9, scale=0.05).log_prob(phi))
    lp += tf.reduce_sum(tfd.Gamma(concentration=2.0, rate=4.0).log_prob(s_eta))
    lp += tf.reduce_sum(tfd.Gamma(concentration=2.0, rate=4.0).log_prob(beta))
    return lp


# ======================================================================
# THETA SAMPLING SCHEMES
# ======================================================================

def sample_naive(
    n_samples: int,
    prior_mu:  np.ndarray,
    prior_std: np.ndarray,
    lb:        np.ndarray,
    ub:        np.ndarray,
    seed:      int = 1,
) -> np.ndarray:
    """
    Naive scheme: i.i.d. Gaussian draws truncated to [lb, ub].

    This is the default approach — quick to implement but wasteful because
    most mass is drawn from the prior bulk, not the posterior region.
    """
    rng = np.random.default_rng(seed)
    samples = []
    while len(samples) < n_samples:
        s = rng.normal(loc=prior_mu, scale=prior_std).astype(np.float32)
        if np.all(s >= lb) and np.all(s <= ub):
            samples.append(s)
    return np.stack(samples)


def sample_lhs(
    n_samples: int,
    lb:        np.ndarray,
    ub:        np.ndarray,
    seed:      int = 2,
) -> np.ndarray:
    """
    Space-filling scheme: Latin Hypercube Sampling (LHS).

    Each dimension is divided into n_samples equal-probability intervals;
    exactly one sample is drawn from each interval per dimension, and the
    intervals are then randomly permuted across dimensions.  This guarantees
    even coverage of the parameter space without the clumping of Monte Carlo.

    Uses scipy.stats.qmc.LatinHypercube if available; falls back to a
    manual implementation otherwise.
    """
    d = len(lb)
    if _HAS_QMC:
        sampler = LatinHypercube(d=d, seed=seed)
        u = sampler.random(n=n_samples).astype(np.float32)   # (n, d) in [0,1]^d
    else:
        rng = np.random.default_rng(seed)
        u = np.zeros((n_samples, d), dtype=np.float32)
        for j in range(d):
            perm  = rng.permutation(n_samples)
            offsets = (np.arange(n_samples) + rng.uniform(size=n_samples)) / n_samples
            u[:, j] = offsets[perm]

    return (lb + u * (ub - lb)).astype(np.float32)


def sample_posterior_pilot(
    n_samples:     int,
    ssm_builder:   Callable,
    prior_log_prob: Callable,
    true_theta:    np.ndarray,
    observations:  tf.Tensor,
    pilot_N:       int  = 50,
    pilot_iters:   int  = 200,
    pilot_burn_in: int  = 80,
    step_size:     float = 0.005,
    leapfrog:      int  = 5,
    seed:          int  = 3,
) -> np.ndarray:
    """
    Posterior-focused scheme: samples from a short pilot HMC chain.

    A SoftResamplingParticleFilter (cheap, correct marginal log-likelihood
    at metrics index [1]) is used as the likelihood estimator.  The resulting
    posterior samples concentrate near the true θ — exactly the region where
    the neural transport map will be queried during the full HMC run.

    Args:
        n_samples    : Number of theta vectors to return.
        pilot_N      : Particles for the pilot filter (small — speed matters).
        pilot_iters  : Total HMC iterations (including burn-in).
        pilot_burn_in: Burn-in steps to discard.
    """
    tf.random.set_seed(seed)
    print("  [Pilot HMC] Running short chain for posterior-focused θ samples …")

    pilot_filter = SoftResamplingParticleFilter(
        num_particles=pilot_N, soft_alpha=0.5, label="Pilot_SoftRes"
    )
    init_ssm = ssm_builder(tf.constant(true_theta, dtype=DTYPE))
    pilot_filter.load_ssm(init_ssm)

    pilot_hmc = HMC(
        ssm_builder=ssm_builder,
        filter_module=pilot_filter,
        prior_log_prob_fn=prior_log_prob,
    )
    results = pilot_hmc.run_chain(
        observations=observations,
        init_theta=tf.constant(true_theta, dtype=DTYPE),
        num_iterations=pilot_iters,
        burn_in=pilot_burn_in,
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
    )
    pilot_samples = results["samples"].numpy()   # (post_burn_in, d)
    print(f"  [Pilot HMC] Done. {len(pilot_samples)} posterior samples collected.")

    # Subsample or tile to reach n_samples
    idx = np.random.choice(len(pilot_samples), size=n_samples, replace=True)
    return pilot_samples[idx].astype(np.float32)


# ======================================================================
# CUSTOM PRETRAIN WITH SCHEME-PROVIDED THETA SAMPLES
# ======================================================================

def pretrain_with_scheme(
    filt:          DeepONetSinkhornLEDHFilter,
    theta_samples: np.ndarray,
    ssm_builder:   Callable,
    steps:         int = 1500,
    n_per_step:    int = 4,
) -> List[float]:
    """
    Supervised pretraining of filt.neural_ot_net with theta-dependent targets.

    Correct training objective
    --------------------------
    The Sinkhorn algorithm is purely geometric: its output depends only on
    particle positions and their weights, NOT on theta directly.  However,
    the particle weights after a likelihood update DO depend on theta, because
    the observation likelihood p(y | x; theta) changes with theta.

    For the network's theta conditioning to be meaningful, the training targets
    must be theta-dependent.  We achieve this by:

      For each of n_per_step theta_j drawn from theta_samples:
        1. Build SSM(theta_j).
        2. Simulate N particles  x_i ~ f_{theta_j}(x_prev) + noise.
        3. Draw one observation  y   ~ h_{theta_j}(x_true) + obs_noise.
        4. Compute likelihood log-weights:
               log w_i = log p(y | x_i ; theta_j)
           These weights are a function of theta_j — different theta values
           assign different importance to the same particle positions.
        5. Run the fixed Sinkhorn algorithm on (x, log_w):
               f*_j = sinkhorn_f(x, log_w_j)
           This is the supervision target for this (x, y, theta_j) tuple.
        6. Minimise  MSE( network(x, y, theta_j),  f*_j ).

    Because f*_j varies with theta_j (via log_w_j), the network must learn
    to USE the theta context to predict the correct transport plan.  A network
    that ignores theta cannot minimise this loss across the full theta pool.

    This is the critical property that makes the three sampling schemes
    genuinely comparable:
      - Naive:      theta_j ~ prior Gaussian → network sees the prior bulk
      - LHS:        theta_j from LHS grid    → even coverage, no clumps
      - Posterior:  theta_j from pilot HMC   → concentrated near true θ*,
                    which is exactly where the network is queried at inference

    Args:
        filt          : Initialised DeepONetSinkhornLEDHFilter (SSM loaded).
        theta_samples : (n_pool, theta_dim) array from one of the three schemes.
        ssm_builder   : Callable theta → SSM, used to rebuild the SSM per theta.
        steps         : Number of Adam gradient steps.
        n_per_step    : Distinct theta contexts accumulated per gradient step.

    Returns
    -------
    loss_curve : per-step average f-potential MSE (length = steps).
    """
    N      = filt.num_particles
    log_b  = tf.fill([N], -tf.math.log(tf.cast(N, DTYPE)))
    n_pool = len(theta_samples)
    loss_curve: List[float] = []

    for step in range(steps):
        idx         = np.random.choice(n_pool, size=n_per_step, replace=True)
        accum_grads = None
        total_loss  = 0.0

        for j in range(n_per_step):
            th_j    = tf.constant(theta_samples[idx[j]], dtype=DTYPE)
            ssm_j   = ssm_builder(th_j)
            comps_j = ssm_j.filter_components()

            f_func_j = comps_j["f_func"]
            h_func_j = comps_j["h_func"]
            Q_j      = comps_j["Q"]
            R_j      = comps_j["R"]

            # ── 1. Simulate particles from SSM(theta_j) ───────────────
            x_prev = tf.random.normal((N, filt.nx), stddev=2.5, dtype=DTYPE)
            proc_noise_j = tf.linalg.cholesky(
                Q_j + tf.eye(filt.nx, dtype=DTYPE) * 1e-6
            )
            x_particles = f_func_j(x_prev) + tf.linalg.matvec(
                proc_noise_j,
                tf.random.normal((N, filt.nx), dtype=DTYPE),
                transpose_a=True,
            )
            x_particles = tf.clip_by_value(x_particles, -1e3, 1e3)

            # ── 2. Draw one observation from SSM(theta_j) ─────────────
            x_true_scalar = f_func_j(
                tf.random.normal((1, filt.nx), stddev=2.5, dtype=DTYPE)
            )
            obs_noise_j   = tf.linalg.cholesky(
                R_j + tf.eye(filt.ny, dtype=DTYPE) * 1e-6
            )
            y_obs = h_func_j(x_true_scalar) + tf.linalg.matvec(
                obs_noise_j,
                tf.random.normal((1, filt.ny), dtype=DTYPE),
                transpose_a=True,
            )                                                             # (1, ny)

            # ── 3. Theta-dependent likelihood log-weights ─────────────
            # log w_i = log p(y | x_i ; theta_j)
            #         = N(y ; h(x_i), R_j).log_prob
            pred_obs   = h_func_j(x_particles)                           # (N, ny)
            obs_expand = tf.reshape(y_obs, [1, filt.ny])
            diff_obs   = obs_expand - pred_obs                           # (N, ny)

            R_j_safe   = R_j + tf.eye(filt.ny, dtype=DTYPE) * 1e-6
            obs_dist_j = tfp.distributions.MultivariateNormalFullCovariance(
                loc=tf.zeros(filt.ny, dtype=DTYPE),
                covariance_matrix=R_j_safe,
            )
            log_w      = obs_dist_j.log_prob(diff_obs)                   # (N,)
            log_w      = tf.maximum(log_w, -1e5)
            log_w_norm = log_w - tf.reduce_logsumexp(log_w)              # (N,)

            # ── 4. Sinkhorn target f*_j = sinkhorn_f(x, log_w_j) ─────
            #    f*_j varies with theta_j because log_w_j does.
            diff_p = (
                x_particles[:, tf.newaxis, :]
                - x_particles[tf.newaxis, :, :]
            )                                                             # (N,N,nx)
            C      = tf.reduce_sum(diff_p ** 2, axis=-1)                 # (N, N)

            f_target, _ = filt.sinkhorn_potentials(log_w_norm, log_b, C)
            f_target     = tf.stop_gradient(f_target)                    # (N,)

            # ── 5. Network prediction and MSE loss ────────────────────
            y_ctx = tf.reshape(y_obs, [filt.ny])

            with tf.GradientTape() as tape:
                f_pred = filt.neural_ot_net(x_particles, y_ctx, th_j)
                loss   = tf.reduce_mean(tf.square(f_pred - f_target))

            grads = tape.gradient(loss, filt.neural_ot_net.trainable_variables)

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = [
                    (g1 + g2) if (g1 is not None and g2 is not None)
                    else (g1 if g1 is not None else g2)
                    for g1, g2 in zip(accum_grads, grads)
                ]
            total_loss += float(loss)

        if accum_grads is not None:
            avg_grads = [
                g / n_per_step if g is not None else g
                for g in accum_grads
            ]
            filt.optimizer.apply_gradients(
                zip(avg_grads, filt.neural_ot_net.trainable_variables)
            )

        avg_loss = total_loss / n_per_step
        loss_curve.append(avg_loss)

        if step % 300 == 0 or step == steps - 1:
            print(f"    step {step:>5}: f-MSE = {avg_loss:.5f}")

    return loss_curve


# ======================================================================
# HMC WRAPPER THAT MAINTAINS THETA CONDITIONING
# ======================================================================

class NeuralHMC(HMC):
    """
    HMC subclass that calls filter.set_theta(theta) after each load_ssm.

    The base HMC calls load_ssm() at every leapfrog step.  Since
    DeepONetSinkhornLEDHFilter.load_ssm() preserves the network after
    the first call (only updating SSM functions), load_ssm() is cheap.
    set_theta() is then called to restore the theta conditioning variable.
    """

    def _compute_log_prob_and_grad(
        self,
        theta: tf.Tensor,
        observations: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            prior_lp = self.prior_log_prob_fn(theta)

            if not tf.math.is_finite(prior_lp):
                return tf.constant(-np.inf, dtype=DTYPE), tf.zeros_like(theta)

            ssm_model = self.ssm_builder(theta)
            self.filter_module.load_ssm(ssm_model)        # cheap after first call
            self.filter_module.set_theta(theta)            # restore theta variable
            initial_state = self.filter_module.initialize_state()
            log_likelihood = self.filter_module._compiled_marginal_log_likelihood(
                observations, initial_state
            )
            total_log_prob = log_likelihood + prior_lp

        grad = tape.gradient(total_log_prob, theta)
        if grad is None or tf.reduce_any(tf.math.is_nan(grad)):
            grad = tf.zeros_like(theta)
        return total_log_prob, grad


# ======================================================================
# MAIN EXPERIMENT
# ======================================================================

def run_scheme_experiment(
    model_name:    str,
    true_theta:    np.ndarray,
    theta_dim:     int,
    ssm_builder:   Callable,
    prior_log_prob: Callable,
    param_names:   List[str],
    lb:            np.ndarray,
    ub:            np.ndarray,
    prior_mu:      np.ndarray,
    prior_std:     np.ndarray,
    T:             int  = 500,
    N:             int  = 80,
    num_basis:     int  = 16,
    embed_dim:     int  = 32,
    pretrain_steps: int = 1500,
    n_theta_pool:  int  = 256,
    pilot_iters:   int  = 200,
    pilot_burn_in: int  = 80,
) -> Dict[str, Any]:
    """
    Run the full comparison for one model.

    Returns a results dict containing per-scheme pretrain loss, filter
    metrics, timing, and the raw estimates for visualization.
    """
    print(f"\n{'='*70}")
    print(f"  MODEL: {model_name}  |  theta_dim={theta_dim}  |  T={T}  |  N={N}")
    print(f"{'='*70}")

    # ── Generate synthetic data ───────────────────────────────────────
    true_ssm   = ssm_builder(tf.constant(true_theta, dtype=DTYPE))
    true_states, observations = true_ssm.simulate(T)
    preprocess = true_ssm.filter_components().get(
        "preprocess_obs", lambda y: y
    )
    obs_processed = preprocess(observations)
    obs_tf        = tf.constant(obs_processed, dtype=DTYPE)

    states_eval = (
        tf.expand_dims(tf.constant(true_states, dtype=DTYPE), -1)
        if theta_dim == SV_THETA_DIM and model_name == "SVSSM"
        else tf.constant(true_states, dtype=DTYPE)
    )

    # ── Build initial SSM for filter initialisation ───────────────────
    init_ssm = ssm_builder(tf.constant(true_theta, dtype=DTYPE))

    # ── Theta sampling pools (all n_theta_pool × theta_dim) ──────────
    print("\n[1/4] Generating theta sampling pools …")

    pool_naive = sample_naive(n_theta_pool, prior_mu, prior_std, lb, ub)
    print(f"  Naive pool:     {pool_naive.shape}  "
          f"mean={pool_naive.mean(axis=0).round(3)}")

    pool_lhs   = sample_lhs(n_theta_pool, lb, ub)
    print(f"  LHS pool:       {pool_lhs.shape}  "
          f"mean={pool_lhs.mean(axis=0).round(3)}")

    pool_post  = sample_posterior_pilot(
        n_theta_pool, ssm_builder, prior_log_prob,
        true_theta, obs_tf,
        pilot_iters=pilot_iters, pilot_burn_in=pilot_burn_in,
    )
    print(f"  Posterior pool: {pool_post.shape}  "
          f"mean={pool_post.mean(axis=0).round(3)}")

    schemes = {
        "Naive (Gaussian)":       pool_naive,
        "Space-filling (LHS)":    pool_lhs,
        "Posterior-focused (HMC)": pool_post,
    }

    results: Dict[str, Any] = {
        "model": model_name, "theta_dim": theta_dim,
        "true_theta": true_theta, "true_states": true_states,
        "observations": np.array(observations),
        "pools": {k: v for k, v in schemes.items()},
        "scheme_results": {},
    }

    # ── Baseline: DiffSinkhornLEDH (fixed Sinkhorn, no network) ──────
    print("\n[2/4] Running baseline filter (fixed Sinkhorn LEDH) …")
    from FilterModules.DifferentiableFilters.diff_ledh import DiffSinkhornLEDHFilter

    baseline = DiffSinkhornLEDHFilter(
        num_particles=N, num_steps=20, ot_epsilon=0.5, label="Baseline-LEDH"
    )
    baseline.load_ssm(init_ssm)
    t0        = time.time()
    bl_metrics = baseline.run_filter(obs_tf, states_eval)
    bl_time    = time.time() - t0
    bl_ess     = float(bl_metrics.get("ess_avg", 0.0))
    print(f"  Baseline RMSE={bl_metrics['rmse']:.4f}  "
          f"ESS={bl_ess:.1f}  time={bl_time:.2f}s")

    results["baseline"] = {
        "rmse": bl_metrics["rmse"], "ess": bl_ess,
        "time": bl_time, "estimates": bl_metrics["estimates"].numpy(),
    }

    # ── Per-scheme: pretrain + filter ─────────────────────────────────
    print("\n[3/4] Training and running DeepONetSinkhornLEDH per scheme …")

    for scheme_name, theta_pool in schemes.items():
        print(f"\n  ── Scheme: {scheme_name} ──")

        filt = DeepONetSinkhornLEDHFilter(
            num_particles=N,
            num_steps=20,
            ot_epsilon=0.5,
            num_basis=num_basis,
            embed_dim=embed_dim,
            theta_dim=theta_dim,
            lr=5e-4,
            label=f"DON-SINHOM [{scheme_name[:3]}]",
        )
        filt.load_ssm(init_ssm)
        filt.set_theta(tf.constant(true_theta, dtype=DTYPE))

        print(f"  Pretraining ({pretrain_steps} steps) …")
        t_pre  = time.time()
        losses = pretrain_with_scheme(filt, theta_pool, ssm_builder, steps=pretrain_steps)
        pretrain_time = time.time() - t_pre
        print(f"  Pretrain done in {pretrain_time:.1f}s  "
              f"final MSE={losses[-1]:.5f}")

        print("  Running filter …")
        t_filt  = time.time()
        metrics = filt.run_filter(obs_tf, states_eval)
        filt_time = time.time() - t_filt
        ess_avg = float(metrics.get("ess_avg", 0.0))
        print(f"  Filter RMSE={metrics['rmse']:.4f}  "
              f"ESS={ess_avg:.1f}  time={filt_time:.2f}s")

        results["scheme_results"][scheme_name] = {
            "rmse":          metrics["rmse"],
            "ess":           ess_avg,
            "filt_time":     filt_time,
            "pretrain_time": pretrain_time,
            "losses":        losses,
            "estimates":     metrics["estimates"].numpy(),
        }

    return results


# ======================================================================
# VISUALISATION
# ======================================================================

SCHEME_COLORS = {
    "Naive (Gaussian)":        "#e07b39",
    "Space-filling (LHS)":     "#4e9fc7",
    "Posterior-focused (HMC)": "#3aaa72",
}
SCHEME_STYLES = {
    "Naive (Gaussian)":        "--",
    "Space-filling (LHS)":     "-.",
    "Posterior-focused (HMC)": "-",
}


def visualise_results(all_results: List[Dict]) -> None:
    """
    Produce a 3-panel figure per model:
      Row 1: Theta pool scatter (first two dimensions)
      Row 2: Pretraining loss curves (log scale)
      Row 3: State tracking (first latent dimension)
    Plus a summary comparison table.
    """
    n_models = len(all_results)
    fig = plt.figure(figsize=(18, 6 * n_models))
    outer = gridspec.GridSpec(n_models, 1, figure=fig, hspace=0.55)

    for m_idx, res in enumerate(all_results):
        model = res["model"]
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[m_idx], wspace=0.35
        )

        # ── Panel 1: theta pool scatter ───────────────────────────────
        ax1 = fig.add_subplot(inner[0])
        pools = res["pools"]
        true_th = res["true_theta"]

        for sname, pool in pools.items():
            ax1.scatter(
                pool[:, 0], pool[:, 1],
                s=12, alpha=0.45, label=sname,
                color=SCHEME_COLORS[sname],
            )
        ax1.axvline(true_th[0], color="red", lw=1.5, ls="--", alpha=0.8)
        ax1.axhline(true_th[1], color="red", lw=1.5, ls="--", alpha=0.8,
                    label="True θ")
        dim0 = "alpha" if res["theta_dim"] == 3 else "phi_0"
        dim1 = "sigma" if res["theta_dim"] == 3 else "phi_1"
        ax1.set_xlabel(dim0)
        ax1.set_ylabel(dim1)
        ax1.set_title(f"{model}\nTheta Sampling Pools (first 2 dims)")
        ax1.legend(fontsize=7, loc="upper right")
        ax1.grid(True, alpha=0.25)

        # ── Panel 2: pretrain loss curves ─────────────────────────────
        ax2 = fig.add_subplot(inner[1])
        for sname, sr in res["scheme_results"].items():
            losses = np.array(sr["losses"])
            # Smooth with a running mean for readability
            window = max(1, len(losses) // 40)
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            ax2.semilogy(
                np.arange(len(smoothed)), smoothed,
                color=SCHEME_COLORS[sname],
                lw=1.8, ls=SCHEME_STYLES[sname], label=sname,
            )
        ax2.set_xlabel("Pretrain step")
        ax2.set_ylabel("f-potential MSE (log)")
        ax2.set_title(f"{model}\nPretraining Loss Curves")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.25, which="both")

        # ── Panel 3: state tracking ───────────────────────────────────
        ax3 = fig.add_subplot(inner[2])
        T_plot  = min(200, len(res["true_states"]))
        t_axis  = np.arange(T_plot)
        states  = res["true_states"][:T_plot]
        x_plot  = states[:, 0] if states.ndim > 1 else states

        ax3.plot(t_axis, x_plot, "k-", lw=2, label="True state", zorder=5)
        # Baseline
        bl_est = res["baseline"]["estimates"][:T_plot]
        b_plot = bl_est[:, 0] if bl_est.ndim > 1 else bl_est
        ax3.plot(t_axis, b_plot, "k--", lw=1.2, alpha=0.5,
                 label=f"Baseline (RMSE={res['baseline']['rmse']:.3f})")

        for sname, sr in res["scheme_results"].items():
            est    = sr["estimates"][:T_plot]
            e_plot = est[:, 0] if est.ndim > 1 else est
            ax3.plot(
                t_axis, e_plot,
                lw=1.4, ls=SCHEME_STYLES[sname],
                color=SCHEME_COLORS[sname],
                alpha=0.85,
                label=f"{sname[:10]} (RMSE={sr['rmse']:.3f})",
            )

        ax3.set_xlabel("Time step")
        ax3.set_ylabel("Latent state (dim 0)")
        ax3.set_title(f"{model}\nState Tracking (first 200 steps)")
        ax3.legend(fontsize=6.5, loc="upper right")
        ax3.grid(True, alpha=0.25)

    plt.suptitle(
        "DeepONetSinkhornLEDH: Theta Sampling Scheme Comparison",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig("scheme_comparison.png", dpi=130, bbox_inches="tight")
    plt.show()
    print("Figure saved → scheme_comparison.png")


def print_summary_table(all_results: List[Dict]) -> None:
    """Print a formatted comparison table across models and schemes."""
    col_w = 26
    cols  = ["Model", "Scheme", "RMSE", "ESS", "Filt(s)", "Pretrain(s)", "f-MSE (final)"]
    widths = [8, col_w, 8, 7, 8, 12, 14]
    sep    = "─" * sum(widths + [3 * len(widths)])

    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print("\n" + "=" * len(sep))
    print("  SUMMARY: DeepONetSinkhornLEDH — Scheme Comparison")
    print("=" * len(sep))
    print(header)
    print(sep)

    for res in all_results:
        model = res["model"]
        # Baseline row
        bl = res["baseline"]
        row = [model, "Baseline (fixed Sinkhorn)",
               f"{bl['rmse']:.4f}", f"{bl['ess']:.1f}",
               f"{bl['time']:.2f}", "—", "—"]
        print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))

        for sname, sr in res["scheme_results"].items():
            final_loss = sr["losses"][-1] if sr["losses"] else float("nan")
            row = [model, sname,
                   f"{sr['rmse']:.4f}", f"{sr['ess']:.1f}",
                   f"{sr['filt_time']:.2f}", f"{sr['pretrain_time']:.1f}",
                   f"{final_loss:.5f}"]
            print(" | ".join(v.ljust(w) for v, w in zip(row, widths)))
        print(sep)


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    T              = 400    # time-series length
    N_PARTICLES    = 80     # particles per filter
    PRETRAIN_STEPS = 1500   # gradient steps for pretraining
    N_THETA_POOL   = 256    # theta samples per pool
    PILOT_ITERS    = 180    # HMC iterations for posterior-focused pilot
    PILOT_BURN_IN  = 60     # pilot burn-in

    all_results = []

    # ── SVSSM experiment ──────────────────────────────────────────────
    sv_res = run_scheme_experiment(
        model_name="SVSSM",
        true_theta=SV_TRUE_THETA,
        theta_dim=SV_THETA_DIM,
        ssm_builder=sv_builder,
        prior_log_prob=sv_prior_log_prob,
        param_names=SV_PARAM_NAMES,
        lb=SV_LB, ub=SV_UB,
        prior_mu=SV_PRIOR_MU, prior_std=SV_PRIOR_STD,
        T=T,
        N=N_PARTICLES,
        num_basis=16, embed_dim=32,
        pretrain_steps=PRETRAIN_STEPS,
        n_theta_pool=N_THETA_POOL,
        pilot_iters=PILOT_ITERS,
        pilot_burn_in=PILOT_BURN_IN,
    )
    all_results.append(sv_res)

    # ── MSVSSM-3 experiment ───────────────────────────────────────────
    msv_res = run_scheme_experiment(
        model_name="MSVSSM-3",
        true_theta=MSV_TRUE_THETA,
        theta_dim=MSV_THETA_DIM,
        ssm_builder=msv_builder,
        prior_log_prob=msv_prior_log_prob,
        param_names=MSV_PARAM_NAMES,
        lb=MSV_LB, ub=MSV_UB,
        prior_mu=MSV_PRIOR_MU, prior_std=MSV_PRIOR_STD,
        T=T,
        N=N_PARTICLES,
        num_basis=16, embed_dim=32,
        pretrain_steps=PRETRAIN_STEPS,
        n_theta_pool=N_THETA_POOL,
        pilot_iters=PILOT_ITERS,
        pilot_burn_in=PILOT_BURN_IN,
    )
    all_results.append(msv_res)

    # ── Output ────────────────────────────────────────────────────────
    print_summary_table(all_results)
    visualise_results(all_results)
