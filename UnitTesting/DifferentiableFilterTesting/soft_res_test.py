import pytest
import numpy as np
import tensorflow as tf

from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 99, 90210, 6718]
DTYPE = tf.float32


class TestSoftResamplingParticleFilter:
    @pytest.fixture
    def lg_ssm(self):
        """Standard 1D Linear Gaussian Model."""
        A, B, C, D, Sigma_init = [[0.9]], [[0.1]], [[1.0]], [[0.5]], [[1.0]]
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def bad_lg_ssm(self):
        """Ill-conditioned LG Model: High condition number, massive observation noise."""
        A = [[0.99, 0.0], [0.0, 0.01]]
        B = [[1.0, 0.0], [0.0, 1.0]]
        C = [[1.0, 0.5]]
        D = [[100.0]]                                     # Noisy
        Sigma_init = [[1000.0, 0.0], [0.0, 1000.0]]
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def lorenz_ssm(self):
        """Standard Lorenz 96 Model."""
        return Lorenz96Model(K=5, F=8.0, dt=0.05)
        
    @pytest.fixture
    def bad_lorenz_ssm(self):
        """Highly chaotic Lorenz 96 Model with larger state space."""
        return Lorenz96Model(K=20, F=15.0, dt=0.05)
        
    @pytest.fixture
    def sv_ssm(self):
        """Standard 1D Stochastic Volatility."""
        return StochasticVolatilityModel(alpha=0.91, sigma=0.5, beta=0.5)

    @pytest.fixture
    def bad_sv_ssm(self):
        """Badly conditioned SV: Near unit-root (alpha=0.999), massive state noise, tiny observation volatility."""
        return StochasticVolatilityModel(alpha=0.999, sigma=5.0, beta=0.01)

    @pytest.fixture
    def msv_ssm(self):
        """Standard Multivariate Stochastic Volatility Model."""
        phi = np.array([0.9, 0.85], dtype=np.float32)
        sigma_eta = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.float32)
        sigma_eps = np.array([[0.5, 0.05], [0.05, 0.5]], dtype=np.float32)
        beta = np.array([0.5, 0.5], dtype=np.float32)
        return MultivariateStochasticVolatilityModel(p=2, phi=phi, sigma_eta=sigma_eta, beta=beta, sigma_eps=sigma_eps)


    def test_mathematical_correctness_alpha_zero(self, lg_ssm):
        """Mathematical Correctness (Alpha = 0.0): Uniform proposal check."""
        N = 100
        pf = SoftResamplingParticleFilter(num_particles=N, soft_alpha=0.0)
        pf.load_ssm(lg_ssm)
        
        particles = tf.random.normal([N, 1], dtype=DTYPE)
        log_weights = tf.random.normal([N], dtype=DTYPE) 
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        W_t = tf.exp(log_w_norm)
        new_particles, new_log_weights = pf.resample(particles, log_weights)
        sum_new_weights = tf.reduce_logsumexp(new_log_weights).numpy()
        np.testing.assert_allclose(sum_new_weights, 0.0, atol=1e-5)


    def test_mathematical_correctness_alpha_one(self, lg_ssm):
        """Mathematical Correctness (Alpha = 1.0): Standard multinomial resampling check."""
        N = 100
        pf = SoftResamplingParticleFilter(num_particles=N, soft_alpha=1.0)
        pf.load_ssm(lg_ssm)
        particles = tf.random.normal([N, 1], dtype=DTYPE)
        log_weights = tf.range(N, dtype=DTYPE) 
        new_particles, new_log_weights = pf.resample(particles, log_weights)
        expected_uniform_log_w = -tf.math.log(float(N))
        np.testing.assert_allclose(new_log_weights.numpy(), expected_uniform_log_w, atol=1e-5)


    def test_conditioning_and_extreme_skew(self, lg_ssm):
        """Tests protections prevent NaNs/Infs when weight discrepancies are massive."""
        N = 50
        pf = SoftResamplingParticleFilter(num_particles=N, soft_alpha=0.5)
        pf.load_ssm(lg_ssm)
        particles = tf.zeros([N, 1], dtype=DTYPE)
        log_weights = tf.concat([[0.0], tf.fill([N-1], -1e9)], axis=0)
        new_particles, new_log_weights = pf.resample(particles, log_weights)
        assert not tf.reduce_any(tf.math.is_nan(new_log_weights)), "NaNs detected in log weights."
        assert not tf.reduce_any(tf.math.is_inf(new_log_weights)), "Infs detected in log weights."


    def test_gradient_norm(self, lg_ssm):
        """Grad Norm (Differentiability): Ensures gradients flow backwards through correction."""
        N = 10
        pf = SoftResamplingParticleFilter(num_particles=N, soft_alpha=0.5)
        pf.load_ssm(lg_ssm)
        particles = tf.random.normal([N, 1], dtype=DTYPE)
        log_weights = tf.Variable(tf.random.normal([N], dtype=DTYPE))
        
        with tf.GradientTape() as tape:
            _, new_log_weights = pf.resample(particles, log_weights)
            loss = tf.math.reduce_variance(new_log_weights)
            
        grad = tape.gradient(loss, log_weights)
        grad_norm = tf.linalg.norm(grad).numpy()
        assert grad is not None, "Gradient is None. Tape lost track of operations."
        assert not np.isnan(grad_norm), "Gradient norm is NaN."
        assert not np.isinf(grad_norm), "Gradient norm exploded to Inf."
        assert grad_norm > 0.0, "Gradient vanished completely (norm is 0)."


    def test_unconditional_resampling_trigger(self, lg_ssm):
        """Architecture check: Verifies resample triggers unconditionally even when ESS = N."""
        N = 100
        pf = SoftResamplingParticleFilter(num_particles=N, soft_alpha=0.5)
        pf.load_ssm(lg_ssm)
        obs = tf.constant([5.0], dtype=DTYPE)
        particles = tf.fill([N, 1], 5.0)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        state_pred = (particles, log_weights)
        (updated_particles, updated_log_weights), x_est, metrics = pf.update(state_pred, obs)

        ess = metrics[0].numpy()
        np.testing.assert_allclose(ess, float(N), atol=1e-4)
        
        expected_uniform_log_w = -np.log(float(N))
        np.testing.assert_allclose(updated_log_weights.numpy(), expected_uniform_log_w, atol=1e-5)


    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "bad_lg_ssm", "lorenz_ssm", "bad_lorenz_ssm", "sv_ssm", "bad_sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_stability_long_horizon(self, ssm_fixture, seed, request):
        """
        Stability: Runs the soft resampling filter across all standard and badly-conditioned 
        SSMs over a long sequence to ensure continuous soft-resampling does not degrade the XLA graph.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        ssm_model = request.getfixturevalue(ssm_fixture)
        pf = SoftResamplingParticleFilter(num_particles=150, soft_alpha=0.3)
        pf.load_ssm(ssm_model)
        
        T = 150
        if(ssm_fixture in ["sv_ssm", "bad_sv_ssm"]):
            observations = tf.math.abs(tf.random.normal((T,), dtype=DTYPE)) + 0.1
        elif(ssm_fixture in ["msv_ssm", "bad_msv_ssm"]):
            observations = tf.math.abs(tf.random.normal((T, pf.ny), dtype=DTYPE)) + 0.1
        else:
            observations = tf.random.normal((T, pf.ny), dtype=DTYPE)
        
        metrics = pf.run_filter(observations)
        estimates = metrics['estimates']
        assert estimates.shape == (T, pf.nx)
        assert not tf.reduce_any(tf.math.is_nan(estimates)), f"NaN discovered in state estimates for {ssm_fixture}."
        assert not tf.reduce_any(tf.math.is_inf(estimates)), f"Inf discovered in state estimates for {ssm_fixture}."