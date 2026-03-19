import pytest
import numpy as np
import tensorflow as tf

from FilterModules.DifferentiableFilters.opt_placement import OptimalPlacementParticleFilter

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42] #, 123, 99, 90210, 6718]
DTYPE = tf.float32


class TestOptimalPlacementParticleFilter:
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
        D = [[100.0]]  
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


    def test_mathematical_correctness_uniform_preservation(self, lg_ssm):
        """
        Mathematical Correctness (Uniform Preservation):
        If particles are already optimally placed and weights are completely uniform, the OPR 
        resampling should act as an identity transform and return the same particles.
        """
        N = 100
        pf = OptimalPlacementParticleFilter(num_particles=N)
        pf.load_ssm(lg_ssm)
        u_targets = (2.0 * np.arange(1, N + 1) - 1.0) / (2.0 * float(N))                     # Target uniform quantiles in the algorithm are (2i-1)/(2N)
        particles = tf.convert_to_tensor(u_targets[:, np.newaxis], dtype=DTYPE)              # Set particles to exactly these quantiles (monotonically increasing)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        
        new_particles, new_log_weights = pf.resample(particles, log_weights)
        np.testing.assert_allclose(new_particles.numpy(), particles.numpy(), atol=1e-5)
        
        expected_uniform_log_w = -np.log(float(N))
        np.testing.assert_allclose(new_log_weights.numpy(), expected_uniform_log_w, atol=1e-5)


    def test_mathematical_correctness_mass_collapse(self, lg_ssm):
            """
            Mathematical Correctness (OPR Smoothed Mass Collapse):
            Verify that the distribution aggressively shifts towards the dominant particle, 
            but preserves diversity (mean ~8.0, median ~10.0).
            """
            N = 50
            pf = OptimalPlacementParticleFilter(num_particles=N)
            pf.load_ssm(lg_ssm)
            particles = tf.concat([[[10.0]], tf.zeros([N-1, 1], dtype=DTYPE)], axis=0)               # Particle 0 is at 10.0, the rest are at 0.0
            log_weights = tf.concat([[0.0], tf.fill([N-1], -1e9)], axis=0)                           # Particle 0 gets practically all the weight
            new_particles, _ = pf.resample(particles, log_weights)
            median_val = np.median(new_particles.numpy())
            mean_val = np.mean(new_particles.numpy())
            assert 9.5 < median_val < 10.5, f"Expected median near 10.0, got {median_val}"
            assert 7.0 < mean_val < 9.0, f"Expected mean near 8.0, got {mean_val}"


    def test_conditioning_and_extreme_skew(self, lg_ssm):
        """
        Conditioning & Edge Case:
        Ensures the inverse_cdf_transform prevent NaNs 
        when taking the log of tail distributions.
        """
        N = 50
        pf = OptimalPlacementParticleFilter(num_particles=N)
        pf.load_ssm(lg_ssm)
        particles = tf.fill([N, 1], 5.0)                                  # All particles have the exact same value (Zero-width CDF segments)
        log_weights = tf.concat([[1e4], tf.fill([N-1], -1e4)], axis=0)    # Extreme weights with many zeros
        new_particles, new_log_weights = pf.resample(particles, log_weights)
        assert not tf.reduce_any(tf.math.is_nan(new_particles)), "NaNs detected in interpolated particles."
        assert not tf.reduce_any(tf.math.is_inf(new_particles)), "Infs detected in interpolated particles."
        assert not tf.reduce_any(tf.math.is_nan(new_log_weights)), "NaNs detected in new log weights."


    def test_gradient_norm(self, lg_ssm):
        """
        Grad Norm (Differentiability):
        Ensures that gradients can flow backwards.
        """
        N = 20
        pf = OptimalPlacementParticleFilter(num_particles=N)
        pf.load_ssm(lg_ssm)
        particles = tf.Variable(tf.random.normal([N, 1], dtype=DTYPE))
        log_weights = tf.Variable(tf.random.normal([N], dtype=DTYPE))
        
        with tf.GradientTape() as tape:
            new_particles, _ = pf.resample(particles, log_weights)
            loss = tf.reduce_sum(new_particles**2)
            
        grads = tape.gradient(loss, [particles, log_weights])
        assert grads[0] is not None, "Gradient w.r.t particles is None. Interpolation flow broken."
        assert grads[1] is not None, "Gradient w.r.t log_weights is None. CDF flow broken."
        
        grad_particles_norm = tf.linalg.norm(grads[0]).numpy()
        grad_weights_norm = tf.linalg.norm(grads[1]).numpy()
        assert not np.isnan(grad_particles_norm), "Gradient norm for particles is NaN."
        assert not np.isnan(grad_weights_norm), "Gradient norm for weights is NaN."
        assert grad_particles_norm > 0.0, "Gradient for particles vanished (is 0)."
        assert grad_weights_norm > 0.0, "Gradient for weights vanished (is 0)."


    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "bad_lg_ssm", "lorenz_ssm", "bad_lorenz_ssm", "sv_ssm", "bad_sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_stability_long_horizon(self, ssm_fixture, seed, request):
        """
        Stability: Runs the OPR filter across all standard and badly-conditioned SSMs.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        ssm_model = request.getfixturevalue(ssm_fixture)
        pf = OptimalPlacementParticleFilter(num_particles=150)
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