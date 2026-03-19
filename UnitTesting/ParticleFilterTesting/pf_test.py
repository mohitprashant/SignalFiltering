import pytest
import numpy as np
import tensorflow as tf

from FilterModules.ParticleFilters.particle import ParticleFilter

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 99, 90210, 6718]
DTYPE = tf.float32



class TestParticleFilter:
    @pytest.fixture
    def lg_ssm(self):
        """1D Linear Gaussian Model for simple verification."""
        A = [[0.9]]
        B = [[0.1]]
        C = [[1.0]]
        D = [[0.5]]
        Sigma_init = [[1.0]]
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def lorenz_ssm(self):
        """Lorenz 96 Model for non-linear stability testing."""
        return Lorenz96Model(K=5, F=8.0, dt=0.05)
        
    @pytest.fixture
    def sv_ssm(self):
        """Stochastic Volatility for testing preprocess_obs logic."""
        return StochasticVolatilityModel(alpha=0.91, sigma=0.5, beta=0.5)

    @pytest.fixture
    def msv_ssm(self):
        """Multivariate Stochastic Volatility Model Fixture."""
        phi = np.array([0.9, 0.85], dtype=np.float32)
        sigma_eta = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.float32)
        sigma_eps = np.array([[0.5, 0.05], [0.05, 0.5]], dtype=np.float32)
        beta = np.array([0.5, 0.5], dtype=np.float32)
        return MultivariateStochasticVolatilityModel(p=2, phi=phi, sigma_eta=sigma_eta, beta=beta, sigma_eps=sigma_eps)



    def test_initial_distribution(self, lg_ssm):
        """Verifies particles are initialized with uniform log weights summing to 1."""
        pf = ParticleFilter(num_particles=100)
        pf.load_ssm(lg_ssm)
        
        particles, log_weights = pf.initialize_state()
        
        assert particles.shape == (100, 1)
        assert log_weights.shape == (100,)
        expected_log_weight = -np.log(100.0)
        np.testing.assert_allclose(log_weights.numpy(), expected_log_weight, atol=1e-5)
        
        sum_weights = tf.reduce_sum(tf.exp(log_weights)).numpy()
        np.testing.assert_allclose(sum_weights, 1.0, atol=1e-6)



    def test_resampling_trigger_low_ess(self, lg_ssm):
        """
        Forces a state where ESS is virtually 1.0 (one particle dominates).
        Verifies that tf.cond triggers resampling and resets weights to uniform.
        """
        N = 100
        pf = ParticleFilter(num_particles=N, resample_threshold_ratio=0.5)
        pf.load_ssm(lg_ssm)
        particles = tf.zeros((N, 1), dtype=DTYPE)
        
        # One particle perfectly matches the observation, others are far off
        obs = tf.constant([10.0], dtype=DTYPE)
        particles = tf.tensor_scatter_nd_update(particles, [[0]], [[10.0]])
        
        # Give them uniform weights initially to pass to the update step
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        state_pred = (particles, log_weights)
        (updated_particles, updated_log_weights), x_est, metrics = pf.update(state_pred, obs)
        ess = metrics[0].numpy()
        
        # Only particle 0 is close to the obs. ESS should plummet below threshold (50)
        assert ess < 50.0 
        
        # Weights should be reset to uniform (-log(N))
        expected_reset_weight = -np.log(float(N))
        np.testing.assert_allclose(updated_log_weights.numpy(), expected_reset_weight, atol=1e-5)
        
        # Particle 0 (value 10.0) should have been heavily duplicated
        assert np.sum(updated_particles.numpy() == 10.0) > (N * 0.9)


    def test_resampling_skipped_high_ess(self, lg_ssm):
        """
        Forces a state where all particles match the observation equally.
        ESS will be N. Verifies that tf.cond skips resampling.
        """
        N = 100
        pf = ParticleFilter(num_particles=N, resample_threshold_ratio=0.5)
        pf.load_ssm(lg_ssm)
        
        # All particles are identical and perfectly match the observation
        obs = tf.constant([5.0], dtype=DTYPE)
        particles = tf.fill([N, 1], 5.0)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        
        state_pred = (particles, log_weights)
        (updated_particles, updated_log_weights), x_est, metrics = pf.update(state_pred, obs)
        
        ess = metrics[0].numpy()
        
        # ESS should be perfectly N (100) since all likelihoods are identical
        np.testing.assert_allclose(ess, float(N), atol=1e-4)
        
        # We verify resampling was skipped by checking that the particles are untouched
        np.testing.assert_array_equal(updated_particles.numpy(), particles.numpy())


    def test_edge_case_zero_resampling(self, lg_ssm):
        """
        If threshold_ratio is 0.0, the filter should NEVER resample.
        We verify weights just keep accumulating and diverging.
        """
        N = 50
        pf = ParticleFilter(num_particles=N, resample_threshold_ratio=0.0)
        pf.load_ssm(lg_ssm)
        
        particles, log_weights = pf.initialize_state()
        
        for _ in range(10):
            particles, log_weights = pf.predict((particles, log_weights))
            obs = tf.random.normal([1], dtype=DTYPE)
            (particles, log_weights), _, _ = pf.update((particles, log_weights), obs)
            
        # Weights should be highly skewed
        assert not np.allclose(log_weights.numpy(), log_weights.numpy()[0])


    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "lorenz_ssm", "sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_long_horizon_stability(self, ssm_fixture, seed, request):
        """
        Tests the PF against different models over a long horizon (T=100) to ensure
        the XLA compilation processes dimensions correctly without NaN/Inf explosions.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        ssm_model = request.getfixturevalue(ssm_fixture)
        pf = ParticleFilter(num_particles=200, resample_threshold_ratio=0.5)
        pf.load_ssm(ssm_model)
        
        T = 100
        
        if (ssm_fixture == "sv_ssm"):
            observations = tf.math.abs(tf.random.normal((T,), dtype=DTYPE)) + 0.1
        elif (ssm_fixture == "msv_ssm"):
            observations = tf.math.abs(tf.random.normal((T, pf.ny), dtype=DTYPE)) + 0.1
        else:
            observations = tf.random.normal((T, pf.ny), dtype=DTYPE)

        metrics = pf.run_filter(observations)
        estimates = metrics['estimates']
        ess_avg = metrics['ess_avg']
        
        assert not tf.reduce_any(tf.math.is_nan(estimates)), f"NaN discovered in Estimates for {ssm_fixture}."
        assert not tf.reduce_any(tf.math.is_inf(estimates)), f"Inf discovered in Estimates for {ssm_fixture}."
        assert 1.0 <= ess_avg <= 200.0, f"Average ESS out of logical bounds: {ess_avg}"