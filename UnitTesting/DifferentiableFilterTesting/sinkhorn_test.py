import pytest
import numpy as np
import tensorflow as tf

from FilterModules.DifferentiableFilters.sinkhorn_ot import SinkhornParticleFilter

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

# tf.config.set_visible_devices([], 'GPU')
SEEDS = [42] #, 123, 99, 90210, 6718]
DTYPE = tf.float32


class TestSinkhornParticleFilter:
    @pytest.fixture
    def lg_ssm(self):
        """Standard 1D Linear Gaussian Model."""
        A, B, C, D, Sigma_init = [[0.9]], [[0.1]], [[1.0]], [[0.5]], [[1.0]]
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def bad_lg_ssm(self):
        """Ill-conditioned LG Model: High condition number, big observation noise."""
        A = [[0.99, 0.0], [0.0, 0.01]]
        B = [[1.0, 0.0], [0.0, 1.0]]
        C = [[1.0, 0.5]]
        D = [[100.0]]                                            # big observation noise
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


    def test_mathematical_correctness_mass_collapse(self, lg_ssm):
        """
        Mathematical correctness of mass collapse:
        If one particle perfectly matches the observation and has 1.0 probability mass, 
        Optimal Transport moves all uniform target mass to that single point. 
        """
        N = 50
        pf = SinkhornParticleFilter(num_particles=N, epsilon=0.05, n_iter=50)                     # Low epsilon for sharper OT
        pf.load_ssm(lg_ssm)
        particles = tf.concat([[[10.0]], tf.zeros([N-1, 1], dtype=DTYPE)], axis=0)                # Set particle 0 to 10.0, all others to 0.0
        log_weights = tf.concat([[0.0], tf.fill([N-1], -1e9)], axis=0)                            # Particle 0 gets all the weight, others are severely suppressed
        new_particles, new_log_weights = pf.resample(particles, log_weights)
        np.testing.assert_allclose(new_particles.numpy(), 10.0, atol=1e-3)                         # All particles should have collapsed onto 10.0
        expected_uniform_log_w = -tf.math.log(float(N))                                            # Log weights should be perfectly uniform (-log N)
        np.testing.assert_allclose(new_log_weights.numpy(), expected_uniform_log_w, atol=1e-5)


    def test_mathematical_correctness_uniform_barycentric(self, lg_ssm):
        """
        Mathematical Correctness of uniformity:
        If particles are distributed normally and weights are uniform, 
        the Sinkhorn projection should maintain the spatial structure of the particles.
        """
        N = 100
        pf = SinkhornParticleFilter(num_particles=N, epsilon=1.0, n_iter=100)
        pf.load_ssm(lg_ssm)
        particles = tf.linspace(-5.0, 5.0, N)[:, tf.newaxis]                        # Use a symmetrically distributed particle set (mean exactly 0.0)
        original_mean = tf.reduce_mean(particles).numpy()
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        new_particles, _ = pf.resample(particles, log_weights)
        new_mean = tf.reduce_mean(new_particles).numpy()
        np.testing.assert_allclose(new_mean, original_mean, atol=1e-4)


    def test_conditioning_and_extreme_skew(self, lg_ssm):
        """
        Conditioning & Edge Case:
        Ensures the `log-domain` stabilized Sinkhorn iterations
        do not produce NaNs when the cost matrix is big or weights are skewed.
        """
        N = 50
        pf = SinkhornParticleFilter(num_particles=N, epsilon=1.0, n_iter=10)           # High epsilon and low iter to test numerical flow in bad scenarios
        pf.load_ssm(lg_ssm)
        particles = tf.concat([[[1e5]], tf.fill([N-1, 1], -1e5)], axis=0)              # Big spatial difference
        log_weights = tf.concat([[1e4], tf.fill([N-1], -1e4)], axis=0)                  # Big weights
        new_particles, new_log_weights = pf.resample(particles, log_weights)
        assert not tf.reduce_any(tf.math.is_nan(new_particles)), "NaNs detected in transported particles."
        assert not tf.reduce_any(tf.math.is_inf(new_particles)), "Infs detected in transported particles."
        assert not tf.reduce_any(tf.math.is_nan(new_log_weights)), "NaNs detected in new log weights."


    def test_gradient_norm(self, lg_ssm):
        """
        Grad Norm (Differentiability):
        Check that gradients can flow backwards through the Sinkhorn solver, 
        through the projection, and all the way back to the original 
        particles and the log_weights.
        """
        N = 20
        pf = SinkhornParticleFilter(num_particles=N, epsilon=0.5, n_iter=20)
        pf.load_ssm(lg_ssm)
        particles = tf.Variable(tf.random.normal([N, 1], dtype=DTYPE))
        log_weights = tf.Variable(tf.random.normal([N], dtype=DTYPE))
        
        with tf.GradientTape() as tape:
            new_particles, _ = pf.resample(particles, log_weights)
            loss = tf.reduce_sum(new_particles**2)
            
        grads = tape.gradient(loss, [particles, log_weights])
        assert grads[0] is not None, "Gradient w.r.t particles is None. OT flow broken."
        assert grads[1] is not None, "Gradient w.r.t log_weights is None. OT flow broken."
        
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
        Ensures the repeated execution of the XLA-compiled 
        OT solver does not degrade or overflow/underflow over time.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        ssm_model = request.getfixturevalue(ssm_fixture)
        pf = SinkhornParticleFilter(num_particles=100, epsilon=0.5, n_iter=30)
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