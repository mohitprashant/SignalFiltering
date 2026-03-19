import pytest
import numpy as np
import tensorflow as tf

from FilterModules.DifferentiableFilters.condnorm_flow import CNFParticleFilter 

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 90210]
DTYPE = tf.float32


class TestCNFParticleFilter:
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



    def test_dynamic_network_initialization(self, msv_ssm):
        """
        Architecture Check.
        """
        pf = CNFParticleFilter(num_particles=100)
        pf.load_ssm(msv_ssm)
        assert pf.shift_net is not None
        assert pf.scale_net is not None

        dummy_obs = tf.random.normal([1, pf.ny])
        shift = pf.shift_net(dummy_obs)
        scale = pf.scale_net(dummy_obs)
        assert shift.shape == (1, pf.nx)
        assert scale.shape == (1, pf.nx)


    def test_mathematical_correctness_identity_initialization(self, lg_ssm):
        """
        Mathematical Correctness (Identity Flow):
        Because the final layers of the Neural Nets use 'zeros' initializers, 
        the flow should perfectly act as an Identity Transformation at step 0.
        """
        pf = CNFParticleFilter(num_particles=100)
        pf.load_ssm(lg_ssm)
        dummy_obs = tf.random.normal([1, pf.ny])
        shift = pf.shift_net(dummy_obs)
        log_scale = pf.scale_net(dummy_obs)
        scale = tf.exp(log_scale)
        log_det_J = tf.reduce_sum(log_scale, axis=-1)
        np.testing.assert_allclose(shift.numpy(), np.zeros((1, pf.nx)), atol=1e-6)
        np.testing.assert_allclose(scale.numpy(), np.ones((1, pf.nx)), atol=1e-6)
        np.testing.assert_allclose(log_det_J.numpy(), np.zeros((1,)), atol=1e-6)


    def test_extreme_observation_edge_case(self, lg_ssm):
        """
        Edge Case:
        Ensures the `tanh` activation in the hidden layer protects the flow from 
        exploding or returning NaNs when exposed to anomalous observation spikes.
        """
        pf = CNFParticleFilter(num_particles=100)
        pf.load_ssm(lg_ssm)
        extreme_obs = tf.fill([1, pf.ny], 1e6)              # Give outlier observation
        shift = pf.shift_net(extreme_obs)
        log_scale = pf.scale_net(extreme_obs)
        assert not tf.reduce_any(tf.math.is_nan(shift))
        assert not tf.reduce_any(tf.math.is_inf(shift))
        assert not tf.reduce_any(tf.math.is_nan(log_scale))
        assert not tf.reduce_any(tf.math.is_inf(log_scale))


    def test_gradient_norm_flow_networks(self, lg_ssm):
        """
        Grad Norm (Differentiability):
        Verifies that gradients can flow from a downstream loss.
        """
        N = 50
        pf = CNFParticleFilter(num_particles=N, soft_alpha=0.5)
        pf.load_ssm(lg_ssm)
        particles = tf.random.normal([N, pf.nx], dtype=DTYPE)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        obs = tf.constant([2.5], dtype=DTYPE)
        state_pred = pf.predict((particles, log_weights))
        
        with tf.GradientTape() as tape:
            (new_particles, new_log_weights), x_est, _ = pf.update(state_pred, obs)
            loss = tf.reduce_sum(x_est**2)
            
        trainable_vars = pf.shift_net.trainable_variables + pf.scale_net.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        for grad, var in zip(grads, trainable_vars):
            assert grad is not None, f"Gradient for {var.name} is None. Flow broken."

            grad_norm = tf.linalg.norm(grad).numpy()
            assert not np.isnan(grad_norm), f"Gradient for {var.name} is NaN."
            assert not np.isinf(grad_norm), f"Gradient for {var.name} exploded."



    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "bad_lg_ssm", "lorenz_ssm", "bad_lorenz_ssm", "sv_ssm", "bad_sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_stability_long_horizon(self, ssm_fixture, seed, request):
        """
        Runs the CNF Particle Filter.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        ssm_model = request.getfixturevalue(ssm_fixture)
        pf = CNFParticleFilter(num_particles=150, soft_alpha=0.5)
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