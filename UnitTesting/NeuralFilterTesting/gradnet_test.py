import pytest
import numpy as np
import tensorflow as tf

from FilterModules.NeuralFilter.gradnet_filter import GradNetParticleFilter

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42] #, 123, 99, 90210, 6718]
DTYPE = tf.float32


class TestGradNetParticleFilter:
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
        """Badly conditioned SV: Near unit-root (alpha=0.999), massive state noise."""
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
        pf = GradNetParticleFilter(num_particles=100)
        pf.load_ssm(msv_ssm)
        assert pf.net is not None
        assert pf.net.nx == pf.nx
        
        dummy_x = tf.random.normal([100, pf.nx])
        dummy_obs = tf.random.normal([100, pf.ny])
        mapped_x = pf.net([dummy_x, dummy_obs])
        assert mapped_x.shape == (100, pf.nx)


    def test_jacobian_determinant_validity(self, lg_ssm):
        """
        Ensure the batch_jacobian and tf.linalg.det logic computes 
        the log determinant correctly without throwing dimension errors or NaNs.
        """
        N = 50
        pf = GradNetParticleFilter(num_particles=N)
        pf.load_ssm(lg_ssm)
        dummy_hat_particles = tf.random.normal([N, pf.nx])
        dummy_obs = tf.random.normal([1, pf.ny])
        particles_new, log_det = pf._apply_transport_map(dummy_hat_particles, dummy_obs)
        assert particles_new.shape == (N, pf.nx)
        assert log_det.shape == (N,)
        assert not tf.reduce_any(tf.math.is_nan(log_det)), "NaNs detected in Jacobian log det."
        assert not tf.reduce_any(tf.math.is_inf(log_det)), "Infs detected in Jacobian log det."


    def test_pretrain_execution(self, lg_ssm):
        """
        Pretraining Logic:
        Runs a very short supervised pretraining loop to ensure the loss computes 
        and weights update correctly based on simulated SSM data.
        """
        pf = GradNetParticleFilter(num_particles=50, lr=0.1)
        pf.load_ssm(lg_ssm)
        initial_weights = [tf.identity(w) for w in pf.net.trainable_variables]
        pf.pretrain(steps=5, batch_size=16)
        weights_changed = False

        for initial_w, trained_w in zip(initial_weights, pf.net.trainable_variables):
            if(not tf.reduce_all(tf.math.equal(initial_w, trained_w))):
                weights_changed = True
                break
        assert weights_changed, "Network weights did not change during pretraining."


    def test_extreme_observation_edge_case(self, lg_ssm):
        """
        Ensure the activations protect the network 
        from returning NaNs when exposed to anomalous observations.
        """
        pf = GradNetParticleFilter(num_particles=100)
        pf.load_ssm(lg_ssm)
        extreme_obs = tf.fill([100, pf.ny], 1e6)
        dummy_x = tf.random.normal([100, pf.nx])
        mapped_x = pf.net([dummy_x, extreme_obs])
        assert not tf.reduce_any(tf.math.is_nan(mapped_x))
        assert not tf.reduce_any(tf.math.is_inf(mapped_x))


    def test_gradient_norm_transport_map(self, lg_ssm):
        """
        Verifies gradients can flow from a downstream loss.
        """
        N = 50
        pf = GradNetParticleFilter(num_particles=N, soft_alpha=0.5)
        pf.load_ssm(lg_ssm)
        particles = tf.random.normal([N, pf.nx], dtype=DTYPE)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        obs = tf.constant([2.5], dtype=DTYPE)

        state_pred = pf.predict((particles, log_weights))

        with tf.GradientTape() as tape:
            (new_particles, new_log_weights), x_est, _ = pf.update(state_pred, obs)
            loss = tf.reduce_sum(x_est**2)
            
        trainable_vars = pf.net.trainable_variables
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
        Stability check.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        ssm_model = request.getfixturevalue(ssm_fixture)
        pf = GradNetParticleFilter(num_particles=50, num_modules=2, soft_alpha=0.5)
        pf.load_ssm(ssm_model)
        
        T = 50
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