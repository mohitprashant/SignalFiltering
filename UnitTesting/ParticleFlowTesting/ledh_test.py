import pytest
import numpy as np
import tensorflow as tf

from FilterModules.flow_base import ParticleFlow
from FilterModules.ParticleFlow.ledh_flow import LocalizedExactDaumHuangFilter

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

# tf.config.set_visible_devices([], 'GPU')
DTYPE = tf.float32
SEEDS = [42, 123, 99, 90210, 6718]


class TestParticleFlowLEDH:
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
        """1D Stochastic Volatility for testing preprocess_obs logic."""
        return StochasticVolatilityModel(alpha=0.91, sigma=0.5, beta=0.5)

    @pytest.fixture
    def msv_ssm(self):
        """Multivariate Stochastic Volatility Model Fixture."""
        phi = np.array([0.9, 0.85], dtype=np.float32)
        sigma_eta = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.float32)
        sigma_eps = np.array([[0.5, 0.05], [0.05, 0.5]], dtype=np.float32)
        beta = np.array([0.5, 0.5], dtype=np.float32)
        return MultivariateStochasticVolatilityModel(p=2, phi=phi, sigma_eta=sigma_eta, beta=beta, sigma_eps=sigma_eps)


    def test_particle_flow_math_ledh(self):
        """
        Tests the batch math of compute_Ab_and_cond in ParticleFlow.
        LEDH uses local linearization, meaning x_lin is a 2D tensor of shape (N, nx).
        """
        pf_base = ParticleFlow()
        nx = 2
        eye_nx = tf.eye(nx, dtype=DTYPE)
        P_HT = eye_nx
        H = eye_nx
        HPH = eye_nx
        R = eye_nx
        
        y_obs = tf.constant([1.0, 1.0], dtype=DTYPE)
        P_HT_R_inv_y = tf.constant([1.0, 1.0], dtype=DTYPE) # P H^T R^-1 y = I I I [1, 1]^T
        
        # LEDH Local Linearization: Pass N=2 particles as x_lin
        # Particle 0 is at [0, 0], Particle 1 is at [1, 1]
        x_lin = tf.constant([[0.0, 0.0], [1.0, 1.0]], dtype=DTYPE)
        lam = 1.0
        
        # Calculate outputs
        A, b, cond = pf_base.compute_Ab_and_cond(P_HT, H, HPH, R, P_HT_R_inv_y, x_lin, lam, eye_nx)
        
        # HAND CALC
        # S_lam = lam * HPH + R = 1.0 * I + I = 2I
        # S_lam_inv = 0.5 * I
        # A = -0.5 * P_HT * S_lam_inv * H = -0.5 * I * 0.5I * I = -0.25I
        expected_A = -0.25 * np.eye(nx)
        np.testing.assert_allclose(A.numpy(), expected_A, atol=1e-5)
        
        # b calculation for LEDH is batch processed per particle:
        # term1 = (I + lam * A) * P_HT_R_inv_y = (I - 0.25I) * [1, 1]^T = 0.75 * [1, 1]^T
        # term2 = A * x_lin
        # P0: -0.25I * [0, 0]^T = [0, 0]^T
        # P1: -0.25I * [1, 1]^T = [-0.25, -0.25]^T
        
        # b = (I + 2*lam*A) * (term1 + term2)
        # P0: 0.5I * ( [0.75, 0.75]^T + [0, 0]^T ) = [0.375, 0.375]^T
        # P1: 0.5I * ( [0.75, 0.75]^T + [-0.25, -0.25]^T ) = [0.25, 0.25]^T
        expected_b = np.array([[0.375, 0.375], [0.25, 0.25]])
        np.testing.assert_allclose(b.numpy(), expected_b, atol=1e-5)
        np.testing.assert_allclose(cond.numpy(), 1.0, atol=1e-5)


    def test_omat_computation(self):
        """Verifies the Weighted Euclidean Error (OMAT) computation."""
        ledh = LocalizedExactDaumHuangFilter(num_particles=3)
        
        # T=2 timesteps, N=3 particles, K=1 dimension
        particles = tf.constant([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]], dtype=DTYPE)
        weights = tf.constant([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=DTYPE)
        true_state = tf.constant([[[0.0]], [[1.0]]], dtype=DTYPE)
        omat_vals = ledh.compute_omat(particles, weights, true_state)
        
        # Expected OMAT for T=0: (1 + 2 + 3) / 3 = 2.0
        # Expected OMAT for T=1: (3 + 4 + 5) / 3 = 4.0
        np.testing.assert_allclose(omat_vals, [2.0, 4.0], atol=1e-5)


    def test_ledh_predict_matches_ekf(self, lg_ssm):
        """
        Since LEDH relies on the EKF for its auxiliary covariance, 
        the predicted m and P must perfectly match standard EKF predictions.
        """
        ledh = LocalizedExactDaumHuangFilter(num_particles=100)
        ledh.load_ssm(lg_ssm)
        particles, m_ekf, P_ekf = ledh.initialize_state()
        m_ekf = tf.constant([2.0], dtype=DTYPE)
        P_ekf = tf.constant([[0.5]], dtype=DTYPE)
        particles_pred, m_pred, P_pred = ledh.predict((particles, m_ekf, P_ekf))
        np.testing.assert_allclose(m_pred.numpy(), [1.8], atol=1e-5)
        np.testing.assert_allclose(P_pred.numpy(), [[0.415]], atol=1e-5)


    def test_ledh_update_flow_migration(self, lg_ssm):
        """
        Verifies that localized particles are actually moved by the flow equations.
        """
        ledh = LocalizedExactDaumHuangFilter(num_particles=50, num_steps=10)
        ledh.load_ssm(lg_ssm)
        state = ledh.initialize_state()
        state_pred = ledh.predict(state)
        particles_prior = state_pred[0]
        
        obs = tf.constant([50.0], dtype=DTYPE)                             # Pass a massive observation to force a large particle drift
        state_upd, est, metrics = ledh.update(state_pred, obs)
        particles_post = state_upd[0]
        drift_magnitude = tf.norm(particles_post - particles_prior).numpy()                # Particles should have moved significantly toward the observation
        assert drift_magnitude > 1.0, "Particles failed to migrate during the flow steps."


    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "lorenz_ssm", "sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_long_horizon_stability(self, ssm_fixture, seed, request):
        """
        Runs the full JIT-compiled LEDH filter loop over various models for 100 timesteps.
        Validates that continuous local flow equations do not generate NaNs.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        ssm_model = request.getfixturevalue(ssm_fixture)
        ledh = LocalizedExactDaumHuangFilter(num_particles=100, num_steps=20)
        ledh.load_ssm(ssm_model)
        
        T = 100
        
        if(ssm_fixture == "sv_ssm"):
            observations = tf.math.abs(tf.random.normal((T,), dtype=DTYPE)) + 0.1
        elif(ssm_fixture == "msv_ssm"):
            observations = tf.math.abs(tf.random.normal((T, ledh.ny), dtype=DTYPE)) + 0.1
        else:
            observations = tf.random.normal((T, ledh.ny), dtype=DTYPE)
            
        true_states = tf.random.normal((T, ledh.nx), dtype=DTYPE)
        metrics = ledh.run_filter(observations, true_states=true_states)
        estimates = metrics['estimates']
        
        assert not tf.reduce_any(tf.math.is_nan(estimates)), f"NaN discovered in Estimates for {ssm_fixture}."
        assert not tf.reduce_any(tf.math.is_inf(estimates)), f"Inf discovered in Estimates for {ssm_fixture}."
        assert 'avg_flow_cond' in metrics
        assert 'omat' in metrics
        assert metrics['avg_flow_cond'] < float('inf')