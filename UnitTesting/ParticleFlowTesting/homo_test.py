import pytest
import numpy as np
import tensorflow as tf

from FilterModules.ParticleFilters.homo_particle import StochPFPF
from FilterModules.homo_solver import HomotopySolver

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

# tf.config.set_visible_devices([], 'GPU')
DTYPE = tf.float32
SEEDS = [42] #, 123, 99, 90210, 6718]


class TestStochPFPFAndHomotopy:
    @pytest.fixture
    def lg_ssm_stable(self):
        """Standard 2D Stable Linear Gaussian Model."""
        A = [[0.9, 0.1], [0.0, 0.9]]
        B = [[0.1, 0.0], [0.0, 0.1]]
        C = [[1.0, 0.0], [0.0, 1.0]]
        D = [[0.5, 0.0], [0.0, 0.5]]
        Sigma_init = [[1.0, 0.0], [0.0, 1.0]]
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def lg_ssm_ill_conditioned(self):
        """
        3D Ill-Conditioned Linear Gaussian Model to test the stability of 
        Homotopy solver SVD and ODE integration.
        """
        A = np.eye(3, dtype=np.float32) * 0.95
        B = np.array([[1.0, 0.9, 0.9],
                      [0.9, 1.0, 0.9], 
                      [0.9, 0.9, 1.0]], dtype=np.float32) * 0.1
        C = np.eye(3, dtype=np.float32)
        D = np.array([[1.0, 0.8, 0.8],
                      [0.8, 1.0, 0.8], 
                      [0.8, 0.8, 1.0]], dtype=np.float32) * 0.5
        Sigma_init = np.eye(3, dtype=np.float32) * 5.0
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def lorenz_high_dim(self):
        """Higher dimensional chaotic system (K=10)."""
        return Lorenz96Model(K=4, F=8.0, dt=0.05, process_std=0.2, obs_std=1.5)

    @pytest.fixture
    def sv_ssm(self):
        """1D Stochastic Volatility for testing 1D edge cases."""
        return StochasticVolatilityModel(alpha=0.91, sigma=0.5, beta=0.5)

    @pytest.fixture
    def msv_dense(self):
        """4D Dense Multivariate Stochastic Volatility Model."""
        p = 4
        phi = np.array([0.9, 0.85, 0.95, 0.8], dtype=np.float32)
        sigma_eta = np.eye(p, dtype=np.float32) * 0.5 + 0.1
        sigma_eps = np.eye(p, dtype=np.float32) * 0.5 + 0.05
        beta = np.ones(p, dtype=np.float32) * 0.5
        return MultivariateStochasticVolatilityModel(p=p, phi=phi, sigma_eta=sigma_eta, beta=beta, sigma_eps=sigma_eps)


    def test_homotopy_solver_math_and_boundaries(self):
        """
        Tests that the Homotopy Solver successfully solves the BVP.
        beta(0) should be 0.0, and beta(1) should successfully hit ~1.0.
        """
        nx = 2
        steps = 20
        solver = HomotopySolver(nx=nx, mu=0.1, steps=steps)
        H0 = tf.eye(nx, dtype=DTYPE)
        Hh = tf.eye(nx, dtype=DTYPE) * 2.0
        betas, beta_dots = solver.solve(H0, Hh)
        
        assert betas.shape == (steps + 1,), "Betas array shape mismatch."
        assert beta_dots.shape == (steps + 1,), "Velocities array shape mismatch."
        np.testing.assert_allclose(betas[0].numpy(), 0.0, atol=1e-5)              # Check Boundary Value Conditions
        np.testing.assert_allclose(betas[-1].numpy(), 1.0, atol=0.2)             # Target bounds valid < 0.2
        assert np.all(np.diff(betas.numpy()) >= -1e-4), "Schedule beta(lambda) is not monotonic."


    def test_homotopy_solver_ill_conditioned_fallback(self):
        """
        Tests the solver's robustness against NaNs or highly ill-conditioned matrices.
        It should fall back to a linear schedule if the ODE breaks down.
        """
        nx = 2
        solver = HomotopySolver(nx=nx, mu=1.0, steps=10)
        H0 = tf.constant([[1e9, -1e9], [-1e9, 1e9]], dtype=DTYPE)      # bad conditioning
        Hh = tf.constant([[0.0, 0.0], [0.0, 0.0]], dtype=DTYPE)
        betas, beta_dots = solver.solve(H0, Hh)
        expected_fallback_betas = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(betas.numpy(), expected_fallback_betas, atol=1e-5)
        np.testing.assert_allclose(beta_dots.numpy(), np.ones(11), atol=1e-5)


    def test_stochpfpf_initialization_shapes(self, lg_ssm_ill_conditioned):
        """Verifies state tuples dimensions match between EKF and PF components."""
        N = 100
        pfpf = StochPFPF(num_particles=N, num_steps=10)
        pfpf.load_ssm(lg_ssm_ill_conditioned)
        particles, log_weights, m_ekf, P_ekf = pfpf.initialize_state()
        nx = 3
        
        assert particles.shape == (N, nx)
        assert log_weights.shape == (N,)
        assert m_ekf.shape == (nx,)
        assert P_ekf.shape == (nx, nx)

        expected_log_weight = -np.log(float(N))
        np.testing.assert_allclose(log_weights.numpy(), expected_log_weight, atol=1e-5)


    def test_stochpfpf_update_optimal_flow_migration(self, lg_ssm_stable):
        """
        Verifies that particles are actively migrated by the optimal beta schedule.
        """
        N = 50
        pfpf = StochPFPF(num_particles=N, num_steps=15, mu=0.5)
        pfpf.load_ssm(lg_ssm_stable)
        
        state_prior = pfpf.initialize_state()
        state_pred = pfpf.predict(state_prior)
        particles_prior = state_pred[0]
        obs = tf.constant([100.0, -100.0], dtype=DTYPE)           # Force heavy flow migration
        state_upd, est, metrics = pfpf.update(state_pred, obs)
        particles_post = state_upd[0]
        drift_magnitude = tf.norm(particles_post - particles_prior).numpy()
        assert drift_magnitude > 25.0, "Particles failed to migrate sufficiently under optimal Homotopy flow."
        
        ess = metrics[0].numpy()
        assert 0.0 < ess <= float(N), "ESS is mathematically out of logical bounds."


    def test_extreme_observation_edge_case(self, lorenz_high_dim):
        """
        Edge Case: An extremely wild observation in a highly chaotic system.
        Ensures the optimal flow solver clippping prevents numerical explosion.
        """
        pfpf = StochPFPF(num_particles=100, num_steps=20)
        pfpf.load_ssm(lorenz_high_dim)
        state = pfpf.initialize_state()
        state_pred = pfpf.predict(state)
        extreme_obs = tf.fill([4], 1e5)
        upd_state, x_est, metrics = pfpf.update(state_pred, extreme_obs)
        particles, log_weights, m_upd, P_upd = upd_state
        
        tf.debugging.assert_all_finite(particles, "Particles crashed to NaN on extreme observation.")
        tf.debugging.assert_all_finite(log_weights, "Weights crashed to NaN on extreme observation.")
        tf.debugging.assert_all_finite(x_est, "Estimate crashed to NaN on extreme observation.")


    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm_stable", "lg_ssm_ill_conditioned", "lorenz_high_dim", "sv_ssm", "msv_dense"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_long_horizon_stability_all_models(self, ssm_fixture, seed, request):
        """
        Tests the compiled StochPFPF loop over 100 timesteps across diverse models.
        Validates continuous tracking without NaNs or condition number explosions.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        ssm_model = request.getfixturevalue(ssm_fixture)
        pfpf = StochPFPF(num_particles=100, num_steps=20, mu=0.1, resample_threshold_ratio=0.5)
        pfpf.load_ssm(ssm_model)
        
        T = 100
        if(ssm_fixture == "msv_dense"):
            observations = tf.math.abs(tf.random.normal((T, pfpf.ny), dtype=DTYPE)) + 0.1
        elif(ssm_fixture == "sv_ssm"):
            observations = tf.math.abs(tf.random.normal((T,), dtype=DTYPE)) + 0.1
        else:
            observations = tf.random.normal((T, pfpf.ny), dtype=DTYPE)
            
        true_states = tf.random.normal((T, pfpf.nx), dtype=DTYPE)
        metrics = pfpf.run_filter(observations, true_states=true_states)
        estimates = metrics['estimates']
        avg_flow_cond = metrics['avg_flow_cond']

        assert not tf.reduce_any(tf.math.is_nan(estimates)), f"NaN discovered in Estimates for {ssm_fixture}."
        assert not tf.reduce_any(tf.math.is_inf(estimates)), f"Inf discovered in Estimates for {ssm_fixture}."
        assert 'rmse' in metrics and not np.isnan(metrics['rmse'])
        assert 'omat' in metrics and not np.isnan(metrics['omat'])
        assert avg_flow_cond < float('inf'), f"Flow Condition Number Exploded for {ssm_fixture}"