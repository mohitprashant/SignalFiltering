import pytest
import numpy as np
import tensorflow as tf

# Import the filter
from FilterModules.KalmanFilters.unscent_kalman import UnscentedKalmanFilter

# Import all supported SSMs
from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 99, 90210, 6718]
DTYPE = tf.float32

class TestUnscentedKalmanFilter:
    
    # ==========================================
    # FIXTURES
    # ==========================================
    
    @pytest.fixture
    def lg_ssm(self):
        """Linear Gaussian Model Fixture."""
        A = [[0.9, 0.1], [0.0, 0.8]]
        B = [[0.1, 0.0], [0.0, 0.1]]
        C = [[1.0, 0.0]]
        D = [[0.5]]
        Sigma_init = [[1.0, 0.0], [0.0, 1.0]]
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def lorenz_ssm(self):
        """Lorenz 96 Model Fixture."""
        return Lorenz96Model(K=10, F=8.0, dt=0.05)

    @pytest.fixture
    def sv_ssm(self):
        """1D Stochastic Volatility Model Fixture."""
        return StochasticVolatilityModel(alpha=0.91, sigma=0.5, beta=0.5)

    @pytest.fixture
    def msv_ssm(self):
        """Multivariate Stochastic Volatility Model Fixture."""
        phi = np.array([0.9, 0.85], dtype=np.float32)
        sigma_eta = np.array([[0.5, 0.1], [0.1, 0.5]], dtype=np.float32)
        sigma_eps = np.array([[0.5, 0.05],[0.05, 0.5]], dtype=np.float32)
        beta = np.array([0.5, 0.5], dtype=np.float32)
        return MultivariateStochasticVolatilityModel(p=2, phi=phi, sigma_eta=sigma_eta, beta=beta, sigma_eps=sigma_eps)

    # ==========================================
    # MATHEMATICAL CORRECTNESS
    # ==========================================

    def test_ukf_weights(self, lg_ssm):
        """Verifies that the Unscented Transform weights are calculated correctly and sum to 1."""
        ukf = UnscentedKalmanFilter()
        ukf.load_ssm(lg_ssm)
        
        # The sum of mean weights should exactly equal 1.0
        sum_Wm = tf.reduce_sum(ukf.Wm).numpy()
        np.testing.assert_allclose(sum_Wm, 1.0, atol=1e-3)
        
        # The sum of covariance weights depends on beta_ukf and alpha_ukf
        # Sum(Wc) = Sum(Wm) + (1 - alpha^2 + beta)
        expected_sum_Wc = 1.0 + (1.0 - ukf.alpha_ukf**2 + ukf.beta_ukf)
        sum_Wc = tf.reduce_sum(ukf.Wc).numpy()
        np.testing.assert_allclose(sum_Wc, expected_sum_Wc, atol=1e-3)

    def test_predict_linear_equivalence(self, lg_ssm):
        """
        The Unscented Transform is exact for linear functions.
        This verifies UKF predict step exactly matches standard analytical KF derivations.
        """
        ukf = UnscentedKalmanFilter()
        ukf.load_ssm(lg_ssm)
        
        x_curr = tf.constant([2.0, -1.0], dtype=DTYPE)
        P_curr = tf.constant([[0.5, 0.1], [0.1, 0.5]], dtype=DTYPE)
        
        # UKF Unscented Transform Output
        x_pred, P_pred = ukf.predict((x_curr, P_curr))
        
        # Analytical Linear Gaussian Output
        A_np = np.array([[0.9, 0.1], [0.0, 0.8]], dtype=np.float32)
        Q_np = ukf.Q.numpy()
        
        expected_x_pred = A_np @ x_curr.numpy()
        expected_P_pred = A_np @ P_curr.numpy() @ A_np.T + Q_np
        
        np.testing.assert_allclose(x_pred.numpy(), expected_x_pred, atol=1e-3)
        np.testing.assert_allclose(P_pred.numpy(), expected_P_pred, atol=1e-3)

    def test_update_linear_equivalence(self, lg_ssm):
        """Verifies that the UKF measurement update exactly matches standard KF for linear models."""
        ukf = UnscentedKalmanFilter()
        ukf.load_ssm(lg_ssm)
        
        x_pred = tf.constant([2.0, -1.0], dtype=DTYPE)
        P_pred = tf.constant([[1.0, 0.2], [0.2, 1.0]], dtype=DTYPE)
        observation = tf.constant([1.5], dtype=DTYPE)
        
        # UKF Update Output
        (x_new, P_new), point_est, metrics = ukf.update((x_pred, P_pred), observation)
        
        # Analytical Update Derivation
        C_np = np.array([[1.0, 0.0]], dtype=np.float32)
        R_np = ukf.R.numpy()
        x_pred_np = x_pred.numpy()
        P_pred_np = P_pred.numpy()
        z_np = observation.numpy()
        
        S_np = C_np @ P_pred_np @ C_np.T + R_np
        K_np = P_pred_np @ C_np.T @ np.linalg.inv(S_np)
        
        expected_x_new = x_pred_np + K_np @ (z_np - C_np @ x_pred_np)
        expected_P_new = P_pred_np - K_np @ S_np @ K_np.T
        
        np.testing.assert_allclose(x_new.numpy(), expected_x_new, atol=1e-4)
        np.testing.assert_allclose(P_new.numpy(), expected_P_new, atol=1e-4)

    # ==========================================
    # INVARIANTS (SYMMETRY & PSD)
    # ==========================================

    def test_invariants_symmetry_and_psd(self, lorenz_ssm):
        """Ensures the covariance matrix remains symmetric and Positive Semi-Definite during highly non-linear steps."""
        ukf = UnscentedKalmanFilter()
        ukf.load_ssm(lorenz_ssm)
        
        x = tf.zeros(ukf.nx, dtype=DTYPE)
        P = ukf.P_init
        
        # Pass through the chaotic non-linear Lorenz ODE multiple times
        for i in range(10):
            x, P = ukf.predict((x, P))
            dummy_obs = tf.fill([ukf.ny], float(i))
            (x, P), _, _ = ukf.update((x, P), dummy_obs)
            
        P_np = P.numpy()
        
        # 1. Symmetry check: P == P^T
        np.testing.assert_allclose(P_np, P_np.T, atol=1e-5)

        # 2. PSD check: All eigenvalues must be >= 0 (with minor tolerance for float32 precision)
        eigenvalues = np.linalg.eigvals(P_np)
        assert np.all(eigenvalues >= -1e-4), f"Covariance matrix lost PSD. Eigenvalues: {eigenvalues}"

    # ==========================================
    # EDGE CASES
    # ==========================================

    def test_edge_case_zero_measurement_noise(self):
        """
        If Measurement Noise (R) is 0, the filter should perfectly trust the observation.
        Variance should collapse to ~0 for the observed state.
        """
        # Linear System with ZERO Observation Noise
        A = [[1.0, 0.0], [0.0, 1.0]]
        B = [[0.1, 0.0], [0.0, 0.1]]
        C = [[1.0, 0.0]]  # Observing State 0
        D = [[0.0]]       # R = 0
        Sigma_init = [[1.0, 0.0], [0.0, 1.0]]
        
        ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
        # Assuming the SSM dynamically binds filter_components, if not, patch here manually.
        
        ukf = UnscentedKalmanFilter()
        ukf.load_ssm(ssm)
        
        x_pred = tf.constant([0.0, 5.0], dtype=DTYPE)
        P_pred = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
        obs = tf.constant([10.0], dtype=DTYPE)  # We observe State 0 as strictly 10.0
        
        (x_new, P_new), _, _ = ukf.update((x_pred, P_pred), obs)
        
        # State 0 should be perfectly updated to 10.0
        assert np.isclose(x_new[0].numpy(), 10.0, atol=1e-5)
        
        # The covariance for state 0 should collapse to ~0 
        # (It will be bounded by the 1e-6 jitter applied in `generate_sigma_points`)
        assert np.isclose(P_new[0, 0].numpy(), 0.0, atol=1e-5)

    # ==========================================
    # STABILITY OVER LONG HORIZONS
    # ==========================================

    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "lorenz_ssm", "sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_long_horizon_stability_all_models(self, ssm_fixture, seed, request):
        """
        Tests the UKF against models over a long horizon (T=500 steps) to ensure
        the Unscented Transform numerical properties do not degrade and explode to NaN/Inf.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        ssm_model = request.getfixturevalue(ssm_fixture)
        ukf = UnscentedKalmanFilter()
        ukf.load_ssm(ssm_model)
        T = 500
        
        # Generate dummy observation bounds depending on the model
        if ssm_fixture == "sv_ssm":
            # Volatility observations are normally strictly positive
            observations = tf.math.abs(tf.random.normal((T,), dtype=DTYPE)) + 0.1
        elif ssm_fixture == "msv_ssm":
            observations = tf.math.abs(tf.random.normal((T, ukf.ny), dtype=DTYPE)) + 0.1
        else:
            observations = tf.random.normal((T, ukf.ny), dtype=DTYPE)
        
        # Run the internal base loop
        metrics = ukf.run_filter(observations)
        estimates = metrics['estimates']
        cond_S = metrics['step_metrics'][:, 0]
        cond_P = metrics['step_metrics'][:, 1]
        
        assert not tf.reduce_any(tf.math.is_nan(estimates)), f"NaN discovered in Estimates for {ssm_fixture}."
        assert not tf.reduce_any(tf.math.is_inf(estimates)), f"Inf discovered in Estimates for {ssm_fixture}."
        assert tf.reduce_max(cond_S) < float('inf'), f"Condition number S exploded for {ssm_fixture}."
        assert tf.reduce_max(cond_P) < float('inf'), f"Condition number P exploded for {ssm_fixture}."