import pytest
import numpy as np
import tensorflow as tf

# Import the filter
# Replace with the actual path in your project
from FilterModules.KalmanFilters.extend_kalman import ExtendedKalmanFilter

# Import all supported SSMs
from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 99, 90210, 6718]
DTYPE = tf.float32


class TestExtendedKalmanFilter:
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
        return Lorenz96Model(K=20, F=8.0, dt=0.05)


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


    def test_jacobian_computation(self):
        """Tests the autodiff Jacobian computation against a known analytical derivative."""
        ekf = ExtendedKalmanFilter()

        # Test function: f(x_0, x_1) = [x_0^2 + x_1, 3 * x_1^3]
        # Analytical Jacobian: [[2*x_0, 1], [0, 9*x_1^2]]
        def test_func(x):
            return tf.stack([x[0]**2 + x[1], 3.0 * x[1]**3])

        x_val = tf.constant([2.0, 1.0], dtype=DTYPE)
        jacobian = ekf.get_jacobian(test_func, x_val)
        expected_jacobian = np.array([[4.0, 1.0], [0.0, 9.0]], dtype=np.float32)
        np.testing.assert_allclose(jacobian.numpy(), expected_jacobian, atol=1e-5)


    def test_predict_mathematical_correctness(self, sv_ssm):
        """Manually computes Taylor expansion for the predict step to verify tf.GradientTape logic."""
        ekf = ExtendedKalmanFilter()
        ekf.load_ssm(sv_ssm)
        
        # Override the transition to a non-linear test function: f(x) = alpha * sin(x)
        alpha_val = sv_ssm.alpha.numpy()
        ekf.f_func = lambda x: alpha_val * tf.math.sin(x)
        
        x_curr = tf.constant([np.pi / 4.0], dtype=DTYPE)
        P_curr = tf.constant([[2.0]], dtype=DTYPE)
        x_pred, P_pred = ekf.predict((x_curr, P_curr))
        
        # F = d(alpha * sin(x))/dx = alpha * cos(x)
        expected_x_pred = alpha_val * np.sin(np.pi / 4.0)
        F_manual = alpha_val * np.cos(np.pi / 4.0)
        Q_np = ekf.Q.numpy()[0, 0]
        
        expected_P_pred = (F_manual**2) * 2.0 + Q_np
        
        np.testing.assert_allclose(x_pred.numpy()[0], expected_x_pred, atol=1e-5)
        np.testing.assert_allclose(P_pred.numpy()[0, 0], expected_P_pred, atol=1e-5)


    def test_update_mathematical_correctness(self, lg_ssm):
        """Manually computes the measurement update step to verify Kalman Gain and State Update logic."""
        ekf = ExtendedKalmanFilter(joseph_form=False)
        ekf.load_ssm(lg_ssm)
        
        x_pred = tf.constant([2.0, -1.0], dtype=DTYPE)
        P_pred = tf.constant([[1.0, 0.2], [0.2, 1.0]], dtype=DTYPE)
        observation = tf.constant([1.5], dtype=DTYPE)
        
        (x_new, P_new), point_est, metrics = ekf.update((x_pred, P_pred), observation)
        H_np = np.array(lg_ssm.C, dtype=np.float32)                                    # Need this to execute, tf error in transpose
        
        R_np = ekf.R.numpy()
        x_pred_np = x_pred.numpy()
        P_pred_np = P_pred.numpy()
        z_np = observation.numpy()
        
        S_np = H_np @ P_pred_np @ H_np.T + R_np
        K_np = P_pred_np @ H_np.T @ np.linalg.inv(S_np)
        expected_x_new = x_pred_np + K_np @ (z_np - H_np @ x_pred_np)
        expected_P_new = (np.eye(2) - K_np @ H_np) @ P_pred_np
        
        np.testing.assert_allclose(x_new.numpy(), expected_x_new, atol=1e-5)
        np.testing.assert_allclose(P_new.numpy(), expected_P_new, atol=1e-5)
        np.testing.assert_allclose(point_est.numpy(), expected_x_new, atol=1e-5)



    @pytest.mark.parametrize("joseph_form", [True, False])
    def test_invariants_symmetry_and_psd(self, lorenz_ssm, joseph_form):
        """Ensures the covariance matrix remains symmetric and Positive Semi-Definite over highly non-linear steps."""
        ekf = ExtendedKalmanFilter(joseph_form=joseph_form)
        ekf.load_ssm(lorenz_ssm)
        
        x, P = ekf.initialize_state()
        for i in range(10):                                 # Run 10 times
            x, P = ekf.predict((x, P))
            dummy_obs = tf.fill([lorenz_ssm.K], float(i))
            (x, P), _, _ = ekf.update((x, P), dummy_obs)
            
        P_np = P.numpy()
        np.testing.assert_allclose(P_np, P_np.T, atol=1e-5)      # Check sym

        eigenvalues = np.linalg.eigvals(P_np)
        assert np.all(eigenvalues >= -1e-5), f"Covariance matrix lost PSD. Eigenvalues: {eigenvalues}"


    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "lorenz_ssm", "sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_long_horizon_stability_all_models(self, ssm_fixture, seed, request):
        """
        Tests the EKF against models over a long horizon.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        ssm_model = request.getfixturevalue(ssm_fixture)
        ekf = ExtendedKalmanFilter(joseph_form=True)
        ekf.load_ssm(ssm_model)
        T = 500
        
        if(ssm_fixture == "sv_ssm"):
            observations = tf.math.abs(tf.random.normal((T,), dtype=DTYPE)) + 0.1               # Non zero the values

        elif(ssm_fixture == "msv_ssm"):
            observations = tf.math.abs(tf.random.normal((T, ekf.ny), dtype=DTYPE)) + 0.1

        else:
            observations = tf.random.normal((T, ekf.ny), dtype=DTYPE)
        
        metrics = ekf.run_filter(observations)
        estimates = metrics['estimates']
        cond_S, cond_P = metrics['step_metrics'][:, 0], metrics['step_metrics'][:, 1]
        
        assert not tf.reduce_any(tf.math.is_nan(estimates)), f"NaN discovered in Estimates for {ssm_fixture}."
        assert not tf.reduce_any(tf.math.is_inf(estimates)), f"Inf discovered in Estimates for {ssm_fixture}."
        assert tf.reduce_max(cond_S) < float('inf'), f"Condition number S exploded for {ssm_fixture}."
        assert tf.reduce_max(cond_P) < float('inf'), f"Condition number P exploded for {ssm_fixture}."