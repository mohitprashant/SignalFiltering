import pytest
import numpy as np
import tensorflow as tf

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from FilterModules.KalmanFilters.kalman import KalmanFilter 

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 99, 90210, 6718]
DTYPE = tf.float32


class TestKalmanFilter:
    @pytest.fixture(params=[
        {
            "A": [[1.0, 0.1], [0.0, 1.0]],            # Constant velocity 1D
            "B": [[0.0], [0.1]],                      # Process noise on velocity
            "C": [[1.0, 0.0]],                       
            "D": [[0.5]],                             # Observation noise
            "Sigma_init": [[1.0, 0.0], [0.0, 1.0]]
        },
        {
            "A": [[0.9, 0.0], [0.0, 0.8]],            # Autoregressive decay checking
            "B": [[1.0, 0.0], [0.0, 1.0]], 
            "C": [[1.0, 1.0]],                   
            "D": [[0.1]],
            "Sigma_init": [[0.5, 0.0], [0.0, 0.5]]
        }
    ])
    def kf_instance(self, request):
        """Fixture providing a configured Kalman Filter and its underlying SSM."""
        c = request.param
        ssm = LinearGaussianSSM(c["A"], c["B"], c["C"], c["D"], c["Sigma_init"])
        kf = KalmanFilter(label="TestKF", joseph_form=True)
        kf.load_ssm(ssm)
        return kf, ssm


    def test_condition_number(self, kf_instance):
        """Verifies the manual condition number calculation."""
        kf, _ = kf_instance
        
        # Condition number of Identity matrix should be 1.0
        I = tf.eye(5, dtype=DTYPE)
        cond_I = kf.get_condition_number(I)
        np.testing.assert_allclose(cond_I.numpy(), 1.0, atol=1e-6)
        
        # Test known scaled diagonal matrix
        diag_mat = tf.linalg.diag([10.0, 5.0, 2.0, 1.0])
        cond_diag = kf.get_condition_number(diag_mat)
        np.testing.assert_allclose(cond_diag.numpy(), 10.0, atol=1e-6)


    def test_predict_mathematical_correctness(self, kf_instance):
        """Manually calculates the prediction step to verify exact mathematical correctness."""
        kf, _ = kf_instance
        x_curr = tf.constant([2.0, -1.0], dtype=DTYPE)
        P_curr = tf.constant([[0.5, 0.1], [0.1, 0.5]], dtype=DTYPE)
        x_pred, P_pred = kf.predict((x_curr, P_curr))

        A_np = kf.A.numpy()
        Q_np = kf.Q.numpy()
        x_curr_np = x_curr.numpy()
        P_curr_np = P_curr.numpy()
        expected_x_pred = A_np @ x_curr_np
        expected_P_pred = A_np @ P_curr_np @ A_np.T + Q_np
        
        np.testing.assert_allclose(x_pred.numpy(), expected_x_pred, atol=1e-6)
        np.testing.assert_allclose(P_pred.numpy(), expected_P_pred, atol=1e-6)


    def test_update_mathematical_correctness(self, kf_instance):
        """Manually calculates the update step in NumPy to verify exact mathematical correctness."""
        kf, _ = kf_instance
        
        x_pred = tf.constant([2.0, -1.0], dtype=DTYPE)
        P_pred = tf.constant([[1.0, 0.2], [0.2, 1.0]], dtype=DTYPE)
        observation = tf.constant([1.5], dtype=DTYPE)
        (x_new, P_new), point_est, metrics = kf.update((x_pred, P_pred), observation)
        
        C_np = kf.C.numpy()
        R_np = kf.R.numpy()
        x_pred_np = x_pred.numpy()
        P_pred_np = P_pred.numpy()
        y_obs_np = observation.numpy()
        
        S_np = C_np @ P_pred_np @ C_np.T + R_np
        K_np = P_pred_np @ C_np.T @ np.linalg.inv(S_np)
        
        expected_x_new = x_pred_np + K_np @ (y_obs_np - C_np @ x_pred_np)
        
        # Joseph form manual calc
        I = np.eye(kf.nx)
        I_KC = I - K_np @ C_np
        expected_P_new = I_KC @ P_pred_np @ I_KC.T + K_np @ R_np @ K_np.T
        
        np.testing.assert_allclose(x_new.numpy(), expected_x_new, atol=1e-5)
        np.testing.assert_allclose(P_new.numpy(), expected_P_new, atol=1e-5)
        np.testing.assert_allclose(point_est.numpy(), expected_x_new, atol=1e-5)


    def test_joseph_vs_standard_form(self, kf_instance):
        """Checks that Joseph form and standard form yield numerically close covariances."""
        kf, _ = kf_instance
        
        x_pred = tf.constant([1.0, 1.0], dtype=DTYPE)
        P_pred = tf.constant([[2.0, 0.5], [0.5, 2.0]], dtype=DTYPE)
        observation = tf.constant([0.0], dtype=DTYPE)

        # Run with Joseph form
        kf.joseph_form = True
        (_, P_joseph), _, _ = kf.update((x_pred, P_pred), observation)
        
        # Run with standard form
        kf.joseph_form = False
        (_, P_standard), _, _ = kf.update((x_pred, P_pred), observation)
        
        np.testing.assert_allclose(P_joseph.numpy(), P_standard.numpy(), atol=1e-5)


    def test_invariants_symmetry_and_psd(self, kf_instance):
        """Ensures the covariance matrix remains symmetric and PSD."""
        kf, _ = kf_instance
        x, P = kf.initialize_state()

        for i in range(5):
            x, P = kf.predict((x, P))
            (x, P), _, _ = kf.update((x, P), tf.constant([float(i)]))
        
        P_np = P.numpy()
        np.testing.assert_allclose(P_np, P_np.T, atol=1e-6)                         # Symmetry check: P == P^T
        
        eigenvalues = np.linalg.eigvals(P_np)
        assert np.all(eigenvalues >= -1e-7), f"Covariance matrix lost PSD. Eigenvalues: {eigenvalues}"         # PSD check


    def test_edge_case_zero_measurement_noise(self):
        """If R=0, the filter should perfectly trust the observation for the observed state."""
        A = [[1.0, 0.0], [0.0, 1.0]]
        B = [[0.1], [0.1]]
        C = [[1.0, 0.0]]
        D = [[0.0]]                                                               # Zero observation noise
        Sigma_init = [[1.0, 0.0], [0.0, 1.0]]
        
        ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
        kf = KalmanFilter()
        kf.load_ssm(ssm)
        
        x_pred = tf.constant([0.0, 5.0], dtype=DTYPE)
        P_pred = tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
        obs = tf.constant([10.0], dtype=DTYPE)                                    # Observe State 1 as 10.0
        (x_new, P_new), _, _ = kf.update((x_pred, P_pred), obs)
        
        assert np.isclose(x_new[0].numpy(), 10.0, atol=1e-5)                       # State 0 should be perfectly updated to 10.0, state 1 unaffected
        assert np.isclose(P_new[0, 0].numpy(), 0.0, atol=1e-5)                    # Covariance for state 0 should collapse to 0


    def generate_large_lg_system(self, nx, ny):
        """Generate a stable high-dimensional system for stability testing."""
        A = np.diag(np.random.uniform(-0.95, 0.95, size=nx)).astype(np.float32)
        B = np.eye(nx, dtype=np.float32) * 0.1
        C = np.random.normal(0, 1.0, size=(ny, nx)).astype(np.float32)
        D = np.eye(ny, dtype=np.float32) * 0.5
        Sigma_init = np.eye(nx, dtype=np.float32)
        return A, B, C, D, Sigma_init
    

    @pytest.mark.parametrize("nx, ny", [(10, 5), (50, 10)])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_long_horizon_stability(self, nx, ny, seed):
        """Tests continuous-time iterations over long horizons for stability and non-explosion."""
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        A, B, C, D, Sigma_init = self.generate_large_lg_system(nx, ny)
        ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
        kf = KalmanFilter()
        kf.load_ssm(ssm)
        
        x, P = kf.initialize_state()
        T = 2000
        y_obs = tf.random.normal((T, ny), dtype=DTYPE)             # Generate dummy observations
        
        cond_S_max = 0.0
        cond_P_max = 0.0
        
        for t in range(T):
            x, P = kf.predict((x, P))
            (x, P), _, metrics = kf.update((x, P), y_obs[t])
            
            cond_S_max = max(cond_S_max, metrics[0].numpy())
            cond_P_max = max(cond_P_max, metrics[1].numpy())
        
        assert not tf.reduce_any(tf.math.is_nan(x)), "Matrix collapse (NaN) in State Estimate."
        assert not tf.reduce_any(tf.math.is_inf(x)), "Exploding gradients (Inf) in State Estimate."
        assert not tf.reduce_any(tf.math.is_nan(P)), "Matrix collapse (NaN) in Covariance."
        assert not tf.reduce_any(tf.math.is_inf(P)), "Exploding gradients (Inf) in Covariance."
        assert cond_S_max < float('inf'), "Condition number of Innovation Covariance S exploded."
        assert cond_P_max < float('inf'), "Condition number of State Covariance P exploded."