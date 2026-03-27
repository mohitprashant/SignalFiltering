import tensorflow as tf
from typing import Any, Tuple, Dict, Optional
import tensorflow_probability as tfp

from .kalman import KalmanFilter
from StateSpaceModels.ssm_base import SSM

DTYPE = tf.float32
tfd = tfp.distributions


class UnscentedKalmanFilter(KalmanFilter):
    """
    Unscented Kalman Filter (UKF).
    """
    def __init__(self, label="UKF", joseph_form=False, alpha_ukf=1.0, beta_ukf=2.0, kappa=0.0):
        """
        Initializes the UKF with model parameters and sigma point weights.

        Args:
            label (str): Label for metric tracking.
            joseph_form (bool): Use Joseph Form for covariance updates
            alpha_ukf (float): Spread of the sigma points. (Set to 1.0 to avoid 
                               float32 catastrophic cancellation).
            beta_ukf (float): Prior knowledge of the distribution (2.0 is optimal for Gaussian).
            kappa (float): Secondary scaling parameter.
        """
        super().__init__(label=label, joseph_form=joseph_form)
        self.dtype = DTYPE
        self.alpha_ukf = alpha_ukf
        self.beta_ukf = beta_ukf
        self.kappa = kappa


    def load_ssm(self, ssm_model: SSM) -> None:
        """
        Loads the State Space Model (SSM) and its parameters into the filter.
        """
        comps = ssm_model.filter_components()
        self.nx, self.ny = comps["nx"], comps["ny"]
        self.f_func, self.h_func = comps["f_func"], comps["h_func"]
        self.Q, self.R, self.P_init = comps["Q"], comps["R"], comps["P_init"]
        self.preprocess_obs = comps["preprocess_obs"]
        self.eye_nx = tf.eye(self.nx, dtype=self.dtype)
        self._init_weights()


    def _init_weights(self) -> None:
        """Initializes the Unscented Transform weights."""
        n = tf.cast(self.nx, self.dtype)
        lam = self.alpha_ukf**2 * (n + self.kappa) - n
        
        w_m0 = lam / (n + lam)
        w_c0 = w_m0 + (1.0 - self.alpha_ukf**2 + self.beta_ukf)
        w_i = tf.fill([2 * self.nx], 1.0 / (2.0 * (n + lam)))

        self.Wm = tf.concat([tf.reshape(w_m0, [1]), w_i], axis=0)
        self.Wc = tf.concat([tf.reshape(w_c0, [1]), w_i], axis=0)
        self.scale = tf.sqrt(n + lam)


    def generate_sigma_points(self, x: tf.Tensor, P: tf.Tensor) -> tf.Tensor:
        """
        Generates 2n+1 sigma points based on current mean and covariance.
        """
        P_safe = (P + tf.transpose(P)) / 2.0 + self.eye_nx * 1e-6   # Enforce symmetry and add jitter for Cholesky numerical stability
        L = tf.linalg.cholesky(P_safe)
        offsets = self.scale * L
        offsets_t = tf.transpose(offsets)                           # Rows form the translation vectors
        
        x_expanded = tf.expand_dims(x, 0)
        sigmas_pos = x_expanded + offsets_t
        sigmas_neg = x_expanded - offsets_t
        return tf.concat([x_expanded, sigmas_pos, sigmas_neg], axis=0)


    def predict(self, state: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs the UKF prediction step via the Unscented Transform.
        """
        x_curr, P_curr = state
        sigmas = self.generate_sigma_points(x_curr, P_curr)
        sigmas_pred = self.f_func(sigmas)
        x_pred = sigmas_pred[0] + tf.reduce_sum(self.Wm[1:, tf.newaxis] * (sigmas_pred[1:] - sigmas_pred[0]), axis=0)
        diff = sigmas_pred - x_pred
        P_pred = tf.matmul(diff, self.Wc[:, tf.newaxis] * diff, transpose_a=True) + self.Q
        return (x_pred, P_pred)


    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Performs the Unscent Kalman Filter update step and calculates step metrics.
        """
        x_pred, P_pred = state_pred
        z = observation
        sigmas_update = self.generate_sigma_points(x_pred, P_pred)
        obs_pts = self.h_func(sigmas_update)
        y_pred = obs_pts[0] + tf.reduce_sum(self.Wm[1:, tf.newaxis] * (obs_pts[1:] - obs_pts[0]), axis=0)
        diff_y = obs_pts - y_pred
        diff_y_scaled = self.Wc[:, tf.newaxis] * diff_y
        
        S = tf.matmul(diff_y, diff_y_scaled, transpose_a=True) + self.R
        S = (S + tf.transpose(S)) / 2.0                                         # Enforce symmetry to prevent MatrixSolve failure
        cond_S = self.get_condition_number(S)

        obs_dist = tfp.distributions.MultivariateNormalFullCovariance(
            loc=tf.zeros_like(y_res), 
            covariance_matrix=S + tf.eye(self.ny, dtype=self.dtype) * 1e-6
        )
        step_log_likelihood = obs_dist.log_prob(y_res)                              # For PMMH likelihood tracking
        
        diff_x = sigmas_update - x_pred                                         
        Pxy = tf.matmul(diff_x, diff_y_scaled, transpose_a=True)
        
        K = tf.transpose(tf.linalg.solve(S, tf.transpose(Pxy)))                 
        y_res = z - y_pred
        x_new = x_pred + tf.linalg.matvec(K, y_res)
        
        P_new = P_pred - tf.matmul(K, tf.matmul(S, K, transpose_b=True))
        P_new = (P_new + tf.transpose(P_new)) / 2.0                             # Enforce symmetry to maintain PSD invariant
        cond_P = self.get_condition_number(P_new)
        
        # metrics = tf.stack([cond_S, cond_P])
        metrics = tf.stack([cond_S, cond_P, step_log_likelihood])
        return (x_new, P_new), x_new, metrics


    # def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
    #     processed_obs = self.preprocess_obs(observations)
    #     return super().run_filter(processed_obs, true_states)
    
    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Executes the UKF over a sequence of observations.
        Pre-processes the observations and leverages the base KalmanFilter 
        to track the marginal log-likelihood.
        """
        processed_obs = self.preprocess_obs(observations)
        
        # This calls the KalmanFilter.run_filter, automatically giving you the log_likelihood
        results = super().run_filter(processed_obs, true_states)
        
        self.metrics = results
        return results