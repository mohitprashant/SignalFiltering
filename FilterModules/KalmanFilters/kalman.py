import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple
import numpy as np
from typing import Any, Tuple, Dict, Optional

from ..filter_base import BaseFilter
from StateSpaceModels.linear_gaussian import LinearGaussianSSM

tfd = tfp.distributions
DTYPE = tf.float32


class KalmanFilter(BaseFilter):
    """
    Kalman Filter.
    
    The filter models a system with the following dynamics:
    x_{t} = A * x_{t-1} + w_t,  w_t ~ N(0, B*B^T)
    y_{t} = C * x_t + v_t,      v_t ~ N(0, D*D^T)
    """

    def __init__(self, label = "KalmanFilter", joseph_form = True) -> None:
        """
        Initializes the Kalman Filter configuration.

        Args:
            label (str): Label for metric tracking.
            joseph_form (bool): If True, uses the Joseph Form for covariance update 
                to ensure the resulting matrix remains symmetric and positive-definite.
        """
        super().__init__(label=label)
        self.joseph_form = joseph_form
        self.dtype = DTYPE


    def load_ssm(self, ssm_model: LinearGaussianSSM) -> None:
        """
        Loads the State Space Model (SSM) and its parameters into the filter.

        Args:
            ssm_model (LinearGaussianSSM): Must contain A, B, C, D, and Sigma_init.
        """
        self.A = tf.convert_to_tensor(ssm_model.A, dtype=self.dtype)
        self.C = tf.convert_to_tensor(ssm_model.C, dtype=self.dtype)
        self.B = tf.convert_to_tensor(ssm_model.B, dtype=self.dtype)
        self.Q = tf.matmul(self.B, self.B, transpose_b=True)
        self.D = tf.convert_to_tensor(ssm_model.D, dtype=self.dtype)
        self.R = tf.matmul(self.D, self.D, transpose_b=True)
        
        self.P_init = tf.convert_to_tensor(ssm_model.Sigma_init, dtype=self.dtype)
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.eye_nx = tf.eye(self.nx, dtype=self.dtype)


    def initialize_state(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initializes the filter at t=0.

        Returns:
            tuple: (x_curr, P_curr) initial state and covariance tensors.
        """
        x_curr = tf.zeros(self.nx, dtype=self.dtype)
        P_curr = self.P_init
        return (x_curr, P_curr)


    def get_condition_number(self, matrix: tf.Tensor) -> tf.Tensor:
        """
        Manually computes condition number using SVD.
        Cond(A) = sigma_max / sigma_min
        """
        s = tf.linalg.svd(matrix, compute_uv=False)       # set compute_uv=False, we only need singular values
        return tf.reduce_max(s) / tf.reduce_min(s)


    def predict(self, state: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs the Kalman Filter prediction step.

        Args:
            state (tuple): (x_curr, P_curr) the current state and covariance estimate.

        Returns:
            tuple: (x_pred, P_pred) predicted state and covariance for the next step.
        """
        x_curr, P_curr = state
        x_pred = tf.linalg.matvec(self.A, x_curr)
        P_pred = tf.matmul(self.A, tf.matmul(P_curr, self.A, transpose_b=True)) + self.Q
        return (x_pred, P_pred)


    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Performs the Kalman Filter update step (Measurement Update) and calculates step metrics.

        Args:
            state_pred (tuple): A tuple of (x_pred, P_pred) representing the predicted state 
                and covariance from the predict() step.
            observation (tf.Tensor): The actual measurement observed at this timestep.

        Returns:
            tuple: (updated_state, point_estimate, step_metrics)
                - updated_state (tuple): (x_new, P_new), the updated state mean and covariance.
                - point_estimate (tf.Tensor): The extracted state estimate (x_new) to be recorded.
                - step_metrics (tf.Tensor): A 1D tensor tracking the condition numbers for 
                  matrices S and P_new at this timestep, formatted as tf.stack([cond_S, cond_P]).
        """
        x_pred, P_pred = state_pred
        y_pred = tf.linalg.matvec(self.C, x_pred)
        y_res = observation - y_pred                                                           # Innovation                          
        
        S = tf.matmul(self.C, tf.matmul(P_pred, self.C, transpose_b=True)) + self.R            # Innovation Covariance     
        cond_S = self.get_condition_number(S)

        obs_dist = tfp.distributions.MultivariateNormalFullCovariance(
            loc=tf.zeros_like(y_pred), 
            covariance_matrix=S + tf.eye(self.ny, dtype=self.dtype) * 1e-6
        )
        step_log_likelihood = obs_dist.log_prob(y_res)                                        # For PMMH likelihood tracking

        CP_T = tf.matmul(self.C, P_pred, transpose_b=True)                                     # Kalman Gain                     
        K_transpose = tf.linalg.solve(S, CP_T) 
        K = tf.transpose(K_transpose)
        
        x_new = x_pred + tf.linalg.matvec(K, y_res)                                             # State Update
        
        KC = tf.matmul(K, self.C)                                                               # Covariance update
        I_KC = self.eye_nx - KC
        
        if(self.joseph_form):
            P_term1 = tf.matmul(I_KC, tf.matmul(P_pred, I_KC, transpose_b=True))
            P_term2 = tf.matmul(K, tf.matmul(self.R, K, transpose_b=True))
            P_new = P_term1 + P_term2
        else:
            P_new = tf.matmul(I_KC, P_pred)

        cond_P = self.get_condition_number(P_new)
        # return (x_new, P_new), x_new, tf.stack([cond_S, cond_P])
        return (x_new, P_new), x_new, tf.stack([cond_S, cond_P, step_log_likelihood])
    

    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Runs the filter and extracts the exact marginal log-likelihood 
        from the tracked step metrics.
        """
        # Call the underlying BaseFilter compiled loop and metrics gathering
        results = super().run_filter(observations, true_states)
        
        if 'step_metrics' in results:
            # step_metrics format: [cond_S, cond_P, step_log_likelihood]
            step_metrics_np = results['step_metrics']
            
            # The exact marginal log-likelihood is the sum over all timesteps
            results['log_likelihood'] = float(np.sum(step_metrics_np[:, 2]))
            
            # Optional: You can also cleanly export your condition numbers here
            results['avg_cond_S'] = float(np.mean(step_metrics_np[:, 0]))
            results['avg_cond_P'] = float(np.mean(step_metrics_np[:, 1]))
            
        self.metrics = results
        return results