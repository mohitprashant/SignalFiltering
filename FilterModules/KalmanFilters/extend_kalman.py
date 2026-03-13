import tensorflow as tf
import numpy as np
from typing import Any, Tuple, Dict, Optional

from .kalman import KalmanFilter
from StateSpaceModels.ssm_base import SSM

DTYPE = tf.float32



class ExtendedKalmanFilter(KalmanFilter):
    """
    Extended Kalman Filter (EKF).

    It uses automatic differentiation (tf.GradientTape.jacobian) to compute 
    multivariate Jacobian matrices dynamically based on the loaded model.
    """
    def __init__(self, label = "EKF", joseph_form = True) -> None:
        """
        Initializes the EKF.

        Args:
            label (str): Label for metric tracking.
            joseph_form (bool): Whether to use Joseph Form for covariance updates to 
                                guarantee PSD.
        """
        super().__init__(label=label, joseph_form=joseph_form)
        self.dtype = DTYPE


    # def load_ssm(self, ssm_model) -> None:
    #     """
    #     Loads the State Space Model (SSM) and its parameters into the filter.

    #     Args:
    #         ssm_model (SSM)
    #     """
    #     model_name = ssm_model.__class__.__name__

    #     if(model_name == "LinearGaussianSSM"):
    #         self.nx = ssm_model.nx
    #         self.ny = ssm_model.ny

    #         self.f_func = lambda x: tf.linalg.matvec(ssm_model.A, x)
    #         self.h_func = lambda x: tf.linalg.matvec(ssm_model.C, x)
    #         self.Q = tf.matmul(ssm_model.B, ssm_model.B, transpose_b=True)
    #         self.R = tf.matmul(ssm_model.D, ssm_model.D, transpose_b=True)
    #         self.P_init = ssm_model.Sigma_init
    #         self.preprocess_obs = lambda y: tf.reshape(y, [-1, self.ny])

    #     elif(model_name == "Lorenz96Model"):
    #         self.nx = ssm_model.K
    #         self.ny = ssm_model.K

    #         self.f_func = lambda x: ssm_model.rk4_step(x)
    #         self.h_func = lambda x: x
    #         self.Q = ssm_model.Q
    #         self.R = ssm_model.R
    #         self.P_init = tf.eye(self.nx, dtype=self.dtype) * (0.1**2)
    #         self.preprocess_obs = lambda y: tf.reshape(y, [-1, self.ny])

    #     elif(model_name == "StochasticVolatilityModel"):
    #         self.nx = 1
    #         self.ny = 1

    #         self.f_func = lambda x: ssm_model.alpha * x
    #         self.h_func = lambda x: x + tf.math.log(ssm_model.beta**2) - 1.2704         # Obs offset
    #         self.Q = tf.reshape(ssm_model.sigma**2, [1, 1])
    #         self.R = tf.reshape(tf.constant((np.pi**2) / 2.0, dtype=self.dtype), [1, 1])
    #         self.P_init = tf.reshape(self.Q / (1.0 - ssm_model.alpha**2), [1, 1])
    #         self.preprocess_obs = lambda y: tf.reshape(tf.math.log(y**2 + 1e-8), [-1, 1])   # Log-squared transform

    #     elif(model_name == "MultivariateStochasticVolatilityModel"):
    #         self.nx = ssm_model.p
    #         self.ny = ssm_model.p

    #         # if(ssm_model.p <= 1):
    #         #     raise ValueError(f"Shape is {ssm_model.phi.shape}. Use univariate svssm for single dim. stoch vol.")
            
    #         phi_matrix = ssm_model.phi
    #         if(len(phi_matrix.shape) == 1):
    #             phi_matrix = tf.linalg.diag(phi_matrix)
            
    #         self.f_func = lambda x: tf.linalg.matvec(phi_matrix, x)
    #         self.h_func = lambda x: x + tf.math.log(ssm_model.beta**2) - 1.2704
    #         self.Q = ssm_model.sigma_eta
    #         self.R = tf.eye(self.ny, dtype=self.dtype) * ((np.pi**2) / 2.0)
            
    #         var = tf.linalg.diag_part(self.Q) / (1.0 - ssm_model.phi**2 + 1e-4)
    #         self.P_init = tf.linalg.diag(var)
    #         self.preprocess_obs = lambda y: tf.reshape(tf.math.log(y**2 + 1e-8), [-1, self.ny])     # Log-squared transform for MSV

    #     else:
    #         raise ValueError(f"Unsupported SSM Model Type: {model_name}")
            
    #     self.eye_nx = tf.eye(self.nx, dtype=self.dtype)

    def load_ssm(self, ssm_model: SSM) -> None:
        """
        Loads the State Space Model (SSM) and its parameters into the filter.

        Args:
            ssm_model (SSM)
        """
        comps = ssm_model.filter_components()
        self.nx, self.ny = comps["nx"], comps["ny"]
        self.f_func, self.h_func = comps["f_func"], comps["h_func"]
        self.Q, self.R, self.P_init = comps["Q"], comps["R"], comps["P_init"]
        self.preprocess_obs = comps["preprocess_obs"]
        self.eye_nx = tf.eye(self.nx, dtype=self.dtype)


    def initialize_state(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Initializes the filter at t=0."""
        x_curr = tf.zeros(self.nx, dtype=self.dtype)
        P_curr = self.P_init
        return (x_curr, P_curr)


    def f(self, x: tf.Tensor) -> tf.Tensor:
        """Applies the state transition function assigned during load_ssm."""
        return self.f_func(x)


    def h(self, x: tf.Tensor) -> tf.Tensor:
        """Applies the measurement function assigned during load_ssm."""
        return self.h_func(x)


    def get_jacobian(self, func: callable, x: tf.Tensor) -> tf.Tensor:
        """
        Computes the Jacobian matrix of a vector-valued function at x.
        Returns a matrix of shape (output_dim, input_dim).
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)
        return tape.jacobian(y, x)


    def predict(self, state: Tuple[tf.Tensor, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Performs the Extended Kalman Filter prediction step.

        Args:
            state (tuple): (x_curr, P_curr) the current state and covariance estimate.

        Returns:
            tuple: (x_pred, P_pred) predicted state and covariance for the next step.
        """
        x_curr, P_curr = state
        x_pred = self.f(x_curr)                            
        F = self.get_jacobian(self.f, x_curr)
        P_pred = tf.matmul(F, tf.matmul(P_curr, F, transpose_b=True)) + self.Q     # P_pred = F * P_curr * F^T + Q
        return (x_pred, P_pred)


    def update(self, state_pred: Tuple[tf.Tensor, tf.Tensor], observation: tf.Tensor) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor, tf.Tensor]:
        """
        Performs the Extended Kalman Filter update step and calculates step metrics.

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
        z = observation
        
        H = self.get_jacobian(self.h, x_pred)
        S = tf.matmul(H, tf.matmul(P_pred, H, transpose_b=True)) + self.R          # Innovation Covariance: S = H * P_pred * H^T + R
        cond_S = self.get_condition_number(S)
        
        HP_T = tf.matmul(H, P_pred, transpose_b=True)                               # Kalman Gain: K = P_pred * H^T * S^-1
        K_transpose = tf.linalg.solve(S, HP_T)
        K = tf.transpose(K_transpose)
        
        y_res = z - self.h(x_pred)
        x_new = x_pred + tf.linalg.matvec(K, y_res)                                 # State Update
        KC = tf.matmul(K, H)                                                        # Covariance Update
        I_KC = self.eye_nx - KC
        
        if(self.joseph_form):
            P_term1 = tf.matmul(I_KC, tf.matmul(P_pred, I_KC, transpose_b=True))
            P_term2 = tf.matmul(K, tf.matmul(self.R, K, transpose_b=True))
            P_new = P_term1 + P_term2
        else:
            P_new = tf.matmul(I_KC, P_pred)

        cond_P = self.get_condition_number(P_new)
        metrics = tf.stack([cond_S, cond_P])
        return (x_new, P_new), x_new, metrics


    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Overrides the superclass method.
        """
        processed_obs = self.preprocess_obs(observations)
        return super().run_filter(processed_obs, true_states)