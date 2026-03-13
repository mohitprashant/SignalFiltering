import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class AuxiliaryEKF:
    """
    Extended Kalman Filter used to estimate the covariance matrix P 
    required by the Daum-Huang particle flow equations.
    """
    def __init__(self, model):
        """
        Initializes the Auxiliary EKF.

        Args:
            model (Lorenz96Model): The system model containing dynamics and noise matrices.
        """
        self.model = model
        self.K = model.K
        self.I = tf.eye(self.K)

    def predict(self, m, P):
        """
        Performs the EKF prediction step.
        
        Args:
            m (tf.Tensor): Mean estimate at k-1.
            P (tf.Tensor): Covariance estimate at k-1.
            
        Returns:
            tuple: (Predicted Mean, Predicted Covariance).
        """
        m_pred = self.model.rk4_step(m)                                 # State: x_{k|k-1} = f(x_{k-1|k-1})
        F = tf.squeeze(self.model.get_jacobian(tf.expand_dims(m, 0)))   # Covariance: P_{k|k-1} = F P F^T + Q
        P_pred = F @ P @ tf.transpose(F) + self.model.Q
        return m_pred, P_pred


    def update(self, m_pred, P_pred, y):
        """
        Performs the EKF update step (assuming Linear Observation H=I).
        
        Args:
            m_pred (tf.Tensor): Predicted mean.
            P_pred (tf.Tensor): Predicted covariance.
            y (tf.Tensor): Current observation.
            
        Returns:
            tuple: (Updated Mean, Updated Covariance).
        """
        
        y_res = y - m_pred                        # Innovation
        S = P_pred + self.model.R
        K_gain = P_pred @ tf.linalg.inv(S)        # Gain K = P S^-1
        
        m_upd = m_pred + tf.linalg.matvec(K_gain, y_res)
        P_upd = (self.I - K_gain) @ P_pred
        return m_upd, P_upd




class ParticleFlow:
    """
    Shared logic for calculating flow parameters A and b.
    Implements the Exact Daum-Huang equations [Ding & Coates].
    """
    def compute_Ab_and_cond(self, P, y_obs, x_lin, lam):
        """
        Computes the flow matrix A, drift vector b, and condition number of S.
        
        Equation: dx/dlambda = A x + b
        
        Args:
            P (tf.Tensor): Predicted Covariance Matrix (K, K).
            y_obs (tf.Tensor): Observation vector (K,).
            x_lin (tf.Tensor): Linearization point. 
                               Shape (K,) for EDH, or (N, K) for LEDH.
            lam (float): Current pseudo-time lambda (0 to 1).
        
        Returns:
            tuple: (A, b, condition_number)
                - A: Flow matrix (K, K).
                - b: Flow drift vector (K,) or (N, K).
                - condition_number: Condition number of the innovation matrix S.
        """
        
        S = lam * P + self.model.R                                                 # Matrix to invert: S = lambda * P + R
        s_vals = tf.linalg.svd(S, compute_uv=False)                                # Track Condition Number of S to monitor stability
        cond_num = tf.reduce_max(s_vals) / tf.reduce_min(s_vals)
        
        # A matrix
        S_inv = tf.linalg.inv(S)
        A = -0.5 * P @ S_inv
        
        # b vector
        # b = (I + 2LA) [ (I + LA) P R^-1 y + A x_lin ]
        R_inv_z = tf.linalg.matvec(tf.linalg.inv(self.model.R), y_obs)
        I = tf.eye(self.model.K)
        
        P_R_inv_z = tf.linalg.matvec(P, R_inv_z)
        term1 = tf.linalg.matvec((I + lam * A), P_R_inv_z)
        
        if len(x_lin.shape) > 1:                                                  # LEDH Case: x_lin is (N, K)
            term2 = tf.transpose(A @ tf.transpose(x_lin))
            inner = term1 + term2 
            b = tf.transpose((I + 2 * lam * A) @ tf.transpose(inner))
        else:                                                                     # EDH Case: x_lin is (K,)
            term2 = tf.linalg.matvec(A, x_lin)
            b = tf.linalg.matvec((I + 2 * lam * A), (term1 + term2))
            
        return A, b, cond_num