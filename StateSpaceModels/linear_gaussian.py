from .ssm_base import SSM
import tensorflow as tf
import tensorflow_probability as tfp

DTYPE = tf.float32
tfd = tfp.distributions


class LinearGaussianSSM(SSM):
    """
    Linear Gaussian Model.
    
    State Space: X in R^{n_x}
    Observation Space: Y in R^{n_y}
    
    Equations:
      X_1 ~ N(0, Sigma_init)
      X_n = A * X_{n-1} + B * V_n,  where V_n ~ N(0, I)
      Y_n = C * X_n + D * W_n,      where W_n ~ N(0, I)
    """
    def __init__(self, A, B, C, D, Sigma_init):
        """
        Initialize matrices as TensorFlow tensors.
        
        Args:
            A: State transition matrix (n_x, n_x)
            B: State noise mapping matrix (n_x, n_v)
            C: Observation matrix (n_y, n_x)
            D: Observation noise mapping matrix (n_y, n_w)
            Sigma_init: Initial state covariance matrix (n_x, n_x)
        """
        self.A = tf.convert_to_tensor(A, dtype=DTYPE)
        self.B = tf.convert_to_tensor(B, dtype=DTYPE)
        self.C = tf.convert_to_tensor(C, dtype=DTYPE)
        self.D = tf.convert_to_tensor(D, dtype=DTYPE)
        self.Sigma_init = tf.convert_to_tensor(Sigma_init, dtype=DTYPE)
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]

        self.Q_scale = tf.linalg.cholesky(tf.matmul(self.B, self.B, transpose_b=True))    # Covariance for transition: Q = B * B^T
        self.R_scale = tf.linalg.cholesky(tf.matmul(self.D, self.D, transpose_b=True))    # Covariance for observation: R = D * D^T
        self.Init_scale = tf.linalg.cholesky(self.Sigma_init)


    def initial_dist(self):
        """
        Returns the distribution for X_1 ~ N(0, Sigma) 
        """
        return tfd.MultivariateNormalTriL(loc=tf.zeros(self.nx, dtype=DTYPE),scale_tril=self.Init_scale)


    def transition_dist(self, x_prev):
        """
        Returns p(x_n | x_{n-1}) = N(A x_{n-1}, B B^T) 
        """
        loc = tf.linalg.matvec(self.A, x_prev)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.Q_scale)


    def observation_dist(self, x_curr):
        """
        Returns p(y_n | x_n) = N(C x_n, D D^T) 
        """
        loc = tf.linalg.matvec(self.C, x_curr)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.R_scale)
    

    def filter_components(self):
        """Returns the components needed by non-linear filters."""
        return {
            "nx": self.nx,
            "ny": self.ny,
            "f_func": lambda x: tf.linalg.matvec(self.A, x),
            "h_func": lambda x: tf.linalg.matvec(self.C, x),
            "Q": tf.matmul(self.B, self.B, transpose_b=True),
            "R": tf.matmul(self.D, self.D, transpose_b=True),
            "P_init": self.Sigma_init,
            "preprocess_obs": lambda y: tf.reshape(y, [-1, self.ny])
        }




