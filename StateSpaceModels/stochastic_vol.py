from .ssm_base import SSM
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

DTYPE = tf.float32
tfd = tfp.distributions


class StochasticVolatilityModel(SSM):
    """
    Univariate Stochastic Volatility Model.

    State Equation:
        X_n = alpha * X_{n-1} + sigma * V_n,  where V_n ~ N(0, 1)
        
    Observation Equation:
        Y_n = beta * exp(X_n / 2) * W_n,      where W_n ~ N(0, 1)
        
    Attributes:
        alpha (tf.Tensor): Autoregression coefficient (0 < alpha < 1).
        sigma (tf.Tensor): State noise standard deviation.
        beta (tf.Tensor): Observation scaling factor.
    """
    def __init__(self, alpha, sigma, beta, static_diff=False):
        """
        Initialize the model parameters.

        Args:
            alpha (float): Autoregression coefficient (e.g., 0.91).
            sigma (float): State noise std dev (e.g., 1.0).
            beta (float): Observation scale (e.g., 0.5).
        """
        if(static_diff):
            self.alpha = tf.convert_to_tensor(alpha, dtype=DTYPE)
            self.sigma = tf.convert_to_tensor(sigma, dtype=DTYPE)
            self.beta = tf.convert_to_tensor(beta, dtype=DTYPE)
        else:
            self.alpha = tf.Variable(alpha, dtype=DTYPE)
            self.sigma = tf.Variable(sigma, dtype=DTYPE)
            self.beta = tf.Variable(beta, dtype=DTYPE)


    def initial_dist(self):
        """
        Returns the initial stationary distribution for X_1.
        X_1 ~ N(0, sigma^2 / (1-alpha^2)).
        
        Returns:
            tfp.distributions.Normal: The distribution object for X_1.
        """
        variance = self.sigma**2 / (1.0 - self.alpha**2)
        return tfd.Normal(loc=0.0, scale=tf.sqrt(variance))


    def transition_dist(self, x_prev):
        """
        Returns the state transition distribution p(x_n | x_{n-1}).
        
        Args:
            x_prev (tf.Tensor): The state at time n-1.
            
        Returns:
            tfp.distributions.Normal: The distribution p(x_n | x_{n-1}).
        """
        return tfd.Normal(loc=self.alpha * x_prev, scale=self.sigma)


    def observation_dist(self, x_curr):
        """
        Returns the observation distribution p(y_n | x_n).
        
        Args:
            x_curr (tf.Tensor): The state at time n.
            
        Returns:
            tfp.distributions.Normal: The distribution p(y_n | x_n).
        """
        scale = self.beta * tf.exp(x_curr / 2.0)
        return tfd.Normal(loc=0.0, scale=scale)


    def filter_components(self):
        """Returns the components needed by non-linear filters."""
        return {
            "nx": 1,
            "ny": 1,
            "f_func": lambda x: self.alpha * x,
            "h_func": lambda x: x + tf.math.log(self.beta**2) - 1.2704,
            "Q": tf.reshape(self.sigma**2, [1, 1]),
            "R": tf.reshape(tf.constant((np.pi**2) / 2.0, dtype=DTYPE), [1, 1]),
            "P_init": tf.reshape((self.sigma**2) / (1.0 - self.alpha**2), [1, 1]),
            "preprocess_obs": lambda y: tf.reshape(tf.math.log(y**2 + 1e-8), [-1, 1])
        }
    

    def dynamic_filter_components(self):
        """
        Returns components as functions of dynamic tensor tuple `theta` for XLA.
        Expected theta structure: (alpha, sigma, beta)
        """
        return {
            "nx": 1,
            "ny": 1,
            
            # f(x) = alpha * x
            "f_func": lambda x, theta: theta[0] * x,
            
            # h(x) = x + log(beta^2) - 1.2704
            "h_func": lambda x, theta: x + tf.math.log(theta[2]**2) - 1.2704,
            
            # Q = sigma^2
            "Q_func": lambda theta: tf.reshape(theta[1]**2, [1, 1]),
            
            # R is derived from the log-chi-square approximation, stays constant
            "R_func": lambda theta: tf.reshape(tf.constant((np.pi**2) / 2.0, dtype=DTYPE), [1, 1]),
            
            # P_init = sigma^2 / (1 - alpha^2)
            "P_init_func": lambda theta: tf.reshape((theta[1]**2) / (1.0 - theta[0]**2 + 1e-4), [1, 1]),
            
            "preprocess_obs": lambda y: tf.reshape(tf.math.log(y**2 + 1e-8), [-1, 1])
        }