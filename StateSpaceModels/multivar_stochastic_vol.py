from .ssm_base import SSM
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

DTYPE = tf.float32
tfd = tfp.distributions


class MultivariateStochasticVolatilityModel(SSM):
    """
    Multivariate Stochastic Volatility (MSV) Model.

    System Dynamics:
        x_{t+1} = diag(phi) * x_t + eta_t
        y_t     = beta * exp(x_t / 2) * eps_t
    """
    def __init__(self, p, phi, sigma_eta, sigma_eps, beta):
        """
        Initializes the Multivariate SV parameters.

        Args:
            p (int): Dimension of the latent state and observation spaces.
            phi (tf.Tensor): Auto-regressive coefficients for the latent log-volatility. 
                Shape (p,).
            sigma_eta (tf.Tensor): Covariance matrix for the state noise (eta). Shape (p, p).
            beta (tf.Tensor): Scaling coefficients for the observations. Shape (p,).
            sigma_eps (tf.Tensor): Covariance matrix for observation noise (epsilon). Shape (p, p).
        """
        self.p = int(p)
        self.phi = tf.convert_to_tensor(phi, dtype=DTYPE)
        self.beta = tf.convert_to_tensor(beta, dtype=DTYPE)
        self.sigma_eta = tf.convert_to_tensor(sigma_eta, dtype=DTYPE)
        self.sigma_eps = tf.convert_to_tensor(sigma_eps, dtype=DTYPE)
        
        self.chol_eps = tf.linalg.cholesky(self.sigma_eps)
        self.chol_eta = tf.linalg.cholesky(self.sigma_eta)
        self.precision_eta = tf.linalg.inv(self.sigma_eta + 1e-5 * tf.eye(self.p))
        self.dtype = DTYPE


    def initial_dist(self):
        """
        Returns the stationary distribution of the latent state process.

        Approximates the initial state distribution X_0 assuming stationarity.
        Variance is computed diagonally: sigma_eta_ii / (1 - phi_ii^2).

        Returns:
            tfd.MultivariateNormalDiag: The initial state distribution.
        """
        vars = tf.linalg.diag_part(self.sigma_eta) / (1.0 - self.phi**2 + 1e-4)
        return tfd.MultivariateNormalDiag(loc=tf.zeros(self.p), scale_diag=tf.sqrt(vars))


    def transition_dist(self, x_prev):
        """
        Computes the transition distribution p(x_t | x_{t-1}).

        Args:
            x_prev (tf.Tensor): Previous latent states. Shape (batch, p).

        Returns:
            tfd.MultivariateNormalTriL: The conditional distribution for the next state.
        """
        return tfd.MultivariateNormalTriL(loc=self.phi * x_prev, scale_tril=self.chol_eta)


    def observation_dist(self, x_curr):
        """
        Computes the observation distribution p(y_t | x_t).

        The covariance of y_t is constructed by scaling the base Cholesky 
        of epsilon by the volatility term exp(x_t / 2).

        Args:
            x_curr (tf.Tensor): Current latent states. Shape (batch, p).

        Returns:
            tfd.MultivariateNormalTriL: The likelihood distribution for observations.
        """
        x_safe = tf.clip_by_value(x_curr, -15.0, 15.0)
        scales = self.beta * tf.exp(x_safe / 2.0)
        scale_tril = scales[..., tf.newaxis] * self.chol_eps
        return tfd.MultivariateNormalTriL(loc=tf.zeros(self.p), scale_tril=scale_tril)
    

    def filter_components(self):
        """Returns the components needed by non-linear filters."""
        phi_matrix = self.phi
        if(len(phi_matrix.shape) == 1):
            phi_matrix = tf.linalg.diag(phi_matrix)

        var = tf.linalg.diag_part(self.sigma_eta) / (1.0 - self.phi**2 + 1e-4)

        return {
            "nx": self.p,
            "ny": self.p,
            "f_func": lambda x: tf.linalg.matvec(phi_matrix, x),
            "h_func": lambda x: x + tf.math.log(self.beta**2) - 1.2704,
            "Q": self.sigma_eta,
            "R": tf.eye(self.p, dtype=self.dtype) * ((np.pi**2) / 2.0),
            "P_init": tf.linalg.diag(var),
            "preprocess_obs": lambda y: tf.reshape(tf.math.log(y**2 + 1e-8), [-1, self.p])
        }
    
