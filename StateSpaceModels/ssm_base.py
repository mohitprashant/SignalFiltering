import abc
import tensorflow as tf
import tensorflow_probability as tfp

DTYPE = tf.float32
tfd = tfp.distributions


class SSM(abc.ABC):
    """
    Abstract base class for State Space Models (SSMs).
    """
    
    @abc.abstractmethod
    def initial_dist(self) -> tfp.distributions.Distribution: # type: ignore
        """
        Defines the initial state distribution p(x_0).
        
        Returns:
            tfp.distributions.Distribution: The distribution for the initial state.
        """
        pass


    @abc.abstractmethod
    def transition_dist(self, x_prev: tf.Tensor) -> tfp.distributions.Distribution: # type: ignore
        """
        Defines the stochastic state transition distribution p(x_t | x_{t-1}).
        
        Args:
            x_prev (tf.Tensor): The state at the previous time step.
            
        Returns:
            tfp.distributions.Distribution: The distribution for the next state.
        """
        pass


    @abc.abstractmethod
    def observation_dist(self, x_curr: tf.Tensor) -> tfp.distributions.Distribution: # type: ignore
        """
        Defines the observation distribution p(y_t | x_t).
        
        Args:
            x_curr (tf.Tensor): The latent state at the current time step.
            
        Returns:
            tfp.distributions.Distribution: The distribution for the observation.
        """
        pass


    @tf.function(jit_compile=True)
    def simulate(self, T: int) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Simulates the state space model for T time steps.
        
        Args:
            T (int): Number of time steps to simulate.
            *args, **kwargs: Additional arguments (e.g. burn-in period).
            
        Returns:
            tuple: (states, observations) where both are tf.Tensor sequences.
        """
        print(f"Simulating for {T} steps...")
        x_history = tf.TensorArray(DTYPE, size=T)
        y_history = tf.TensorArray(DTYPE, size=T)

        x = self.initial_dist().sample()
        y = self.observation_dist(x).sample()
        x_history = x_history.write(0, x)
        y_history = y_history.write(0, y)

        for i in tf.range(1, T):
            x = self.transition_dist(x).sample()
            y = self.observation_dist(x).sample()
            x_history = x_history.write(i, x)
            y_history = y_history.write(i, y)

        return x_history.stack(), y_history.stack()
    

    @abc.abstractmethod
    def filter_components(self):
        """Returns the components needed by non-linear filters."""
        pass