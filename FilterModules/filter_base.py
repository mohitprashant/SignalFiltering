import abc
import time
import tracemalloc
import numpy as np
import tensorflow as tf
from typing import Any, Dict, Optional, Tuple
from StateSpaceModels.ssm_base import SSM

DTYPE = tf.float32


class BaseFilter(abc.ABC):
    """
    Abstract Base Class for filter implementations.
    Provides an interface and loop for running filters
    and gathering performance metrics.
    """
    def __init__(self, label = "BaseFilter") -> None:
        self.label = label
        self.ssm = None
        self.metrics = {}


    @abc.abstractmethod
    def load_ssm(self, ssm_model : SSM) -> None:
        """
        Loads the State Space Model (SSM) and its parameters into the filter.
        
        Args:
            ssm_model (SSM): An instance inheriting from your base SSM class.
        """
        pass


    @abc.abstractmethod
    def initialize_state(self) -> Any:
        """
        Initializes the filter at t=0.
        
        Returns:
            Any: The initial state representation.
        """
        pass


    @abc.abstractmethod
    def predict(self, state) -> Any:
        """
        Performs the prediction step (Time Update).
        
        Args:
            state: The current state representation of the filter.
            
        Returns:
            Any: The predicted state for the next timestep.
        """
        pass
    

    @abc.abstractmethod
    def update(self, state_pred: Any, observation: tf.Tensor) -> Tuple[Any, tf.Tensor, tf.Tensor]:
        """
        Performs the update step.
        
        Args:
            state_pred: The predicted state from the predict() step.
            observation (tf.Tensor): The current measurement/observation.
            
        Returns:
            tuple: (updated_state, point_estimate, step_metrics)
                   - updated_state: The new state representation after the update.
                   - point_estimate (tf.Tensor): The extracted point estimate (e.g. mean) to be recorded.
                   - step_metrics (tf.Tensor): A 1D tensor tracking metrics for steps.
        """
        pass


    @tf.function(jit_compile=True)
    def _compiled_loop(self, observations: tf.Tensor, initial_state: Any) -> tf.Tensor:
        """
        The core iterative loop. Internal.
        Uses tf.TensorArray to dynamically store estimates within the TF Graph.
        """
        T = tf.shape(observations)[0]
        estimates_ta = tf.TensorArray(dtype=DTYPE, size=T, clear_after_read=False)
        metrics_ta = tf.TensorArray(dtype=DTYPE, size=T, clear_after_read=False)
        state = initial_state

        for t in tf.range(T):
            state_pred = self.predict(state)
            state, estimate, step_metrics = self.update(state_pred, observations[t])
            estimates_ta = estimates_ta.write(t, estimate)
            metrics_ta = metrics_ta.write(t, step_metrics)

        return estimates_ta.stack(), metrics_ta.stack()


    def run_filter(self, observations: tf.Tensor, true_states: Optional[tf.Tensor] = None) -> Dict[str, Any]:
        """
        Runs the filter over a series of observations.
        Collects and stores performance metrics (Time, Memory, RMSE...).
        
        Args:
            observations (tf.Tensor): Tensor of shape (T, ...) containing observed values.
            true_states (tf.Tensor, optional): Ground truth states for RMSE calculation.
            
        Returns:
            dict: Performance metrics including RMSE, runtime, memory, and estimates.
        """
        state = self.initialize_state()

        tracemalloc.start()
        start_time = time.time()

        estimates_tensor, step_metrics_tensor = self._compiled_loop(observations, state)      # JIT Compiled XLA loop

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time = time.time() - start_time

        rmse = 0.0
        if(true_states is not None):
            rmse = np.sqrt(np.mean((true_states.numpy() - estimates_tensor.numpy())**2))

        self.metrics = {
            'label': self.label,
            'rmse': rmse,
            'time': total_time,
            'mem': peak_mem,
            'estimates': estimates_tensor,
            'step_metrics': step_metrics_tensor
        }
        return self.metrics