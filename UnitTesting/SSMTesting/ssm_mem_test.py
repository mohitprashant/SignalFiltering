import pytest
import gc
import tracemalloc
import tensorflow as tf

from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.linear_gaussian import LinearGaussianSSM

tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 99, 90210, 6718]
DTYPE = tf.float32


class TestMemoryUsage:
    """
    Checks for memory leaks during repetitive executions of the SSMs.
    TensorFlow graph compilation allocates memory on the first run, so we warm up first.
    """
    @pytest.fixture(params=[
        ("LGSSM_2D", LinearGaussianSSM(
            A=[[0.9, 0.1], [0.0, 0.8]], B=[[1.0, 0.0], [0.0, 1.0]],
            C=[[1.0, 0.0]], D=[[0.5]], Sigma_init=[[1.0, 0.0], [0.0, 1.0]]
        )),
        ("LGSSM_3D", LinearGaussianSSM(
            A=[[0.5, 0.1, 0.0], [0.0, 0.5, 0.1], [0.0, 0.0, 0.5]], 
            B=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            C=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
            D=[[0.5, 0.0], [0.0, 0.5]], 
            Sigma_init=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )),

        ("Lorenz_K5", Lorenz96Model(K=5, F=8.0, dt=0.01)),
        ("Lorenz_K10", Lorenz96Model(K=10, F=10.0, dt=0.05)),
        
        ("SV_Standard", StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)),
        ("SV_HighPersistence", StochasticVolatilityModel(alpha=0.99, sigma=0.1, beta=1.0)),
        
        ("MSV_2D", MultivariateStochasticVolatilityModel(
            p=2,
            phi=[0.9, 0.8], 
            sigma_eta=[[1.0, 0.1],[0.1, 1.0]], 
            sigma_eps=[[0.5, 0.05], [0.05, 0.5]], 
            beta=[0.5, 0.5]
        )),
        ("MSV_1D_Fallback", MultivariateStochasticVolatilityModel(
            p=1,
            phi=[0.95], 
            sigma_eta=[[1.0]], 
            sigma_eps=[[0.5]],
            beta=[1.0]
        ))
    ])
    def model_config(self, request):
        """Provides parameterized instances of all models and their config."""
        return request.param


    @pytest.mark.parametrize("seed", SEEDS)
    def test_simulate_memory_leaks(self, model_config, seed):
        """
        Runs simulate() multiple times and verifies that memory usage 
        does not grow unboundedly after graph compilation.
        """
        name, model = model_config
        tf.random.set_seed(seed)
        model.simulate(10)        
        gc.collect()                           # Force garbage collection
        
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        for _ in range(10):                    # Mimic training loop
            _ = model.simulate(100)
            
        gc.collect()                           # Clean up tensors generated during the loop
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_leak_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        total_leak_mb = total_leak_bytes / (1024 * 1024)

        # TF might have cache fluctuations, cannot assert == 0
        assert total_leak_mb < 1.0, (
            f"Memory leak detected in {name} ({model.__class__.__name__}) on seed {seed} "
            f"Leaked {total_leak_mb:.4f} MB over 10 iterations."
        )


    @pytest.mark.parametrize("seed", SEEDS)
    def test_tensorarray_scaling_memory(self, model_config, seed):
        """
        Tests if memory scales reasonably when increasing the trajectory length (T) 
        across all models and configurations. Massive trajectories shouldn't Out of Memory 
        the python process.
        """
        name, model = model_config
        tf.random.set_seed(seed)
        
        model.simulate(10)      # Warm-up
        gc.collect()

        tracemalloc.start()

        T_large = 10_000
        x, y = model.simulate(T_large)
        current_mem, peak_mem = tracemalloc.get_traced_memory()

        tracemalloc.stop()

        peak_mem_mb = peak_mem / (1024 * 1024)
        assert peak_mem_mb < 30.0, (
            f"Python memory spiked excessively to {peak_mem_mb:.2f} MB "
            f"in {name} ({model.__class__.__name__}) on seed {seed}!"
        )