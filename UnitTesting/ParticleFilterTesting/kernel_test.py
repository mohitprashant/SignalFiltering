import pytest
import numpy as np
import tensorflow as tf

from FilterModules.ParticleFlow.kernel_flow import KernelizedParticleFlowFilter
from FilterModules.kernel_flow_base import ParticleFlowKernel

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

tf.config.set_visible_devices([], 'GPU')

DTYPE = tf.float32
SEEDS = [42, 123, 99, 90210, 6718]



class TestKernelizedParticleFlow:
    @pytest.fixture
    def lg_ssm(self):
        """2D Stable Linear Gaussian Model."""
        A = [[0.9, 0.1], [0.0, 0.9]]
        B = [[0.1, 0.0], [0.0, 0.1]]
        C = [[1.0, 0.0], [0.0, 1.0]]
        D = [[0.5, 0.0], [0.0, 0.5]]
        Sigma_init = [[1.0, 0.0], [0.0, 1.0]]
        return LinearGaussianSSM(A, B, C, D, Sigma_init)

    @pytest.fixture
    def lorenz_ssm(self):
        """Higher dimensional chaotic system (K=5)."""
        return Lorenz96Model(K=5, F=8.0, dt=0.05, process_std=0.2, obs_std=1.5)

    @pytest.fixture
    def sv_ssm(self):
        """1D Stochastic Volatility Model (Highly Non-linear Observation)."""
        return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)

    @pytest.fixture
    def msv_ssm(self):
        """3D Multivariate Stochastic Volatility Model."""
        p = 3
        phi = np.array([0.9, 0.85, 0.95], dtype=np.float32)
        sigma_eta = np.eye(p, dtype=np.float32) * 0.5 + 0.1
        sigma_eps = np.eye(p, dtype=np.float32) * 0.5 + 0.05
        beta = np.ones(p, dtype=np.float32) * 0.5
        return MultivariateStochasticVolatilityModel(p=p, phi=phi, sigma_eta=sigma_eta, beta=beta, sigma_eps=sigma_eps)


    def test_matrix_kernel_math(self):
        """
        Manually computes the matrix kernel drift for a simple 1D, 2-particle case
        and verifies the tensor operations exactly match analytical expectations.
        """
        kernel = ParticleFlowKernel(alpha=0.5, kernel_type='matrix', num_particles=2)
        particles = tf.constant([[0.0], [1.0]], dtype=DTYPE)
        grad_log_p = tf.constant([[1.0], [1.0]], dtype=DTYPE)
        B_diag = tf.constant([1.0], dtype=DTYPE)                                                # Variance
        
        # manuak calculation:
        # scale = 2.0 * alpha * B_diag = 2.0 * 0.5 * 1.0 = 1.0
        # diff = [[0, -1], [1, 0]]
        # diff_sq = [[0, 1], [1, 0]]
        # K_vals = exp(-diff_sq / 1.0) = [[1, e^-1], [e^-1, 1]]
        
        # Term 1 (Kernel smoothed gradients):
        # K * grad = [[1*1 + e^-1*1], [e^-1*1 + 1*1]] = [[1 + e^-1], [1 + e^-1]]
        
        # Term 2 (Divergence term):
        # div_K = K * (diff / 0.5) = K * (2 * diff) = [[0, -2e^-1], [2e^-1, 0]]
        # sum(div_K) = [[-2e^-1], [2e^-1]]
        
        # Total Drift = (Term 1 + Term 2) / N
        # Particle 0: (1 + e^-1 - 2e^-1) / 2 = (1 - e^-1) / 2
        # Particle 1: (1 + e^-1 + 2e^-1) / 2 = (1 + 3e^-1) / 2
        
        drift = kernel.compute_drift(particles, grad_log_p, B_diag)
        e_inv = np.exp(-1.0)
        expected_drift = np.array([[(1.0 - e_inv) / 2.0], [(1.0 + 3.0 * e_inv) / 2.0]], dtype=np.float32)
        np.testing.assert_allclose(drift.numpy(), expected_drift, atol=1e-4)


    def test_scalar_kernel_math(self):
        """Tests the scalar kernel's median distance heuristic."""
        kernel = ParticleFlowKernel(alpha=0.5, kernel_type='scalar', num_particles=3)
        particles = tf.constant([[0.0], [2.0], [4.0]], dtype=DTYPE)
        grad_log_p = tf.constant([[0.0], [0.0], [0.0]], dtype=DTYPE) # zero out gradient to isolate divergence
        B_diag = tf.constant([1.0], dtype=DTYPE)
        
        # diff_sq = [[0, 4, 16], 
        #            [4, 0, 4], 
        #            [16, 4, 0]]
        # dists (sum over features) = same as diff_sq since it's 1D.
        # scale_scalar = 4.0 / (2 * ln(2)) # 2.885
        
        drift = kernel.compute_drift(particles, grad_log_p, B_diag)
        tf.debugging.assert_all_finite(drift, "Scalar kernel produced NaNs")
        np.testing.assert_allclose(drift.numpy()[1, 0], 0.0, atol=1e-5)


    def test_edge_case_coincident_particles(self):
        """
        Edge Case: Particles collapse to the exact same state (distance = 0).
        Ensures the division by scale or log calculations don't trigger Inf/NaN.
        """
        N = 5
        kernel = ParticleFlowKernel(alpha=0.5, kernel_type='matrix', num_particles=N)
        particles = tf.zeros((N, 3), dtype=DTYPE)
        grad_log_p = tf.ones((N, 3), dtype=DTYPE)
        B_diag = tf.zeros((3,), dtype=DTYPE)
        drift = kernel.compute_drift(particles, grad_log_p, B_diag)

        tf.debugging.assert_all_finite(drift, "Coincident particles caused NaN/Inf drift.")
        np.testing.assert_allclose(drift.numpy(), grad_log_p.numpy(), atol=1e-4)               # Drift should exactly equal the gradient.


    @pytest.mark.parametrize("ssm_fixture", ["lg_ssm", "lorenz_ssm", "sv_ssm", "msv_ssm"])
    @pytest.mark.parametrize("kernel_type", ["matrix", "scalar"])
    @pytest.mark.parametrize("seed", SEEDS)
    def test_kpff_long_horizon_stability(self, ssm_fixture, kernel_type, seed, request):
        """
        Tests the compiled Kernelized Particle Flow Filter loop over 500 timesteps.
        Validates continuous tracking across all dynamic models.
        without exploding gradients or NaNs.
        """
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        ssm_model = request.getfixturevalue(ssm_fixture)
        kpff = KernelizedParticleFlowFilter(
            num_particles=50, 
            kernel_type=kernel_type, 
            alpha=0.1, 
            num_steps=10, 
            dt=0.05
        )
        kpff.load_ssm(ssm_model)
        
        T = 50

        if(ssm_fixture in ["sv_ssm", "msv_ssm"]):
            observations = tf.math.abs(tf.random.normal((T, kpff.ny), dtype=DTYPE)) + 0.1
        else:
            observations = tf.random.normal((T, kpff.ny), dtype=DTYPE)
            
        true_states = tf.random.normal((T, kpff.nx), dtype=DTYPE)
        metrics = kpff.run_filter(observations, true_states=true_states)
        
        estimates = metrics['estimates']

        assert not tf.reduce_any(tf.math.is_nan(estimates)), f"NaN discovered in Estimates for {ssm_fixture}."
        assert not tf.reduce_any(tf.math.is_inf(estimates)), f"Inf discovered in Estimates for {ssm_fixture}."
        assert 'rmse' in metrics and not np.isnan(metrics['rmse'])
        
        avg_spread = metrics.get('avg_spread', 0.0)
        assert avg_spread >= 0.0 and avg_spread < float('inf'), "Particle spread exploded or became negative."