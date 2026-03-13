import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.linear_gaussian import LinearGaussianSSM


tf.config.set_visible_devices([], 'GPU')
SEEDS = [42, 123, 99, 90210, 6718]
DTYPE = tf.float32



class TestMultivariateStochasticVolatilityModel:
    @pytest.fixture(params=[
        {   
            "p" : 2,
            "phi": [0.9, 0.8], 
            "beta": [0.5, 0.5],
            "sigma_eta": [[0.5, 0.05], [0.05, 0.5]],
            "sigma_eps": [[1.0, 0.1], [0.1, 1.0]]
        },
        {
            "p": 1,
            "phi": [0.95], 
            "beta": [1.0],
            "sigma_eta": [[0.5]],
            "sigma_eps": [[1.0]]
        }
    ])
    def msv_model(self, request):
        """Fixture providing parameterized MultivariateStochasticVolatilityModel instances."""
        c = request.param
        return MultivariateStochasticVolatilityModel(
            p=c["p"], 
            phi=c["phi"], 
            sigma_eta=c["sigma_eta"], 
            beta=c["beta"],
            sigma_eps=c["sigma_eps"]
        )


    def test_initial_dist_moments(self, msv_model):
        """Tests if the initial stationary distribution computes the correct mean and variance."""
        dist = msv_model.initial_dist()
        phi = msv_model.phi.numpy()
        sigma_eta = msv_model.sigma_eta.numpy()
        np.testing.assert_allclose(dist.mean().numpy(), np.zeros(msv_model.p))

        expected_var = np.diag(sigma_eta) / (1.0 - phi**2 + 1e-4)
        np.testing.assert_allclose(dist.variance().numpy(), expected_var, atol=1e-3)


    def test_transition_dist_mean(self, msv_model):
        """Tests the autoregressive property and mean of the state transition."""
        x_prev = tf.ones([msv_model.p], dtype=tf.float32)
        dist = msv_model.transition_dist(x_prev)
        expected_mean = msv_model.phi.numpy() * x_prev.numpy()
        np.testing.assert_allclose(dist.mean().numpy(), expected_mean, atol=1e-6)


    def test_observation_dist_clipping_edge_case(self, msv_model):
        """Tests numerical stability clipping inside the observation distribution."""
        x_curr = tf.fill([1, msv_model.p], 100.0) 
        dist = msv_model.observation_dist(x_curr)
        
        beta = msv_model.beta.numpy()
        chol_eps = msv_model.chol_eps.numpy()
        
        # The scale is clipped at x=15.0 and multiplied by the base cholesky decomposition of epsilon
        expected_scale = beta[0] * np.exp(15.0 / 2.0) * chol_eps[0, 0]
        actual_scale = dist.scale.to_dense().numpy()[0, 0, 0] 
        
        assert np.isclose(actual_scale, expected_scale, rtol=1e-4)


    @pytest.mark.parametrize("seed", SEEDS)
    def test_simulate_shapes(self, msv_model, seed):
        """Tests that the simulation loop compiles and returns correctly shaped tensors."""
        tf.random.set_seed(seed)
        T = 5
        x, y = msv_model.simulate(T)
        assert (x.shape[0], x.shape[-1]) == (T, msv_model.p)
        assert (y.shape[0], y.shape[-1]) == (T, msv_model.p)


    def generate_high_dim_msv_config(self, p):
        """Generates a high-dimensional configuration for the MSV model."""
        phi = np.random.uniform(0.8, 0.98, size=p).astype(np.float32)
        beta = np.random.uniform(0.1, 1.0, size=p).astype(np.float32)
        
        # Generate positive semi-definite p x p matrices for state noise
        noise_eta = np.random.uniform(-0.05, 0.05, size=(p, p)).astype(np.float32)
        sigma_eta = np.eye(p, dtype=np.float32) + noise_eta @ noise_eta.T
        
        # Generate positive semi-definite p x p matrices for observation noise
        noise_eps = np.random.uniform(-0.05, 0.05, size=(p, p)).astype(np.float32)
        sigma_eps = np.eye(p, dtype=np.float32) + noise_eps @ noise_eps.T
        
        return {
            "p": p, 
            "phi": phi, 
            "sigma_eta": sigma_eta, 
            "beta": beta, 
            "sigma_eps": sigma_eps
        }


    @pytest.mark.parametrize("p", [2, 10, 50])
    def test_msv_dimensional_scaling(self, p):
        """Tests if the Cholesky and MVN initializations scale to higher dimensions."""
        config = self.generate_high_dim_msv_config(p)
        model = MultivariateStochasticVolatilityModel(**config)
        
        assert model.sigma_eps.shape == (p, p)
        assert model.sigma_eta.shape == (p, p)

        init_sample = model.initial_dist().sample()
        assert init_sample.shape == (p,)


    @pytest.mark.parametrize("T", [1000, 20000])
    def test_msv_long_horizon_stability(self, T):
        """Simulates over a long horizon to ensure the observation term doesn't overflow."""
        config = self.generate_high_dim_msv_config(p=5)
        model = MultivariateStochasticVolatilityModel(**config)
        x, y = model.simulate(T)
        
        assert not tf.reduce_any(tf.math.is_nan(x))
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(x))
        assert not tf.reduce_any(tf.math.is_inf(y))



class TestStochasticVolatilityModel:
    @pytest.fixture(params=[
        (0.91, 1.0, 0.5),
        (0.99, 0.1, 1.0),
        (0.50, 2.0, 0.1)
    ])
    def sv_model(self, request):
        """Fixture providing parameterized StochasticVolatilityModel instances."""
        return StochasticVolatilityModel(*request.param)


    def test_initial_dist(self, sv_model):
        """Verifies the stationary initial variance of the 1D SV model."""
        dist = sv_model.initial_dist()
        alpha = sv_model.alpha.numpy()
        sigma = sv_model.sigma.numpy()
        expected_var = sigma**2 / (1.0 - alpha**2)
        np.testing.assert_allclose(dist.variance().numpy(), expected_var, atol=1e-4)


    def test_transition_and_observation(self, sv_model):
        """Checks the means and standard deviations of the transition and observation distributions."""
        x_prev = tf.constant(2.0)
        trans_dist = sv_model.transition_dist(x_prev)
        expected_trans_mean = sv_model.alpha.numpy() * 2.0
        assert np.isclose(trans_dist.mean().numpy(), expected_trans_mean, atol=1e-6)
        assert np.isclose(trans_dist.stddev().numpy(), sv_model.sigma.numpy(), atol=1e-6)

        x_curr = tf.constant(0.0)
        obs_dist = sv_model.observation_dist(x_curr)
        assert np.isclose(obs_dist.stddev().numpy(), sv_model.beta.numpy(), atol=1e-6)


    @pytest.mark.parametrize("seed", SEEDS)
    def test_simulate(self, sv_model, seed):
        """Tests that the 1D SV simulation yields correctly shaped outputs."""
        tf.random.set_seed(seed)
        T = 10
        x, y = sv_model.simulate(T)
        assert x.shape == (T,)
        assert y.shape == (T,)


    @pytest.mark.parametrize("alpha, sigma, beta", [
        (0.999, 0.1, 1.0),
        (-0.95, 1.0, 0.5),
        (0.01, 10.0, 0.1),
        (0.90, 0.01, 100.0)
    ])
    def test_sv_extreme_values(self, alpha, sigma, beta):
        """Tests the SV model across boundary-pushing parameters for numerical stability."""
        model = StochasticVolatilityModel(alpha=alpha, sigma=sigma, beta=beta)
        T = 15000
        x, y = model.simulate(T)
        assert not tf.reduce_any(tf.math.is_nan(x))
        assert not tf.reduce_any(tf.math.is_inf(x))
        assert not tf.reduce_any(tf.math.is_nan(y))
        assert not tf.reduce_any(tf.math.is_inf(y))



class TestLorenz96Model:
    @pytest.fixture(params=[
        {"K": 5, "F": 8.0, "dt": 0.01},
        {"K": 10, "F": 10.0, "dt": 0.05}
    ])
    def lorenz_model(self, request):
        """Fixture providing parameterized Lorenz96Model instances."""
        return Lorenz96Model(**request.param)


    def test_rk4_equilibrium_invariant(self, lorenz_model):
        """Checks that starting exactly at the forcing constant keeps the RK4 system stationary."""
        F_val = lorenz_model.F.numpy()
        K_val = lorenz_model.K
        equilibrium_state = tf.fill([K_val], F_val)
        next_state = lorenz_model.rk4_step(equilibrium_state)
        np.testing.assert_allclose(next_state.numpy(), equilibrium_state.numpy(), atol=1e-6)


    def test_get_jacobian_shape(self, lorenz_model):
        """Verifies that the automatic differentiation Jacobian has the correct shape."""
        batch_size = 3
        x = tf.random.normal((batch_size, lorenz_model.K))
        jacobian = lorenz_model.get_jacobian(x)
        assert jacobian.shape == (batch_size, lorenz_model.K, lorenz_model.K)


    @pytest.mark.parametrize("seed", SEEDS)
    def test_simulate_burn_in(self, lorenz_model, seed):
        """Tests that simulation with burn_in yields the exact requested step size."""
        tf.random.set_seed(seed)
        T = 15
        x, y = lorenz_model.simulate(T, burn_in=5)
        assert x.shape == (T, lorenz_model.K)
        assert y.shape == (T, lorenz_model.K)


    @pytest.mark.parametrize("K", [10, 40, 256])
    def test_lorenz_dimensional_scaling(self, K):
        """Tests the ODE integrator and Jacobian scaling over massive state spaces."""
        model = Lorenz96Model(K=K, F=8.0, dt=0.01)
        x = tf.random.normal((1, K))
        next_x = model.rk4_step(x)
        assert next_x.shape == (1, K)

        jacobian = model.get_jacobian(x)
        assert jacobian.shape == (1, K, K)


    @pytest.mark.parametrize("dt, F", [
        (0.001, 8.0),
        (0.05, 15.0),
        (0.01, 0.0)
    ])
    def test_lorenz_long_horizon(self, dt, F):
        """Tests continuous-time chaos over long horizons for stability and non-explosion."""
        model = Lorenz96Model(K=20, F=F, dt=dt)
        T = 10000
        x, y = model.simulate(T, burn_in=1000)
        assert not tf.reduce_any(tf.math.is_nan(x))
        assert not tf.reduce_any(tf.math.is_inf(x))

        if(F == 0.0):
            final_mean = tf.reduce_mean(tf.abs(x[-100:]))
            assert final_mean < 1.0



class TestLinearGaussianSSM:
    @pytest.fixture(params=[
        {
            "A": [[0.9, 0.1], [0.0, 0.8]], 
            "B": [[1.0, 0.0], [0.0, 1.0]],
            "C": [[1.0, 0.0]], 
            "D": [[0.5]], 
            "Sigma_init": [[1.0, 0.0], [0.0, 1.0]]
        },
        {
            "A": [[0.5, 0.1, 0.0], [0.0, 0.5, 0.1], [0.0, 0.0, 0.5]], 
            "B": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "C": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
            "D": [[0.5, 0.0], [0.0, 0.5]], 
            "Sigma_init": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        }
    ])
    def lg_model(self, request):
        """Fixture providing parameterized LinearGaussianSSM instances."""
        c = request.param
        return LinearGaussianSSM(c["A"], c["B"], c["C"], c["D"], c["Sigma_init"])


    def test_transition_dist(self, lg_model):
        """Checks mean projection and covariance calculations for the LG model transition."""
        x_prev = tf.ones([lg_model.nx], dtype=tf.float32)
        dist = lg_model.transition_dist(x_prev)
        expected_mean = lg_model.A.numpy() @ x_prev.numpy()
        expected_cov = lg_model.B.numpy() @ lg_model.B.numpy().T
        np.testing.assert_allclose(dist.mean().numpy(), expected_mean, atol=1e-6)
        np.testing.assert_allclose(dist.covariance().numpy(), expected_cov, atol=1e-6)


    def test_observation_dist(self, lg_model):
        """Checks emission matrix mappings and covariances for the observation distribution."""
        x_curr = tf.fill([lg_model.nx], 2.0)
        dist = lg_model.observation_dist(x_curr)
        expected_mean = lg_model.C.numpy() @ x_curr.numpy()
        expected_cov = lg_model.D.numpy() @ lg_model.D.numpy().T
        np.testing.assert_allclose(dist.mean().numpy(), expected_mean, atol=1e-6)
        np.testing.assert_allclose(dist.covariance().numpy(), expected_cov, atol=1e-6)


    @pytest.mark.parametrize("seed", SEEDS)
    def test_simulate(self, lg_model, seed):
        """Tests that the simulation outputs match the expected dimensional shapes."""
        tf.random.set_seed(seed)
        T = 8
        x, y = lg_model.simulate(T)
        assert x.shape == (T, lg_model.nx)
        assert y.shape == (T, lg_model.ny)


    def generate_large_lg_system(self, nx, ny):
        """Generates a highly-dimensional, stable linear gaussian system."""
        A = np.diag(np.random.uniform(-0.95, 0.95, size=nx)).astype(np.float32)
        B = np.eye(nx, dtype=np.float32) * 0.5
        C = np.random.normal(0, 1.0, size=(ny, nx)).astype(np.float32)
        D = np.eye(ny, dtype=np.float32) * 0.5
        Sigma_init = np.eye(nx, dtype=np.float32)
        return {"A": A, "B": B, "C": C, "D": D, "Sigma_init": Sigma_init}


    @pytest.mark.parametrize("nx, ny", [(10, 5), (50, 10), (200, 50)])
    def test_lg_scaling(self, nx, ny):
        """Verifies Cholesky decompositions and matrix multiplications work at large scales."""
        config = self.generate_large_lg_system(nx, ny)
        model = LinearGaussianSSM(**config)
        assert model.Q_scale.shape == (nx, nx)
        assert model.R_scale.shape == (ny, ny)

        x, y = model.simulate(10)
        assert x.shape == (10, nx)
        assert y.shape == (10, ny)


    def test_lg_long_horizon_stability(self):
        """Tests if the simulated paths collapse or explode over massive time horizons."""
        config = self.generate_large_lg_system(nx=20, ny=5)
        model = LinearGaussianSSM(**config)
        T = 50000
        x, y = model.simulate(T)
        assert not tf.reduce_any(tf.math.is_nan(x)), "Matrix collapse (NaN) in Latent State."
        assert not tf.reduce_any(tf.math.is_nan(y)), "Matrix collapse (NaN) in Observations."
        assert not tf.reduce_any(tf.math.is_inf(x)), "Exploding gradients (Inf) in Latent State."