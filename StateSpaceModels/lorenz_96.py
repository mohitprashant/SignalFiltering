from .ssm_base import SSM
import tensorflow as tf
import tensorflow_probability as tfp

DTYPE = tf.float32
tfd = tfp.distributions


class Lorenz96Model(SSM):
    """
    Lorenz 96 State Space Model.
    
    A continuous-time dynamical system often used as a proxy for atmospheric 
    turbulence and chaotic systems.
    
    Dynamics (ODE): 
        dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F
    
    Observation: 
        Y_k = X_k + v_k  where v_k ~ N(0, R)
    """
    def __init__(self, K=40, F=8.0, dt=0.05, process_std=0.1, obs_std=1.0, static_diff=False):
        """
        Initializes the Lorenz 96 model parameters.

        Args:
            K (int): State dimension (number of variables). Default is 40.
            F (float): Forcing constant. F=8.0 typically induces chaotic behavior.
            dt (float): Integration time step. Default is 0.05.
            process_std (float): Standard deviation of the additive process noise.
            obs_std (float): Standard deviation of the additive observation noise.
        """
        
        
        # # Noise Covariances
        # self.Q_diag = tf.fill([K], tf.constant(process_std, dtype=DTYPE)**2)
        # self.R_diag = tf.fill([K], tf.constant(obs_std, dtype=DTYPE)**2)
        # self.R = tf.linalg.diag(self.R_diag)
        # self.Q = tf.linalg.diag(self.Q_diag)

        self.K = K
        self.dtype = DTYPE

        if(static_diff):
            self.F = tf.constant(F, dtype=DTYPE)
            self.dt = tf.constant(dt, dtype=DTYPE)
            self.process_std = tf.constant(process_std, dtype=DTYPE)
            self.obs_std = tf.constant(obs_std, dtype=DTYPE)
        else:
            self.F = tf.Variable(F, dtype=DTYPE)
            self.dt = tf.Variable(dt, dtype=DTYPE)
            self.process_std = tf.Variable(process_std, dtype=DTYPE)
            self.obs_std = tf.Variable(obs_std, dtype=DTYPE)

        # self.Q_diag = tf.fill([K], self.process_std**2)
        # self.R_diag = tf.fill([K], self.obs_std**2)
        # self.R = tf.linalg.diag(self.R_diag)
        # self.Q = tf.linalg.diag(self.Q_diag)

    @property
    def Q_diag(self):
        return tf.fill([self.K], self.process_std**2)

    @property
    def R_diag(self):
        return tf.fill([self.K], self.obs_std**2)

    @property
    def R(self):
        return tf.linalg.diag(self.R_diag)

    @property
    def Q(self):
        return tf.linalg.diag(self.Q_diag)



    def rk4_step(self, x):
        """
        Performs a single Runge-Kutta 4 (RK4) integration step for the deterministic dynamics.
        
        Args:
            x (tf.Tensor): Current state tensor of shape (..., K).
            
        Returns:
            tf.Tensor: Next state tensor after time dt, shape (..., K).
        """
        def ode(state):
            x_p1 = tf.roll(state, shift=-1, axis=-1)
            x_m1 = tf.roll(state, shift=1, axis=-1)
            x_m2 = tf.roll(state, shift=2, axis=-1)
            return (x_p1 - x_m2) * x_m1 - state + self.F

        k1 = ode(x)
        k2 = ode(x + 0.5 * self.dt * k1)
        k3 = ode(x + 0.5 * self.dt * k2)
        k4 = ode(x + self.dt * k3)
        return x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


    def transition(self, x):
        """
        Applies the stochastic state transition: x_k = f(x_{k-1}) + q_k.
        
        Args:
            x (tf.Tensor): State at time k-1.
            
        Returns:
            tf.Tensor: State at time k with process noise added.
        """
        mu = self.rk4_step(x)
        return mu + tf.random.normal(tf.shape(mu), stddev=tf.sqrt(self.Q_diag[0]))


    def observation(self, x):
        """
        Generates a noisy observation from the state: y_k = x_k + r_k.
        
        Args:
            x (tf.Tensor): Current state vector.
            
        Returns:
            tf.Tensor: Noisy observation vector.
        """
        return x + tf.random.normal(tf.shape(x), stddev=tf.sqrt(self.R_diag[0]))


    def get_jacobian(self, x):
        """
        Computes the Jacobian F = df/dx using Automatic Differentiation.
        
        Args:
            x (tf.Tensor): Input state tensor.
            
        Returns:
            tf.Tensor: Jacobian matrix of the RK4 transition function at x.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.rk4_step(x)
        return tape.batch_jacobian(y, x)
    

    def initial_dist(self):
        """
        Returns the initial distribution for the state X_0.
        
        Initialized near the equilibrium forcing F with small perturbations to 
        ensure divergence into chaotic behavior.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Initial distribution.
        """
        loc = tf.fill([self.K], self.F)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.ones(self.K)*0.1)        


    def transition_dist(self, x_prev):
        """
        Returns the probabilistic state transition: p(x_t | x_{t-1}).
        
        Modeled as: x_t = RK4(x_{t-1}) + Gaussian Noise

        Args:
            x_prev (tf.Tensor): State at time t-1.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Transition distribution.
        """
        loc = self.rk4_step(x_prev)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.fill([self.K], self.Q_diag[0]))
    
    
    def observation_dist(self, x_curr):
        """
        Returns the observation distribution: p(y_t | x_t).
        
        Modeled as fully observed state with Gaussian noise: y_t = x_t + Noise.

        Args:
            x_curr (tf.Tensor): State at time t.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Observation distribution.
        """
        return tfd.MultivariateNormalDiag(loc=x_curr, scale_diag=tf.fill([self.K], self.R_diag[0]))


    @tf.function(jit_compile=True)
    def simulate(self, T, burn_in=1000):
        """
        Simulates the Lorenz 96 system for T time steps.
        
        Includes a burn-in period to allow the system to settle onto its chaotic attractor
        before recording begins.

        Args:
            T (int): Number of time steps to record.
            burn_in (int): Number of steps to discard initially. Default 1000.

        Returns:
            tuple: (states, observations)
                - states (tf.Tensor): Recorded states of shape (T, K).
                - observations (tf.Tensor): Recorded observations of shape (T, K).
        """
        current_state = self.initial_dist().sample()

        print(f"Burning in for {burn_in} steps...")
        for _ in tf.range(burn_in):
            current_state = self.rk4_step(current_state)

        return SSM.simulate(self, T)


    def filter_components(self):
        """Returns the components needed by non-linear filters."""
        return {
            "nx": self.K,
            "ny": self.K,
            "f_func": lambda x: self.rk4_step(x),
            "h_func": lambda x: x,
            "Q": self.Q,
            "R": self.R,
            "P_init": tf.eye(self.K, dtype=self.dtype) * (0.1**2),
            "preprocess_obs": lambda y: tf.reshape(y, [-1, self.K])
        }
    

    def dynamic_filter_components(self):
        """
        Returns components as functions of dynamic tensor tuple `theta` for XLA.
        Expected theta structure: (F, process_std, obs_std)
        """
        def rk4_dynamic(x, F):
            """Dynamic RK4 step that accepts varying F values"""
            def ode(state):
                x_p1 = tf.roll(state, shift=-1, axis=-1)
                x_m1 = tf.roll(state, shift=1, axis=-1)
                x_m2 = tf.roll(state, shift=2, axis=-1)
                return (x_p1 - x_m2) * x_m1 - state + F

            k1 = ode(x)
            k2 = ode(x + 0.5 * self.dt * k1)
            k3 = ode(x + 0.5 * self.dt * k2)
            k4 = ode(x + self.dt * k3)
            return x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return {
            "nx": self.K,
            "ny": self.K,
            
            # f(x) dynamically computes the RK4 step using the current F
            "f_func": lambda x, theta: rk4_dynamic(x, theta[0]),
            
            # h(x) = x
            "h_func": lambda x, theta: x,
            
            # Q = diag(process_std^2)
            "Q_func": lambda theta: tf.linalg.diag(tf.fill([self.K], theta[1]**2)),
            
            # R = diag(obs_std^2)
            "R_func": lambda theta: tf.linalg.diag(tf.fill([self.K], theta[2]**2)),
            
            # P_init remains a small constant diagonal matrix 
            "P_init_func": lambda theta: tf.eye(self.K, dtype=self.dtype) * (0.1**2),
            
            "preprocess_obs": lambda y: tf.reshape(y, [-1, self.K])
        }