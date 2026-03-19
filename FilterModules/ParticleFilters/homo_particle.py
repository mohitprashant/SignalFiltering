import tensorflow as tf
from FilterModules.ParticleFilters.ledh_particle import PFPF_LEDHFilter
from FilterModules.homo_solver import HomotopySolver
from FilterModules.homo_solver import robust_inv_xla

DTYPE = tf.float32



class StochPFPF(PFPF_LEDHFilter):
    """
    Optimal Homotopy Particle Flow Particle Filter (Stoch-PFPF).
    """
    def __init__(self, num_particles=100, num_steps=20, mu=0.1, resample_threshold_ratio=0.5, label=None):
        """
        Initializes the Stoch-PFPF Filter.

        Args:
            num_particles (int, optional): Number of particles for the filter. Defaults to 100.
            num_steps (int, optional): Number of flow integration steps. Defaults to 20.
            mu (float, optional): Penalty weighting factor for schedule optimization. Defaults to 0.1.
            resample_threshold_ratio (float, optional): Fraction of ESS below which resampling triggers. Defaults to 0.5.
            label (str, optional): Custom string label for metric tracking.
        """
        label = label or f"Stoch-PFPF (N={num_particles}, Steps={num_steps}, mu={mu})"
        super().__init__(
            num_particles=num_particles, 
            num_steps=num_steps, 
            resample_threshold_ratio=resample_threshold_ratio, 
            label=label
        )
        self.mu = mu


    def load_ssm(self, ssm_model) -> None:
        """
        Loads the State Space Model (SSM) and initializes the Homotopy solver.

        Args:
            ssm_model (SSM): The state space model containing transition and observation functions.
        """
        super().load_ssm(ssm_model)
        self.homotopy_solver = HomotopySolver(nx=self.nx, mu=self.mu, steps=self.steps)


    def particle_migration(self, particles, m_pred, P_pred, y_curr, log_weights, local=True):
        """
        Migrates particles from the prior to the posterior using an optimal homotopy flow.
        
        Calculates the prior and observation information matrices, solves for the optimal 
        schedule `beta(lambda)`, and migrates the particles using localized Exact Daum-Huang 
        (LEDH) flow equations mapped over the optimal schedule.

        Args:
            particles (tf.Tensor): Current particle states, shape (num_particles, nx).
            m_pred (tf.Tensor): Mean prediction from the auxiliary EKF, shape (nx,).
            P_pred (tf.Tensor): Covariance prediction from the auxiliary EKF, shape (nx, nx).
            y_curr (tf.Tensor): Current observation vector, shape (ny,).
            log_weights (tf.Tensor): Unnormalized log weights, shape (num_particles,).
            local (bool, optional): If True, uses local linearizations. Defaults to True.

        Returns:
            tuple:
                - particles (tf.Tensor): Migrated particle states, shape (num_particles, nx).
                - total_cond (tf.Tensor): Scalar representing the accumulated condition number.
                - log_det_J (tf.Tensor): Accumulated log-determinant of the Jacobian (via trace) 
                  for weight correction, shape (num_particles,).
        """
        H = self.ekf.get_jacobian(self.ekf.h_func, m_pred)
        # H0 = robust_pinv(P_pred)                                 
        # H0 = tf.linalg.pinv(P_pred, rcond=1e-6)
        H0 = robust_inv_xla(P_pred)

        Hh = tf.matmul(H, tf.matmul(self.R_inv, H), transpose_a=True)
        betas, beta_dots = self.homotopy_solver.solve(H0, Hh)
        
        dt = tf.constant(1.0 / float(self.steps), dtype=DTYPE)
        dH = Hh - H0
        I = tf.eye(self.nx, dtype=DTYPE)
        
        log_det_J = tf.zeros(self.num_particles, dtype=DTYPE)
        total_cond = tf.constant(0.0, dtype=DTYPE)
        
        R_inv_y = tf.linalg.matvec(self.R_inv, y_curr)
        HT_R_inv_y = tf.linalg.matvec(H, R_inv_y, transpose_a=True)

        for k in range(self.steps):
            beta = betas[k]
            beta_dot = beta_dots[k]
            
            M = H0 + beta * dH + tf.constant(1e-6, dtype=DTYPE) * I
            # M_inv = robust_pinv(M)
            # M_inv = tf.linalg.pinv(M, rcond=1e-6)  
            M_inv = robust_inv_xla(M)
            
            A = tf.constant(-0.5, dtype=DTYPE) * beta_dot * tf.matmul(M_inv, Hh)
            b_vec = tf.constant(0.5, dtype=DTYPE) * beta_dot * tf.linalg.matvec(M_inv, HT_R_inv_y)
            
            drift = tf.transpose(tf.matmul(A, particles, transpose_b=True)) + b_vec
            drift = tf.clip_by_value(drift, tf.constant(-100.0, dtype=DTYPE), tf.constant(100.0, dtype=DTYPE))
            particles = particles + dt * drift
            log_det_J += dt * tf.linalg.trace(A)                        # Liouville's theorem for trace to avoid XLA incompatible
            s = tf.linalg.svd(M, compute_uv=False)
            total_cond += tf.reduce_max(s) / (tf.reduce_min(s) + tf.constant(1e-9, dtype=DTYPE))

        return particles, total_cond, log_det_J