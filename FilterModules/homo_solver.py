import tensorflow as tf

DTYPE = tf.float32



# def robust_pinv(matrix, rcond=1e-6):
#     """Computes pseudo-inverse robustly."""
#     return tf.linalg.pinv(matrix, rcond=rcond)

# def robust_svd_eig(matrix):
#     """
#     Computes singular values and vectors.
#     Since the matrices in this context (covariance/information) are PSD, 
#     the SVD naturally provides eigenvalues (s) and eigenvectors (u).
#     """
#     s, u, _ = tf.linalg.svd(matrix)
#     return s, u

def robust_inv_xla(matrix, jitter=1e-6):
    """
    XLA-compatible pseudo-inverse alternative.
    Adds diagonal jitter to guarantee invertibility, then uses standard inverse.
    """
    I = tf.eye(tf.shape(matrix)[-1], dtype=matrix.dtype)
    matrix_safe = matrix + jitter * I
    return tf.linalg.inv(matrix_safe)


class HomotopySolver:
    """
    Solves for the optimal stiffness-mitigating schedule beta(lambda).
    
    This solver formulates the selection of the particle flow schedule as a 
    Boundary Value Problem (BVP). It computes an optimal flow schedule that 
    minimizes the condition number of the intermediate precision matrices, 
    thus avoiding numerical stiffness during the particle migration step.
    
    Attributes:
        nx (int): Dimensionality of the state space.
        mu (tf.Tensor): Weighting factor controlling the penalty on schedule velocity.
            Higher mu yields smoother, but potentially less optimal, schedules.
        steps (int): Number of integration steps for the lambda flow from 0 to 1.
    """
    def __init__(self, nx: int, mu: float = 0.1, steps: int = 20):
        """
        Initializes the HomotopySolver.

        Args:
            nx (int): The dimensionality of the state space.
            mu (float, optional): Penalty factor for schedule velocity. Defaults to 0.1.
            steps (int, optional): Number of discrete steps for the ODE integration. Defaults to 20.
        """
        self.nx = nx
        self.mu = tf.constant(mu, dtype=DTYPE)
        self.steps = steps


    @tf.function
    def compute_kappa_grad(self, beta, H0, dH):
        """
        Computes the gradient of the matrix condition number with respect to beta.
        
        Evaluates d(cond(M))/d(beta), where M = H0 + beta * dH is the intermediate
        information matrix. SVD is used to extract the maximum and minimum eigenvalues
        to compute a proxy for the gradient of the condition number.

        Args:
            beta (tf.Tensor): Current value of the scalar schedule beta.
            H0 (tf.Tensor): Prior information matrix, shape (nx, nx).
            dH (tf.Tensor): Information difference matrix (Hh - H0), shape (nx, nx).

        Returns:
            tf.Tensor: A scalar tensor representing the clipped gradient.
        """
        beta_c = tf.clip_by_value(beta, tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE))
        M = H0 + beta_c * dH
        is_bad = tf.reduce_any(tf.math.is_nan(M))
        
        def safe_grad():
            # e_vals, e_vecs = robust_svd_eig(M) 
            # e_vals, e_vecs, _ = tf.linalg.svd(M)
            e_vals, e_vecs = tf.linalg.eigh(M)                # Potentially more XLA friendly
            lam_max, lam_min = e_vals[0], e_vals[-1]
            v_max, v_min = e_vecs[:, 0], e_vecs[:, -1]
            
            term_min = tf.tensordot(v_min, tf.linalg.matvec(dH, v_min), axes=1)
            term_max = tf.tensordot(v_max, tf.linalg.matvec(dH, v_max), axes=1)
            safe_min = tf.maximum(lam_min, tf.constant(1e-4, dtype=DTYPE))
            grad = (safe_min * term_max - lam_max * term_min) / (tf.square(safe_min) + tf.constant(1e-12, dtype=DTYPE))
            return tf.clip_by_value(grad, tf.constant(-100.0, dtype=DTYPE), tf.constant(100.0, dtype=DTYPE))

        return tf.cond(is_bad, lambda: tf.constant(0.0, dtype=DTYPE), safe_grad)


    @tf.function
    def integrate_ode(self, v0, H0, dH):
        """
        Integrates the Euler-Lagrange ODE for the homotopy schedule using 4th-order Runge-Kutta (RK4).
        
        The ODE governs the schedule dynamics: d^2(beta)/dt^2 = mu * grad_beta(condition_number).

        Args:
            v0 (tf.Tensor): Initial velocity (slope) of the schedule, u(0).
            H0 (tf.Tensor): Prior information matrix, shape (nx, nx).
            dH (tf.Tensor): Information difference matrix (Hh - H0), shape (nx, nx).

        Returns:
            tuple:
                - final_beta (tf.Tensor): Scalar tensor of the schedule value at t=1.
                - beta_arr (tf.Tensor): Array of schedule values over all steps, shape (steps+1,).
                - vel_arr (tf.Tensor): Array of velocity values over all steps, shape (steps+1,).
        """
        dt = tf.constant(1.0 / float(self.steps), dtype=DTYPE)
        state = tf.stack([tf.constant(0.0, dtype=DTYPE), tf.cast(v0, DTYPE)])
        beta_arr = tf.TensorArray(DTYPE, size=self.steps+1).write(0, tf.constant(0.0, dtype=DTYPE))
        vel_arr = tf.TensorArray(DTYPE, size=self.steps+1).write(0, tf.cast(v0, DTYPE))
        
        for i in tf.range(self.steps):
            def ode_func(y):
                accel = self.mu * self.compute_kappa_grad(y[0], H0, dH)
                return tf.stack([y[1], accel])
            
            k1 = ode_func(state)
            k2 = ode_func(state + tf.constant(0.5, dtype=DTYPE) * dt * k1)
            k3 = ode_func(state + tf.constant(0.5, dtype=DTYPE) * dt * k2)
            k4 = ode_func(state + dt * k3)
            
            sixth = tf.constant(1.0 / 6.0, dtype=DTYPE)
            state = state + (dt * sixth) * (k1 + tf.constant(2.0, dtype=DTYPE) * k2 + tf.constant(2.0, dtype=DTYPE) * k3 + k4)
            beta_arr = beta_arr.write(i+1, state[0])
            vel_arr = vel_arr.write(i+1, state[1])
            
        return state[0], beta_arr.stack(), vel_arr.stack()


    @tf.function
    def solve(self, H0, Hh):
        """
        Solves the Boundary Value Problem (BVP) to find the optimal schedule sequence.
        
        Uses a Secant Method to iteratively adjust the initial velocity v0 
        such that the final schedule value beta(1) closely approximates 1.0, given 
        the boundary condition beta(0) = 0.0.

        Args:
            H0 (tf.Tensor): Prior information matrix, shape (nx, nx).
            Hh (tf.Tensor): Observation information matrix, shape (nx, nx).

        Returns:
            tuple:
                - betas (tf.Tensor): The optimized schedule values, shape (steps+1,).
                - vels (tf.Tensor): The rate of change of the schedule, shape (steps+1,).
        """
        dH = Hh - H0

        v_a = tf.constant(0.5, dtype=DTYPE)
        v_b = tf.constant(1.5, dtype=DTYPE)
        
        b_a, _, _ = self.integrate_ode(v_a, H0, dH)
        b_b, _, _ = self.integrate_ode(v_b, H0, dH)

        def perform_solve():
            err_a = b_a - tf.constant(1.0, dtype=DTYPE)
            err_b = b_b - tf.constant(1.0, dtype=DTYPE)
            
            init = [tf.constant(0, dtype=tf.int32), v_a, v_b, err_a, err_b]
            
            def body(i, v_p, v_c, e_p, e_c):
                denom = e_c - e_p
                eps = tf.constant(1e-6, dtype=DTYPE)
                safe_denom = tf.where(tf.abs(denom) < eps, tf.sign(denom) * eps, denom)
                v_new = v_c - e_c * (v_c - v_p) / safe_denom
                v_new = tf.clip_by_value(v_new, tf.constant(0.1, dtype=DTYPE), tf.constant(10.0, dtype=DTYPE))
                
                b_end, _, _ = self.integrate_ode(v_new, H0, dH)
                return i + 1, v_c, v_new, e_c, b_end - tf.constant(1.0, dtype=DTYPE)

            _, _, v_opt, _, _ = tf.while_loop(lambda i, vp, vc, ep, ec: i < 8, body, init)
            _, betas, vels = self.integrate_ode(v_opt, H0, dH)
            
            valid = tf.math.logical_and(
                tf.math.is_finite(betas[-1]),
                tf.abs(betas[-1] - tf.constant(1.0, dtype=DTYPE)) < tf.constant(0.2, dtype=DTYPE)
            )
            
            return tf.cond(valid,
                lambda: (tf.clip_by_value(betas, tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE)), vels),
                lambda: (tf.linspace(tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE), self.steps+1), tf.ones(self.steps+1, dtype=DTYPE))
            )

        is_bad = tf.math.logical_or(tf.math.is_nan(b_a), tf.math.is_nan(b_b))
        return tf.cond(is_bad,
            lambda: (tf.linspace(tf.constant(0.0, dtype=DTYPE), tf.constant(1.0, dtype=DTYPE), self.steps+1), tf.ones(self.steps+1, dtype=DTYPE)),
            perform_solve
        )