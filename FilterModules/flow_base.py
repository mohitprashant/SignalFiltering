import tensorflow as tf


class ParticleFlow:
    """
    Implements the Exact Daum-Huang equations.
    """
    def compute_Ab_and_cond(self, P_HT, H, HPH, R, P_HT_R_inv_y, x_lin, lam, eye_nx):
        """
        Computes the flow matrix A, drift vector b, and condition number of S.
        
        Equation: dx/dlambda = A x + b
        
        Args:
            P_HT (tf.Tensor): Covariance multiplied by Jacobian transpose (P @ H^T).
            H (tf.Tensor): Observation Jacobian matrix.
            HPH (tf.Tensor): H @ P @ H^T.
            R (tf.Tensor): Observation noise covariance.
            P_HT_R_inv_y (tf.Tensor): P @ H^T @ R^-1 @ y_obs.
            x_lin (tf.Tensor): Linearization point. 
                               Shape (nx,) for EDH, or (N, nx) for LEDH.
            lam (float): Current pseudo-time lambda (0 to 1).
            eye_nx (tf.Tensor): Identity matrix of shape (nx, nx).
        
        Returns:
            tuple: (A, b, condition_number)
        """
        S_lam = lam * HPH + R                                                     # Innovation Covariance mapped by lambda: S = lam * HPH^T + R
        S_lam_inv = tf.linalg.inv(S_lam)
        s_vals = tf.linalg.svd(S_lam, compute_uv=False)
        cond_num = tf.reduce_max(s_vals) / tf.reduce_min(s_vals)

        A = -0.5 * tf.matmul(P_HT, tf.matmul(S_lam_inv, H))                        # A matrix: A = -0.5 * P H^T S^-1 H
        term1 = tf.linalg.matvec((eye_nx + lam * A), P_HT_R_inv_y)                 # b vector: b = (I + 2*lam*A) [ (I + lam*A) P H^T R^-1 y + A x_lin ]
        
        if(len(x_lin.shape) > 1):                                                   # LEDH Case: x_lin is (N, nx)
            term2 = tf.transpose(tf.matmul(A, tf.transpose(x_lin)))
            inner = term1 + term2 
            b = tf.transpose(tf.matmul((eye_nx + 2.0 * lam * A), tf.transpose(inner)))
        else:                                                                       # EDH Case: x_lin is (nx,)
            term2 = tf.linalg.matvec(A, x_lin)
            b = tf.linalg.matvec((eye_nx + 2.0 * lam * A), (term1 + term2))
        return A, b, cond_num