import tensorflow as tf

from FilterModules.NeuralFilter.Components.static_trunk import StaticGNM_Module

DTYPE = tf.float32


class DeepONetGradNet(tf.keras.Model):
    """
    Deep Operator Network (DeepONet) for Particle Transport.

    Approximates the optimal transport map as an operator:
    - Trunk Net: Learns continuous static basis functions phi_k(x) over the state space.
    - Branch Net: Learns coefficients c_k(y) conditioned on the observation.
    """
    def __init__(self, nx: int, num_basis: int = 16, embed_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.nx = nx
        self.trunk_basis = [StaticGNM_Module(nx, embed_dim) for _ in range(num_basis)]
        
        self.branch_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_basis)
        ])
        self.bias_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(nx)
        ])
        self.y_norm = tf.keras.layers.Normalization(axis=-1)


    def call(self, inputs, return_jacobian=False):
        x, y = inputs
        y_n = tf.clip_by_value(self.y_norm(y), -10.0, 10.0)
        weights = tf.math.softplus(self.branch_net(y_n))                    # Softplus ensures Branch weights are pos. to maintain SPD Jacobian property
        bias = self.bias_net(y_n)

        if(return_jacobian):
            N = tf.shape(x)[0]
            jac_tot = tf.eye(self.nx, batch_shape=[N], dtype=self.dtype)
            z = x
            
            for k, mod in enumerate(self.trunk_basis):
                mod_out, mod_jac = mod(x, return_jacobian=True)
                w_k = weights[:, k:k+1]
                z += w_k * mod_out
                jac_tot += tf.expand_dims(w_k, -1) * mod_jac
            return z + bias, jac_tot
            
        else:
            z = x
            for k, mod in enumerate(self.trunk_basis):
                z += weights[:, k:k+1] * mod(x, return_jacobian=False)
            return z + bias