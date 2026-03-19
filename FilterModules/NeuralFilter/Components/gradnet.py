import tensorflow as tf
from FilterModules.NeuralFilter.Components.cond_gate import ConditionalGNM

DTYPE = tf.float32


class CondGradNet(tf.keras.Model):
    """
    Conditional Gradient Network for Particle Transport.
    """
    def __init__(self, nx: int, num_modules: int = 4, embed_dim: int = 32, **kwargs):
        """
        Initializes the Conditional Gradient Network.

        Args:
            nx (int): Dimensionality of the latent state space (x).
            num_modules (int, optional): Number of ConditionalGNM layers to stack.
            embed_dim (int, optional): Dimensionality of the internal embedding 
                space for each GNM layer.
            **kwargs: Additional keyword arguments passed to tf.keras.Model.
        """
        super().__init__(**kwargs)
        self.nx = nx
        self.modules_list = [ConditionalGNM(nx, embed_dim) for _ in range(num_modules)]
        self.alpha_net = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(num_modules)])
        self.bias_net = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(nx)])
        self.y_norm = tf.keras.layers.Normalization(axis=-1)


    def call(self, inputs, return_jacobian=False):
        """
        Executes the forward pass of the transport map.

        Args:
            inputs (tuple): A tuple containing:
                - x (tf.Tensor): The input particle states of shape (N, nx).
                - y (tf.Tensor): The condition/observation vector of shape (N, ny).
            return_jacobian (bool, optional): If True, computes and returns the 
                analytical Jacobian matrix alongside the transformed particles.

        Returns:
            tf.Tensor or Tuple[tf.Tensor, tf.Tensor]:
                - If return_jacobian is False: Returns the mapped particles (z) 
                  of shape (N, nx).
                - If return_jacobian is True: Returns a tuple (z, jac_tot), where 
                  jac_tot is the Jacobian tensor of shape (N, nx, nx).
        """
        x, y = inputs
        y_n = tf.clip_by_value(self.y_norm(y), -10.0, 10.0)             # Rough fix for explode
        y_n = self.y_norm(y)
        z = x
        alphas = tf.math.softplus(self.alpha_net(y_n))
        
        if(return_jacobian):
            N = tf.shape(x)[0]
            jac_tot = tf.eye(self.nx, batch_shape=[N], dtype=self.dtype)
            
            for i, mod in enumerate(self.modules_list):
                mod_out, mod_jac = mod(x, y_n, return_jacobian=True)
                alpha_i = alphas[:, i:i+1]
                z += alpha_i * mod_out
                jac_tot += tf.expand_dims(alpha_i, -1) * mod_jac           # Residual Jacobian components addition
                
            return z + self.bias_net(y_n), jac_tot
            
        else:
            for i, mod in enumerate(self.modules_list):
                z += alphas[:, i:i+1] * mod(x, y_n)
            return z + self.bias_net(y_n)
