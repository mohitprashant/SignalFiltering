import tensorflow as tf

DTYPE = tf.float32


class ConditionalGNM(tf.keras.layers.Layer):
    """
    Conditional Gated Network Module (GNM).
    Returns both the transformed particle states and the analytical Jacobian.
    """
    def __init__(self, nx: int, embed_dim: int = 32, **kwargs):
        """
        Initializes the Conditional GNM layer.

        Args:
            nx (int): Dimensionality of the input and output state space (x).
            embed_dim (int): Dimensionality of the internal embedding.
            **kwargs: Additional keyword arguments passed to tf.keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.nx = nx
        self.embed_dim = embed_dim
        self.activation = tf.nn.tanh 


    def build(self, input_shape):
        """
        Builds the trainable components of the layer.
        
        Initializes the core projection matrix `W` and the `context` neural 
        network which maps the observation `y` to the dynamic parameters `b` and `beta`.

        Args:
            input_shape: The shape of the input tensors (unused directly, 
                as dimensions are explicitly provided in __init__).
        """
        self.W = self.add_weight(shape=(self.nx, self.embed_dim), initializer='glorot_normal', trainable=True)
        self.context = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(self.embed_dim + 1)
        ])


    def call(self, x, y, return_jacobian=False):
        """
        Executes the forward transformation and optionally computes the analytical Jacobian.

        Args:
            x (tf.Tensor): A batch of particle states of shape (N, nx).
            y (tf.Tensor): A batch of context/observation vectors of shape (N, ny).
            return_jacobian (bool): If True, computes and returns the 
                exact analytical Jacobian matrix alongside the transformed particles.

        Returns:
            tf.Tensor or Tuple[tf.Tensor, tf.Tensor]:
                - If return_jacobian is False: Returns the mapped particles (out) 
                  of shape (N, nx).
                - If return_jacobian is True: Returns a tuple (out, jac), where 
                  jac is the Jacobian tensor of shape (N, nx, nx).
        """
        params = self.context(y)
        params = tf.clip_by_value(params, -15.0, 15.0)         # Rough fix for explode
        b = params[:, :self.embed_dim]

        beta = tf.math.softplus(params[:, self.embed_dim:]) + 0.01              
        z = tf.matmul(x, self.W) + b
        u = z * beta
        act = self.activation(u)
        out = tf.matmul(act, self.W, transpose_b=True)

        if(return_jacobian):  # Manually do the jacobian comp for XLA
            d_act = (1.0 - tf.square(act)) * beta
            W_diag = tf.expand_dims(self.W, 0) * tf.expand_dims(d_act, 1)
            W_T = tf.expand_dims(tf.transpose(self.W), 0)
            jac = tf.matmul(W_diag, W_T)
            return out, jac
            
        return out