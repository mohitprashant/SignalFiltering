import tensorflow as tf

DTYPE = tf.float32



class StaticGNM_Module(tf.keras.layers.Layer):
    """
    Trunk Network Module: Static Monotone Basis Function phi(x).
    """
    def __init__(self, nx: int, embed_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.nx = nx
        self.embed_dim = embed_dim
        self.activation = tf.nn.tanh
        
    def build(self, input_shape):
        self.W = self.add_weight(shape=(self.nx, self.embed_dim), initializer='glorot_normal', trainable=True)
        self.b = self.add_weight(shape=(self.embed_dim,), initializer='zeros', trainable=True)
        self.beta_weight = self.add_weight(shape=(1,), initializer=tf.constant_initializer(1.0), trainable=True)
        
    def call(self, x, return_jacobian=False):
        beta = tf.math.softplus(self.beta_weight) + 0.01
        z = tf.matmul(x, self.W) + self.b
        u = z * beta
        act = self.activation(u)
        out = tf.matmul(act, self.W, transpose_b=True)

        if(return_jacobian):
            d_act = (1.0 - tf.square(act)) * beta
            W_diag = tf.expand_dims(self.W, 0) * tf.expand_dims(d_act, 1)
            W_T = tf.expand_dims(tf.transpose(self.W), 0)
            jac = tf.matmul(W_diag, W_T)
            return out, jac
            
        return out