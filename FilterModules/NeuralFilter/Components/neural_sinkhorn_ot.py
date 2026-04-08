import tensorflow as tf

DTYPE = tf.float32


class ScalarTrunkModule(tf.keras.layers.Layer):
    """
    Scalar-valued trunk basis function phi_k(x) for Sinkhorn potential prediction.

    Unlike StaticGNM_Module (which maps R^{nx} → R^{nx} as a displacement field),
    this module maps each particle position x_i ∈ R^{nx} to a single scalar
    phi_k(x_i) ∈ R, making it suitable as a trunk basis for predicting
    per-particle Sinkhorn dual potentials.

    Architecture: phi_k(x) = v^T · tanh(beta · (x W + b))

    Parameters
    ----------
    nx       : State dimension.
    embed_dim: Width of the hidden feature space.
    """

    def __init__(self, nx: int, embed_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.nx        = nx
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(self.nx, self.embed_dim),
            initializer="glorot_normal",
            trainable=True,
            name="W",
        )
        self.b = self.add_weight(
            shape=(self.embed_dim,),
            initializer="zeros",
            trainable=True,
            name="b",
        )
        self.beta_weight = self.add_weight(
            shape=(1,),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
            name="beta",
        )
        # Projection vector: maps embed_dim → scalar
        self.v = self.add_weight(
            shape=(self.embed_dim,),
            initializer="glorot_normal",
            trainable=True,
            name="v",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Args:
            x : (N, nx) particle positions

        Returns:
            phi : (N,) scalar potential values per particle
        """
        beta = tf.math.softplus(self.beta_weight) + 0.01       # positive scale
        z    = tf.matmul(x, self.W) + self.b                   # (N, embed_dim)
        act  = tf.nn.tanh(beta * z)                            # (N, embed_dim)
        phi  = tf.linalg.matvec(act, self.v)                   # (N,)  act @ v
        return phi


class NeuralSinkhornPotentialNet(tf.keras.Model):
    """
    DeepONet that predicts Sinkhorn f-potentials as a neural operator conditioned
    on the current observation y_obs and SSM parameter vector theta_ssm.

    Architecture (DeepONet operator structure)
    ------------------------------------------
    Trunk : num_basis scalar-valued basis functions {phi_k}_{k=1}^{K}
            Each phi_k : R^{nx} → R  (ScalarTrunkModule).
    Branch: MLP over context [y_obs ∥ theta_ssm] → coefficients c ∈ R^K.
    Output: f_i = Σ_k c_k(y, θ) · phi_k(x_i)   for each particle i.

    The predicted f is then used as the initialised f-potential inside the
    neural Sinkhorn resampling step, replacing K iterations of the fixed
    dual algorithm with a single forward pass.

    Why the DeepONet structure?
    ---------------------------
    DeepONet factorises the operator as an inner product between trunk (state-
    space geometry) and branch (context-dependent coefficients).  This means
    the network acts as a proper *operator*: given a new SSM parameter θ,
    the branch re-evaluates c(y, θ) in O(K) time, while the trunk basis
    phi_k(x_i) can be cached or re-evaluated independently.  The result is
    a transport plan that adapts smoothly to HMC parameter updates.

    Parameters
    ----------
    nx        : State dimension.
    ny        : Observation dimension.
    theta_dim : Dimension of the SSM parameter vector θ.
    num_basis : Number of trunk basis functions K.
    embed_dim : Embedding dimension per trunk module.
    """

    def __init__(
        self,
        nx:        int,
        ny:        int,
        theta_dim: int,
        num_basis: int = 16,
        embed_dim: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nx        = nx
        self.ny        = ny
        self.theta_dim = theta_dim
        self.num_basis = num_basis

        # Trunk: K scalar-valued basis functions phi_k : R^{nx} → R
        self.trunk_basis = [ScalarTrunkModule(nx, embed_dim) for _ in range(num_basis)]

        # Branch: context encoder [y ∥ θ] → coefficients c ∈ R^K
        self.branch_net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_basis),   # raw; no activation needed
            ],
            name="branch_net",
        )

        # Input normalisation for the context vector [y_obs ∥ theta_ssm]
        self.context_norm = tf.keras.layers.Normalization(axis=-1)

    # ------------------------------------------------------------------

    def call(
        self,
        particles:  tf.Tensor,   # (N, nx)
        y_obs:      tf.Tensor,   # (ny,) or (1, ny)
        theta_ssm:  tf.Tensor,   # (theta_dim,)
    ) -> tf.Tensor:
        """
        Predict Sinkhorn f-potentials f_i = Σ_k c_k(y,θ) · phi_k(x_i).

        Returns
        -------
        f : (N,) predicted f-potential, one scalar per particle.
        """
        # Build context [y ∥ θ] of shape (1, ny + theta_dim)
        y_flat   = tf.reshape(y_obs,     [self.ny])
        th_flat  = tf.reshape(theta_ssm, [self.theta_dim])
        context  = tf.concat([y_flat, th_flat], axis=0)[tf.newaxis, :]        # (1, ny+theta_dim)
        context_n = tf.clip_by_value(
            self.context_norm(context), -10.0, 10.0
        )                                                                      # (1, ny+theta_dim)

        # Branch coefficients c_k ∈ R^K  →  shape (1, K)
        c = self.branch_net(context_n)

        # Trunk: evaluate each phi_k at all N particles → stack to (N, K)
        phi = tf.stack(
            [mod(particles) for mod in self.trunk_basis], axis=1
        )                                                                      # (N, K)

        # DeepONet inner product: f_i = Σ_k c_k · phi_k(x_i)
        f = tf.reduce_sum(phi * c, axis=1)                                     # (N,)
        return f
