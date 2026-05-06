"""
ParamEstimationPipeline/online_stats.py

Online summary statistics for HMC warmup adaptation.

Classes
-------
WelfordVariance   — per-dimension running mean and variance (diagonal mass matrix)
WelfordCovariance — running mean and full covariance matrix (dense mass matrix)
"""

import numpy as np


class WelfordVariance:
    """
    Online mean and per-dimension variance via Welford's one-pass algorithm.

    Used during HMC warmup to estimate the marginal variances of each
    parameter dimension.  The diagonal mass matrix is set to the inverse
    of these variances: M_ii = 1/σ̂_i².

    Attributes
    ----------
    count : int      — number of samples seen so far
    mean  : ndarray  — running mean vector
    M2    : ndarray  — sum of squared deviations (Bessel-corrected on request)
    """

    def __init__(self, d: int) -> None:
        self.count = 0
        self.mean  = np.zeros(d, dtype=np.float32)
        self.M2    = np.zeros(d, dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        delta2      = x - self.mean
        self.M2    += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones_like(self.mean)
        return self.M2 / (self.count - 1)

    @property
    def mass_diag(self) -> np.ndarray:
        """M_ii = 1/σ̂_i²  (precision as diagonal mass)."""
        return (1.0 / np.maximum(self.variance, 1e-6)).astype(np.float32)


class WelfordCovariance:
    """
    Online mean and full covariance matrix via Welford's algorithm.

    Used during HMC warmup to estimate the posterior covariance Σ̂.
    The dense mass matrix is set to M = Σ̂⁻¹ (posterior precision),
    Cholesky-factorised for efficient kinetic-energy evaluation.

    Attributes
    ----------
    count : int         — number of samples seen so far
    mean  : ndarray     — running mean vector
    C     : ndarray     — unnormalised sum-of-outer-products accumulator
    """

    def __init__(self, d: int) -> None:
        self.count = 0
        self.mean  = np.zeros(d, dtype=np.float32)
        self.C     = np.zeros((d, d), dtype=np.float32)

    def update(self, x: np.ndarray) -> None:
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        delta2      = x - self.mean
        self.C     += np.outer(delta, delta2)

    @property
    def covariance(self) -> np.ndarray:
        if self.count < 2:
            return np.eye(len(self.mean), dtype=np.float32)
        return (self.C / (self.count - 1)).astype(np.float32)

    def mass_chol_L(self, reg: float = 1e-4) -> np.ndarray:
        """
        Lower Cholesky factor L of M = Σ̂⁻¹ + reg·I  (precision matrix).

        Convention:  M = L Lᵀ
          Sample momentum:  p = L z,  z ~ N(0,I)  →  p ~ N(0, M)
          Kinetic energy:   T = ‖L⁻¹ p‖² / 2
          Position update:  M⁻¹ p  via two triangular solves

        Parameters
        ----------
        reg : float — regularisation added to the precision before Cholesky
                      (guards against ill-conditioning when sample count is low)

        Raises
        ------
        numpy.linalg.LinAlgError if the regularised precision is not PD.
        """
        cov  = self.covariance
        prec = np.linalg.inv(cov + reg * np.eye(cov.shape[0], dtype=np.float32))
        prec = 0.5 * (prec + prec.T)                          # symmetrise
        prec = prec + reg * np.eye(prec.shape[0], dtype=np.float32)
        return np.linalg.cholesky(prec).astype(np.float32)
