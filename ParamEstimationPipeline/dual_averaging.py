"""
ParamEstimationPipeline/dual_averaging.py

Nesterov dual averaging for HMC step size adaptation (Stan's formulation).

Algorithm
---------
At each warmup iteration m, given acceptance probability α_m:

    H̄_m  = (1 − 1/(m+t₀)) H̄_{m-1} + (δ_target − α_m) / (m+t₀)
    log ε_m  = µ − √m / γ · H̄_m
    log ε̄_m  = m^{−κ} log ε_m + (1 − m^{−κ}) log ε̄_{m-1}

After warmup the frozen step size is ε̄ = exp(log ε̄_m).

Parameters
----------
µ     = log(10 · ε₀)  — target level (log of 10× the initial step size)
γ     = 0.05           — free parameter controlling adaptation rate
t₀    = 10             — stabilises early updates
κ     = 0.75           — exponent that shrinks the weight of new information

References
----------
Hoffman & Gelman (2014). The No-U-Turn Sampler. JMLR 15, 1593–1623.
"""

import numpy as np


class DualAveraging:
    """
    Nesterov dual averaging for HMC step size adaptation.

    Parameters
    ----------
    init_eps   : initial step size ε₀
    target_acc : desired mean acceptance probability (default 0.65 for HMC)
    gamma, t0, kappa : algorithm hyper-parameters (Stan defaults)
    """

    def __init__(
        self,
        init_eps:      float = 0.01,
        target_acc:    float = 0.65,
        gamma:         float = 0.05,
        t0:            int   = 10,
        kappa:         float = 0.75,
        min_step_size: float = 1e-4,
    ) -> None:
        self.target        = target_acc
        self.gamma         = gamma
        self.t0            = t0
        self.kappa         = kappa
        self.min_step_size = min_step_size
        self._log_min      = np.log(min_step_size)
        self.mu            = np.log(10.0 * init_eps)
        self.eps           = init_eps
        self._log_eps_bar  = np.log(init_eps)
        self._H_bar        = 0.0
        self._m            = 0

    def update(self, alpha: float) -> float:
        """
        Ingest one acceptance probability and return the updated exploratory ε.

        Parameters
        ----------
        alpha : float — acceptance probability of the last HMC proposal (in [0,1])

        Returns
        -------
        float — new exploratory step size for the next iteration
        """
        self._m += 1
        m        = self._m
        eta      = 1.0 / (m + self.t0)
        self._H_bar   = (1.0 - eta) * self._H_bar + eta * (self.target - alpha)
        log_eps        = self.mu - np.sqrt(m) / self.gamma * self._H_bar
        log_eps        = max(log_eps, self._log_min)   # floor
        self.eps       = np.exp(log_eps)
        m_kappa        = m ** (-self.kappa)
        self._log_eps_bar = (m_kappa * log_eps
                             + (1.0 - m_kappa) * self._log_eps_bar)
        return float(self.eps)

    @property
    def final_step_size(self) -> float:
        """Smoothed (frozen) step size ε̄ to use after warmup ends."""
        return float(np.exp(self._log_eps_bar))
