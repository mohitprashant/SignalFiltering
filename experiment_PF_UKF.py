import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel
from FilterModules.KalmanFilters.extend_kalman import ExtendedKalmanFilter
from FilterModules.KalmanFilters.unscent_kalman import UnscentedKalmanFilter
from FilterModules.ParticleFilters.particle import ParticleFilter


def condition_cov(dim: int, condition_factor: float) -> np.ndarray:
    eigenvalues = np.linspace(1.0, condition_factor, dim)
    H = np.random.randn(dim, dim)
    Q, _ = np.linalg.qr(H)
    Cov = Q @ np.diag(eigenvalues) @ Q.T
    return (Cov + Cov.T) / 2.0


def get_svssm(condition: str) -> StochasticVolatilityModel:
    if(condition == "well"):
        return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)
    else:
        return StochasticVolatilityModel(alpha=0.99, sigma=2.0, beta=0.1)


def get_msvssm(dim: int, condition: str) -> MultivariateStochasticVolatilityModel:
    beta = np.ones(dim, dtype=np.float32) * 0.5
    if(condition == "well"):
        phi = np.ones(dim, dtype=np.float32) * 0.90
        sigma_eta = np.eye(dim, dtype=np.float32) * 0.5
        sigma_eps = np.eye(dim, dtype=np.float32) * 1.0
    else:
        phi = np.ones(dim, dtype=np.float32) * 0.98
        sigma_eta = condition_cov(dim, condition_factor=100.0).astype(np.float32)
        sigma_eps = condition_cov(dim, condition_factor=50.0).astype(np.float32)
    return MultivariateStochasticVolatilityModel(p=dim, phi=phi, sigma_eta=sigma_eta, sigma_eps=sigma_eps, beta=beta)


def run_comparisons():
    T = 1000
    scenarios = [
        {"name": "SVSSM", "dim": 1, "cond": "well"},
        {"name": "SVSSM", "dim": 1, "cond": "bad"},
        {"name": "MSVSSM", "dim": 2, "cond": "well"},
        {"name": "MSVSSM", "dim": 2, "cond": "bad"},
        {"name": "MSVSSM", "dim": 5, "cond": "well"},
        {"name": "MSVSSM", "dim": 5, "cond": "bad"}
    ]
    
    results = []
    
    print("-" * 120)
    print(f"{'Model':<10} | {'Dim':<5} | {'Cond':<6} | {'Filter':<6} | {'RMSE':<10} | {'Time (s)':<10} | {'Mem (MB)':<10} | {'Mean Cond(P)':<12} | {'Mean ESS':<10}")
    print("-" * 120)
    
    for s in scenarios:
        if(s["name"] == "SVSSM"):
            ssm = get_svssm(s["cond"])
        else:
            ssm = get_msvssm(s["dim"], s["cond"])

        states, obs = ssm.simulate(T)

        if(s["dim"] == 1):
            states_eval = tf.expand_dims(states, -1)
        else:
            states_eval = states
            
        filters = [
            ExtendedKalmanFilter(label="EKF"),
            UnscentedKalmanFilter(label="UKF"),
            ParticleFilter(num_particles=500, label="PF_500"),
            ParticleFilter(num_particles=100, label="PF_100"),
            ParticleFilter(num_particles=10, label="PF_10"),
        ]
        
        scenario_results = {"info": s, "states": states_eval.numpy(), "obs": obs.numpy(), "metrics": {}}

        for f in filters:
            f.load_ssm(ssm)
            metrics = f.run_filter(obs, states_eval)
            scenario_results["metrics"][f.label] = metrics
            cond_p_str = "N/A"
            ess_str = "N/A"
            
            if(f.label in ["EKF", "UKF"]):
                mean_cond_P = np.mean(metrics['step_metrics'][:, 1].numpy())
                cond_p_str = f"{mean_cond_P:.4e}"
            elif('PF' in f.label):
                mean_ess = metrics.get('ess_avg', 0.0)
                ess_str = f"{mean_ess:.1f}"
            
            peak_mem_mb = metrics.get('mem', 0) / (1024 * 1024)

            print(f"{s['name']:<10} | {s['dim']:<5} | {s['cond']:<6} | {f.label:<6} | {metrics['rmse']:<10.4f} | {metrics['time']:<10.4f} | {peak_mem_mb:<10.4f} | {cond_p_str:<12} | {ess_str:<10}")
        print("-" * 120)
        results.append(scenario_results)
        
    return results


def visualize_svssm_tracking(results):
    sv_results = [r for r in results if r["info"]["name"] == "SVSSM"]
    fig, axes = plt.subplots(len(sv_results), 1, figsize=(14, 5 * len(sv_results)))

    if(len(sv_results) == 1):
        axes = [axes]
        
    time_steps = np.arange(len(sv_results[0]["states"]))
    
    for ax, res in zip(axes, sv_results):
        cond = res["info"]["cond"]
        states = res["states"][:, 0]
        
        ekf_est = res["metrics"]["EKF"]["estimates"].numpy()[:, 0]
        ukf_est = res["metrics"]["UKF"]["estimates"].numpy()[:, 0]
        pf_est_1 = res["metrics"]["PF_10"]["estimates"].numpy()[:, 0]
        pf_est_2 = res["metrics"]["PF_500"]["estimates"].numpy()[:, 0]
        
        ax.plot(time_steps, states, label='True Log-Volatility (X)', color='black', linewidth=1.0, zorder=5)
        ax.plot(time_steps, ekf_est, '--', label='EKF Estimate', color='blue', alpha=0.8, linewidth=1.0)
        ax.plot(time_steps, ukf_est, '-.', label='UKF Estimate', color='orange', alpha=0.8, linewidth=1.0)
        ax.plot(time_steps, pf_est_1, ':', label='PF Estimate (N=10)', color='green', alpha=0.9, linewidth=1.0)
        ax.plot(time_steps, pf_est_2, ':', label='PF Estimate (N=500)', color='red', alpha=0.9, linewidth=1.0)
        
        ax.set_title(f'SVSSM Latent State Tracking ({cond.capitalize()} Conditioned)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Log-Volatility')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    experiment_results = run_comparisons()
    visualize_svssm_tracking(experiment_results)