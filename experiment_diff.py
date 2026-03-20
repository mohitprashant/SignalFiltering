import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel

from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter
from FilterModules.DifferentiableFilters.sinkhorn_ot import SinkhornParticleFilter
from FilterModules.DifferentiableFilters.opt_placement import OptimalPlacementParticleFilter
from FilterModules.DifferentiableFilters.condnorm_flow import CNFParticleFilter


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


def tune_sinkhorn_ot(ssm, obs, states_eval, num_particles=100):
    eps_range = [0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
    iter_range = [10, 20, 50]
    
    best_rmse = float('inf')
    best_params = {'eps': 0.1, 'iter': 20}
    
    print(f"[Tuning Sinkhorn OT -]")
    for eps in eps_range:
        for it in iter_range:
            try:
                trial_filter = SinkhornParticleFilter(num_particles=num_particles, epsilon=eps, n_iter=it, label="Trial_Sinkhorn")
                trial_filter.load_ssm(ssm)
                metrics = trial_filter.run_filter(obs, states_eval)
                
                if(np.any(np.isnan(metrics['estimates'].numpy()))):
                    continue
                    
                if(metrics['rmse'] < best_rmse):
                    best_rmse = metrics['rmse']
                    best_params = {'eps': eps, 'iter': it}
            except Exception:
                continue

    print(f"[Best Sinkhorn Params -> eps: {best_params['eps']}, n_iter: {best_params['iter']} | Tuning RMSE: {best_rmse:.4f}]")
    return best_params


def run_comparisons():
    T = 1000  
    N = 100  
    
    scenarios = [
        {"name": "SVSSM", "dim": 1, "cond": "well"},
        {"name": "SVSSM", "dim": 1, "cond": "bad"},
        {"name": "MSVSSM", "dim": 2, "cond": "well"},
        {"name": "MSVSSM", "dim": 2, "cond": "bad"},
        {"name": "MSVSSM", "dim": 5, "cond": "well"},
        {"name": "MSVSSM", "dim": 5, "cond": "bad"}
    ]
    
    results = []
    
    print("-" * 135)
    header = f"{'Model':<8} | {'Dim':<3} | {'Cond':<5} | {'Filter':<18} | {'RMSE':<8} | {'OMAT':<8} | {'Time(s)':<8} | {'Mem(MB)':<8} | {'ESS':<6} | {'Cond(Mat)':<10}"
    print(header)
    print("-" * 135)
    
    for s in scenarios:
        if(s["name"] == "SVSSM"):
            ssm = get_svssm(s["cond"])
        else:
            ssm = get_msvssm(s["dim"], s["cond"])
            
        states, obs = ssm.simulate(T)
        states_eval = tf.expand_dims(states, -1) if s["dim"] == 1 else states
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        best_sinkhorn = tune_sinkhorn_ot(ssm, obs_tensor, states_eval, num_particles=N)
            
        filters = [
            SoftResamplingParticleFilter(num_particles=N, soft_alpha=0.5, label="SoftResample"),
            SinkhornParticleFilter(num_particles=N, epsilon=best_sinkhorn['eps'], n_iter=best_sinkhorn['iter'], label="Sinkhorn-OT"),
            OptimalPlacementParticleFilter(num_particles=N, label="OptPlacement"),
            CNFParticleFilter(num_particles=N, soft_alpha=0.5, label="CNF-PF")
        ]
        
        scenario_results = {"info": s, "states": states_eval.numpy(), "obs": obs.numpy(), "metrics": {}}
        
        for f in filters:
            f.load_ssm(ssm)
            
            try:
                metrics = f.run_filter(obs_tensor, states_eval)
                
                if(np.any(np.isnan(metrics['estimates'].numpy()))):
                    raise ValueError("NaNs detected")
                
                scenario_results["metrics"][f.label] = metrics

                rmse = f"{metrics.get('rmse', 0):.4f}"
                omat = f"{metrics.get('omat', 0):.4f}" if 'omat' in metrics and metrics['omat'] > 0 else "N/A"
                time_s = f"{metrics.get('time', 0):.2f}"
                mem_mb = f"{metrics.get('mem', 0) / (1024*1024):.2f}"
                ess = f"{metrics.get('ess_avg', 0):.1f}" if 'ess_avg' in metrics else "N/A"
                
                cond_mat = "N/A"
                if('step_metrics' in metrics and metrics['step_metrics'].shape[-1] > 1):
                     cond_mat = f"{np.mean(metrics['step_metrics'][:, 1]):.2e}"
                     
                print(f"{s['name']:<8} | {s['dim']:<3} | {s['cond']:<5} | {f.label:<18} | {rmse:<8} | {omat:<8} | {time_s:<8} | {mem_mb:<8} | {ess:<6} | {cond_mat:<10}")

            except (tf.errors.InvalidArgumentError, ValueError, Exception) as e:
                scenario_results["metrics"][f.label] = None
                print(f"{s['name']:<8} | {s['dim']:<3} | {s['cond']:<5} | {f.label:<18} | {'FAILED':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<6} | {'-':<10}")
            
        print("-" * 135)
        results.append(scenario_results)
    return results


def visualize_svssm_tracking(results):
    sv_results = [r for r in results if r["info"]["name"] == "SVSSM"]
    
    if(not sv_results):
        return
        
    fig, axes = plt.subplots(len(sv_results), 1, figsize=(15, 6 * len(sv_results)))
    if len(sv_results) == 1: axes = [axes]
        
    for ax, res in zip(axes, sv_results):
        cond = res["info"]["cond"]
        states = res["states"][:, 0]
        time_steps = np.arange(len(states))
        
        ax.plot(time_steps, states, label='True Log-Volatility (X)', color='black', linewidth=2.5, zorder=5)
        
        colors = {'SoftResample': 'blue', 'Sinkhorn-OT': 'orange', 'OptPlacement': 'green', 'CNF-PF': 'red'}
        styles = {'SoftResample': '--', 'Sinkhorn-OT': '-.', 'OptPlacement': ':', 'CNF-PF': '-'}
        
        for label, color in colors.items():
            if res["metrics"].get(label) is not None:
                est = res["metrics"][label]["estimates"].numpy()[:, 0]
                ax.plot(time_steps, est, styles[label], label=f'{label} Estimate', color=color, alpha=0.8, linewidth=1.5)
        
        ax.set_title(f'1D SVSSM Differentiable Tracking Comparison ({cond.capitalize()} Conditioned)')
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