import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel
from StateSpaceModels.lorenz_96 import Lorenz96Model

from FilterModules.ParticleFlow.edh_flow import ExactDaumHuangFilter
from FilterModules.ParticleFlow.ledh_flow import LocalizedExactDaumHuangFilter
from FilterModules.ParticleFilters.ledh_particle import PFPF_LEDHFilter
from FilterModules.ParticleFilters.homo_particle import StochPFPF



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


def get_lorenz(dim: int, condition: str) -> Lorenz96Model:
    if(condition == "well"):
        return Lorenz96Model(K=dim, F=8.0, dt=0.05, process_std=0.1, obs_std=1.0)
    else:
        return Lorenz96Model(K=dim, F=16.0, dt=0.05, process_std=0.5, obs_std=5.0)          # Increase forcing and obs noise


def run_comparisons():
    T = 1000
    N = 50
    steps = 15
    
    scenarios = [
        {"name": "SVSSM", "dim": 1, "cond": "well"},
        {"name": "SVSSM", "dim": 1, "cond": "bad"},
        {"name": "MSVSSM", "dim": 2, "cond": "well"},
        {"name": "MSVSSM", "dim": 2, "cond": "bad"},
        {"name": "Lorenz", "dim": 5, "cond": "well"},
        {"name": "Lorenz", "dim": 5, "cond": "bad"},
        {"name": "MSVSSM", "dim": 20, "cond": "well"},
        {"name": "MSVSSM", "dim": 20, "cond": "bad"}
    ]
    
    results = []
    
    print("-" * 140)
    header = f"{'Model':<8} | {'Dim':<3} | {'Cond':<5} | {'Filter':<10} | {'RMSE':<8} | {'OMAT':<8} | {'Time(s)':<8} | {'Mem(MB)':<8} | {'Flow Cond':<10} | {'Cond(P)':<10} | {'ESS':<6}"
    print(header)
    print("-" * 140)
    
    for s in scenarios:
        if(s["name"] == "SVSSM"):
            ssm = get_svssm(s["cond"])
        elif(s["name"] == "MSVSSM"):
            ssm = get_msvssm(s["dim"], s["cond"])
        else:
            ssm = get_lorenz(s["dim"], s["cond"])
            
        states, obs = ssm.simulate(T)
        states_eval = tf.expand_dims(states, -1) if s["dim"] == 1 else states

        filters = [
            ExactDaumHuangFilter(num_particles=N, num_steps=steps, label="EDH-Flow"),
            LocalizedExactDaumHuangFilter(num_particles=N, num_steps=steps, label="LEDH-Flow"),
            PFPF_LEDHFilter(num_particles=N, num_steps=steps, label="LEDH-PF"),
            StochPFPF(num_particles=N, num_steps=steps, mu=0.1, label="Homo-PF")
        ]
        
        scenario_results = {"info": s, "states": states_eval.numpy(), "obs": obs.numpy(), "metrics": {}}
        
        for f in filters:
            f.load_ssm(ssm)
            
            try:
                metrics = f.run_filter(obs, states_eval)
                if(np.any(np.isnan(metrics['estimates'].numpy()))):
                    raise ValueError("NaNs detected")
                
                scenario_results["metrics"][f.label] = metrics

                rmse = f"{metrics.get('rmse', 0):.4f}"
                omat = f"{metrics.get('omat', 0):.4f}"
                time_s = f"{metrics.get('time', 0):.2f}"
                mem_mb = f"{metrics.get('mem', 0) / (1024*1024):.2f}"
                
                flow_cond = f"{metrics.get('avg_flow_cond', 0):.2e}"
                cond_p = f"{metrics.get('avg_ekf_P_cond', 0):.2e}"
                ess = f"{metrics.get('ess_avg', 0):.1f}" if 'ess_avg' in metrics else "N/A"
                
                print(f"{s['name']:<8} | {s['dim']:<3} | {s['cond']:<5} | {f.label:<10} | {rmse:<8} | {omat:<8} | {time_s:<8} | {mem_mb:<8} | {flow_cond:<10} | {cond_p:<10} | {ess:<6}")

            except (tf.errors.InvalidArgumentError, ValueError, Exception) as e:
                scenario_results["metrics"][f.label] = None
                print(f"{s['name']:<8} | {s['dim']:<3} | {s['cond']:<5} | {f.label:<10} | {'FAILED':<8} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<10} | {'-':<10} | {'-':<6}")
            
        print("-" * 140)
        results.append(scenario_results)
    return results


def visualize_svssm_tracking(results):
    sv_results = [r for r in results if r["info"]["name"] == "SVSSM"]
    
    if(not sv_results):
        return
        
    fig, axes = plt.subplots(len(sv_results), 1, figsize=(14, 6 * len(sv_results)))
    if len(sv_results) == 1: axes = [axes]
        
    for ax, res in zip(axes, sv_results):
        cond = res["info"]["cond"]
        states = res["states"][:, 0]
        time_steps = np.arange(len(states))
        
        ax.plot(time_steps, states, label='True Log-Volatility (X)', color='black', linewidth=1.0, zorder=5)
        
        colors = {'EDH-Flow': 'blue', 'LEDH-Flow': 'orange', 'LEDH-PF': 'green', 'Homo-PF': 'red'}
        styles = {'EDH-Flow': '--', 'LEDH-Flow': '-.', 'LEDH-PF': ':', 'Homo-PF': '-'}
        
        for label, color in colors.items():
            if res["metrics"].get(label) is not None:
                est = res["metrics"][label]["estimates"].numpy()[:, 0]
                ax.plot(time_steps, est, styles[label], label=f'{label} Estimate', color=color, alpha=0.8, linewidth=1.5)
        
        ax.set_title(f'SVSSM Tracking Comparison ({cond.capitalize()} Conditioned)')
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