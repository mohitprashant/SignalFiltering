import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from StateSpaceModels.linear_gaussian import LinearGaussianSSM
from FilterModules.KalmanFilters.kalman import KalmanFilter


def create_lgssm(dim: int, condition_factor: float) -> LinearGaussianSSM:
    A = np.eye(dim) * 0.95
    C = np.eye(dim)
    
    eigenvalues = np.linspace(1.0, condition_factor, dim)
    
    def random_orthogonal(n):
        H = np.random.randn(n, n)
        Q, _ = np.linalg.qr(H)
        return Q

    U_Q = random_orthogonal(dim)
    B = U_Q @ np.diag(np.sqrt(eigenvalues))
    
    U_R = random_orthogonal(dim)
    D = U_R @ np.diag(np.sqrt(eigenvalues))
    
    Sigma_init = np.eye(dim, dtype=np.float32) * 1.0
    return LinearGaussianSSM(A, B, C, D, Sigma_init)


def analyze_filters():
    dimensions = [2, 5]
    condition_factors = [1e1, 1e5] # second one is badly conditioned
    T = 1000
    
    results = []
    
    print("-" * 80)
    print(f"{'Dim':<5} | {'Cond Factor':<12} | {'Filter':<10} | {'RMSE':<10} | {'Time (s)':<10} | {'Mean Cond(P)':<12}")
    print("-" * 80)
    
    for dim in dimensions:
        for cond in condition_factors:
            ssm = create_lgssm(dim, cond)
            states, obs = ssm.simulate(T)

            kf_standard = KalmanFilter(label=f"Standard_D{dim}_C{cond}", joseph_form=False)
            kf_joseph = KalmanFilter(label=f"Joseph_D{dim}_C{cond}", joseph_form=True)
            
            kf_standard.load_ssm(ssm)
            kf_joseph.load_ssm(ssm)

            metrics_std = kf_standard.run_filter(obs, states)
            metrics_jos = kf_joseph.run_filter(obs, states)
            
            mean_cond_P_std = np.mean(metrics_std['step_metrics'][:, 1].numpy())
            mean_cond_P_jos = np.mean(metrics_jos['step_metrics'][:, 1].numpy())
            
            print(f"{dim:<5} | {cond:<12.1e} | {'Standard':<10} | {metrics_std['rmse']:<10.4f} | {metrics_std['time']:<10.4f} | {mean_cond_P_std:<12.4e}")
            print(f"{dim:<5} | {cond:<12.1e} | {'Joseph':<10} | {metrics_jos['rmse']:<10.4f} | {metrics_jos['time']:<10.4f} | {mean_cond_P_jos:<12.4e}")
            print("-" * 80)
            
            results.append({
                'dim': dim,
                'cond': cond,
                'states': states.numpy(),
                'obs': obs.numpy(),
                'metrics_std': metrics_std,
                'metrics_jos': metrics_jos
            })
            
    return results


def visualize_tracking(results):
    res_target = next(r for r in results if r['dim'] == 2 and r['cond'] == 1e5)    # check the badly conditioned lgssm
    
    states = res_target['states']
    obs = res_target['obs']
    est_std = res_target['metrics_std']['estimates'].numpy()
    est_jos = res_target['metrics_jos']['estimates'].numpy()

    cond_P_std = res_target['metrics_std']['step_metrics'][:, 1].numpy()
    cond_P_jos = res_target['metrics_jos']['step_metrics'][:, 1].numpy()
    
    time_steps = np.arange(len(states))
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, states[:, 0], label='True State (Dim 0)', linewidth=2, color='black')
    plt.scatter(time_steps, obs[:, 0], color='red', s=15, alpha=0.5, label='Observations', zorder=5)
    plt.plot(time_steps, est_std[:, 0], '--', label='Standard KF Estimate', alpha=0.9, color='blue')
    plt.plot(time_steps, est_jos[:, 0], ':', label='Joseph KF Estimate', alpha=0.9, color='orange', linewidth=2)
    plt.title(f'State Tracking (Dim=2, High Conditioning)')
    plt.xlabel('Time Step')
    plt.ylabel('State Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, cond_P_std, label='Standard Form Cond(P)', color='blue')
    plt.plot(time_steps, cond_P_jos, label='Joseph Form Cond(P)', color='orange')
    plt.yscale('log')
    plt.title('Covariance Matrix P Condition Number (Log Scale)')
    plt.xlabel('Time Step')
    plt.ylabel('Condition Number')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    experiment_results = analyze_filters()
    visualize_tracking(experiment_results)