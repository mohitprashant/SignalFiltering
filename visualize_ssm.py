import matplotlib.pyplot as plt
import numpy as np

from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel
from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.lorenz_96 import Lorenz96Model
from StateSpaceModels.linear_gaussian import LinearGaussianSSM


def demo_svssm():
    alpha = 0.91
    sigma = 1.0
    beta = 0.5
    T = 100    # Timesteps

    sv_model = StochasticVolatilityModel(alpha, sigma, beta)
    
    print(f"Simulating Stochastic Volatility Model (T={T})...")
    states, observations = sv_model.simulate(T)
    states_np = states.numpy()
    obs_np = observations.numpy()
    time_steps = np.arange(T)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, states_np, color='blue', linewidth=1, label='Volatility (State X)')
    plt.scatter(time_steps, obs_np, color='red', s=10, marker='*', label='Observations (Y)')
    plt.title(f'Simulated Volatility Sequence (alpha={alpha}, sigma={sigma}, beta={beta})')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def demo_lgssm():
    dt = 0.1
    T = 100
    A = [[1.0, dt], [0.0, 1.0]]                 # Transition Matrix A: Constant velocity model
    B = [[0.5 * dt**2, 0.0], [dt, 0.0]]         # Noise Mapping B: Random acceleration
    C = [[1.0, 0.0]]                            # Observation Matrix C: We only observe position
    D = [[0.5]]                                 # Observation Noise Mapping D: Scalar noise
    Sigma_init = [[1.0, 0.0], [0.0, 1.0]]       # Initial Covariance
                  
    ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)

    print(f"Simulating Linear Gaussian Model for T={T} steps...")
    states, observations = ssm.simulate(T)
    states_np = states.numpy()
    obs_np = observations.numpy()
    time_steps = np.arange(T)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(time_steps, states_np[:, 0], label='True State (Position)', linewidth=2)
    plt.scatter(time_steps, obs_np[:, 0], color='red', s=15, alpha=0.6, label='Observations')
    plt.title('State X[0] vs Observations Y')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(time_steps, states_np[:, 1], color='orange', label='True State (Velocity)')
    plt.title('Latent State X[1] (Velocity)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



def demo_lorenz():
    K = 40          # L96 dimension
    F = 8.0         # chaotic forcing
    dt = 0.05       # Integration step
    T = 500         # Steps

    l96 = Lorenz96Model(K=K, F=F, dt=dt, process_std=0.01, obs_std=1.0)

    print(f"Simulating Lorenz Model for T={T} steps...")
    states, observations = l96.simulate(T, burn_in=500)
    states_np = states.numpy()
    obs_np = observations.numpy()

    plt.figure(figsize=(12, 6))
    plt.imshow(states_np.T, aspect='auto', cmap='RdBu_r', origin='lower',
               extent=[0, T, 0, K], vmin=0, vmax=12)
    plt.colorbar(label='State Value ($X_k$)')
    plt.title(f'Lorenz 96 Hovmoller Diagram (F={F}, K={K})')
    plt.xlabel('Time Step')
    plt.ylabel('State Index (k)')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states_np[:, 0], states_np[:, 1], states_np[:, 2], 
            lw=0.8, color='teal', label='True Trajectory')
    ax.scatter(obs_np[:, 0], obs_np[:, 1], obs_np[:, 2], 
               color='red', s=10, alpha=0.4, label='Observations', marker='o')
    ax.set_title("Phase Space Projection (X0, X1, X2)\nTrajectory vs Noisy Observations")
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("X2")
    ax.legend()
    plt.tight_layout()
    plt.show()



def demo_msvssm():
    phi = [0.95, 0.90]                # High persistence in log-volatility
    beta = [1.0, 0.8]                 # Observation scaling
    sigma_eta = [[1.0, 0.2], [0.2, 1.0]]
    sigma_eps = [[0.5, 0.1], [0.1, 0.5]]
    T = 200

    msv_model = MultivariateStochasticVolatilityModel(p=2, phi=phi, sigma_eta=sigma_eta, sigma_eps=sigma_eps, beta=beta)
    
    print(f"Simulating Multivariate Stochastic Volatility Model for T={T} steps...")
    states, observations = msv_model.simulate(T)
    states_np = states.numpy()
    obs_np = observations.numpy()
    time_steps = np.arange(T)
    p = states_np.shape[1]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    colors = ['blue', 'orange']
    for i in range(p):
        plt.plot(time_steps, states_np[:, i], color=colors[i % len(colors)], 
                 linewidth=1.5, label=f'Volatility Dim {i+1} (State X)')
    plt.title(f'Latent Log-Volatility Sequences (phi={phi})')
    plt.xlabel('Time Step')
    plt.ylabel('Log-Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    for i in range(p):
        # print(len(time_steps), len(obs_np[:, i]))
        plt.scatter(time_steps, obs_np[:, i], color=colors[i % len(colors)], 
                    s=15, marker='*', alpha=0.7, label=f'Returns Dim {i+1} (Obs Y)')
        plt.plot(time_steps, obs_np[:, i], color=colors[i % len(colors)], 
                 alpha=0.2, linewidth=0.8)
        
    plt.title(f'Simulated Returns Sequences (beta={beta})')
    plt.xlabel('Time Step')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # demo_lgssm()
    # demo_svssm()
    # demo_lorenz()
    demo_msvssm()