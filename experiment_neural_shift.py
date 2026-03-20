import time
import tracemalloc
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from FilterModules.DifferentiableFilters.soft_resample import SoftResamplingParticleFilter
from FilterModules.NeuralFilter.deeponet_filter import DeepONetParticleFilter


def get_svssm(condition: str) -> StochasticVolatilityModel:
    if(condition == "well"):
        return StochasticVolatilityModel(alpha=0.6, sigma=1.0, beta=0.5)
    else:
        return StochasticVolatilityModel(alpha=0.91, sigma=3.0, beta=0.1)


def run_dynamic_evaluation(filter_obj, ssm, obs_tensor, true_states, change_config=None):
    tf.config.run_functions_eagerly(True)
    tracemalloc.start()
    start_time = time.time()
    
    T = obs_tensor.shape[0]
    state = filter_obj.initialize_state()
    estimates = []
    step_metrics = []
    grad_norms = []
    final_N = 50
    
    init_alpha, init_sigma = 0.91, 1.0                # Parameters for gradual interpolation
    target_alpha, target_sigma = 0.6, 3.0
    
    try:                                   # Shift the model parameters
        for t in range(T):
            if(change_config):
                if(change_config.get('mutate_ssm') == 'gradual'):
                    progress = tf.cast(t / T, tf.float32)
                    new_alpha = init_alpha + progress * (target_alpha - init_alpha)
                    new_sigma = init_sigma + progress * (target_sigma - init_sigma)
                    
                    ssm.alpha = tf.constant(new_alpha, dtype=tf.float32)
                    ssm.sigma = tf.constant(new_sigma, dtype=tf.float32)
                    
                    if(hasattr(filter_obj, 'process_noise_dist')):
                        filter_obj.process_noise_dist = tfp.distributions.Normal(loc=tf.zeros(1, dtype=tf.float32), scale=new_sigma)

                elif(change_config.get('mutate_ssm') == 'sudden' and t == T // 2):
                    ssm.alpha = tf.constant(target_alpha, dtype=tf.float32)
                    ssm.sigma = tf.constant(target_sigma, dtype=tf.float32)
                    
                    if(hasattr(filter_obj, 'process_noise_dist')):
                        filter_obj.process_noise_dist = tfp.distributions.Normal(loc=tf.zeros(1, dtype=tf.float32), scale=target_sigma)
            
            with tf.GradientTape() as tape:                    # Got to disable XLA to execute this bit, track gradient
                tape.watch(obs_tensor)
                state_pred = filter_obj.predict(state)
                state_new, est, met = filter_obj.update(state_pred, obs_tensor[t])
                proxy_loss = tf.reduce_sum(est)
                
            grad = tape.gradient(proxy_loss, obs_tensor)
            grad_norm = tf.norm(grad).numpy() if grad is not None else 0.0
            grad_norms.append(grad_norm)
                    
            if(change_config and change_config.get('particle_jump') and t == T // 2 and hasattr(filter_obj, 'set_particle_count')):
                prev_grad = grad_norms[t-1] if t > 0 else 1e-8
                grad_diff = abs(grad_norm - prev_grad)                           # Calculate relative change in gradient
                relative_change = grad_diff / (prev_grad + 1e-8)
                
                new_N = int(50 * (1.0 + relative_change))                      # Scale particle number based on grad diff
                new_N = max(50, new_N)
                new_N = min(new_N, 2000)
                final_N = new_N
                
                filter_obj.set_particle_count(new_N)
                particles, log_weights = state_new
                indices = tf.random.categorical(log_weights[tf.newaxis, :], new_N)[0]
                new_particles = tf.gather(particles, indices)
                new_log_weights = tf.fill([new_N], -tf.math.log(float(new_N)))
                state_new = (new_particles, new_log_weights)

            state = state_new
            estimates.append(est)
            step_metrics.append(met)
            
        estimates_tensor = tf.stack(estimates)
        step_metrics_tensor = tf.stack(step_metrics)
        
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time = time.time() - start_time
        
        rmse = np.sqrt(np.mean((true_states.numpy() - estimates_tensor.numpy())**2))
        
        return {
            'rmse': rmse,
            'time': total_time,
            'mem': peak_mem,
            'estimates': estimates_tensor,
            'step_metrics': step_metrics_tensor,
            'mean_grad': np.mean(grad_norms),
            'final_N': final_N
        }
    finally:
        tf.config.run_functions_eagerly(False)


def run_comparisons():
    T = 1000  
    N = 50  
    PRETRAIN_STEPS = 500  

    scenarios = [
        {"name": "SVSSM", "label": "Baseline", "cfg": None},
        {"name": "SVSSM", "label": "Gradual Drift", "cfg": {'mutate_ssm': 'gradual'}},
        {"name": "SVSSM", "label": "Gradual + Particle Inc.", "cfg": {'mutate_ssm': 'gradual', 'particle_jump': True}},
        {"name": "SVSSM", "label": "Sudden Drift", "cfg": {'mutate_ssm': 'sudden'}},
        {"name": "SVSSM", "label": "Sudden + Particle Inc.", "cfg": {'mutate_ssm': 'sudden', 'particle_jump': True}}
    ]
    
    results = []
    
    print("-" * 145)
    header = f"{'Dynamic Event':<20} | {'Filter':<14} | {'RMSE':<8} | {'Time(s)':<8} | {'Mem(MB)':<8} | {'ESS':<6} | {'GradNorm':<8} | {'Final N':<8}"
    print(header)
    print("-" * 145)
    
    ssm_master = get_svssm("well")
    states, obs = ssm_master.simulate(T)
    states_eval = tf.expand_dims(states, -1)
    obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
    
    for s in scenarios:
        ssm = get_svssm("well")
            
        filters = [
            SoftResamplingParticleFilter(num_particles=N, soft_alpha=0.5, label="SoftResample"),
            DeepONetParticleFilter(num_particles=N, lr=0.005, num_basis=16, embed_dim=32, label="DeepONet-PF")
        ]
        
        scenario_results = {"info": s, "states": states_eval.numpy(), "obs": obs.numpy(), "metrics": {}}
        
        for f in filters:
            if(s["cfg"] and s["cfg"].get('particle_jump') and not hasattr(f, 'set_particle_count')):
                continue
                
            f.load_ssm(ssm)
            ssm.alpha = tf.constant(0.91, dtype=tf.float32)    # Reset SSM
            ssm.sigma = tf.constant(1.0, dtype=tf.float32)
            
            try:
                if(isinstance(f, DeepONetParticleFilter)):
                    tf.get_logger().setLevel('ERROR')
                    f.set_particle_count(N)
                    f.pretrain(steps=PRETRAIN_STEPS, batch_size=64)
                
                metrics = run_dynamic_evaluation(f, ssm, obs_tensor, states_eval, change_config=s["cfg"])
                
                if(np.any(np.isnan(metrics['estimates'].numpy()))):
                    raise ValueError("NaNs detected (Filter Diverged)")
                
                scenario_results["metrics"][f.label] = metrics
                
                rmse = f"{metrics.get('rmse', 0):.4f}"
                time_s = f"{metrics.get('time', 0):.2f}"
                mem_mb = f"{metrics.get('mem', 0) / (1024*1024):.2f}"
                ess = f"{np.mean(metrics['step_metrics'][:, 0].numpy()):.1f}" if 'step_metrics' in metrics else "N/A"
                grad_norm = f"{metrics.get('mean_grad', 0):.2e}"
                final_N = metrics.get('final_N', 100)
                     
                print(f"{s['label']:<20} | {f.label:<14} | {rmse:<8} | {time_s:<8} | {mem_mb:<8} | {ess:<6} | {grad_norm:<8} | {final_N:<8}")

            except Exception as e:
                scenario_results["metrics"][f.label] = None
                err_msg = str(e).split('\n')[0][:15]
                print(f"{s['label']:<20} | {f.label:<14} | {'ERR: '+err_msg:<8} | {'-':<8} | {'-':<8} | {'-':<6} | {'-':<8} | {'-':<8}")
            
        print("-" * 145)
        results.append(scenario_results)
    return results



def visualize_svssm_tracking(results):
    if(not results):
        return
        
    fig, axes = plt.subplots(len(results), 1, figsize=(16, 5 * len(results)))
    if len(results) == 1: axes = [axes]
        
    for ax, res in zip(axes, results):
        cfg = res["info"]["cfg"]
        label_event = res["info"]["label"]
        states = res["states"][:, 0]
        time_steps = np.arange(len(states))
        
        # if(cfg and cfg.get('mutate_ssm') == "gradual"):
        #     ax.axvspan(0, len(states), color='yellow', alpha=0.1, label='Gradual Environmental Drift')
            
        ax.plot(time_steps, states, label='True Log-Volatility (X)', color='black', linewidth=1.0, zorder=5)
        
        colors = {'SoftResample': 'blue', 'DeepONet-PF': 'orange'}
        styles = {'SoftResample': '--', 'DeepONet-PF': '-.'}
        
        for label, color in colors.items():
            if(res["metrics"].get(label) is not None):
                est = res["metrics"][label]["estimates"].numpy()[:, 0]
                ax.plot(time_steps, est, styles[label], label=f'{label} Estimate', color=color, alpha=0.8, linewidth=1.5)
        
        if(cfg and (cfg.get('mutate_ssm') == "sudden" or cfg.get('particle_jump'))):   # Highlight changes
            event_text = 'Sudden Drift' if cfg.get('mutate_ssm') == "sudden" else ''
            if(cfg.get('particle_jump')):
                event_text += ' + Proportional N Jump' if event_text else 'Proportional N Jump'
            ax.axvline(x=len(states)//2, color='red', linestyle=':', linewidth=2, label=f'Trigger: {event_text}')
        
        ax.set_title(f'Tracking Comparison | {label_event}')
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