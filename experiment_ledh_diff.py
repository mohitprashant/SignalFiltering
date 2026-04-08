"""
experiment_ledh_vs_diff.py

Compares PFPF_LEDHFilter (hard categorical resampling) against
DiffSinkhornLEDHFilter (Sinkhorn OT barycentric projection resampling)
on:
    - SVSSM  (1D, well-conditioned)
    - MSVSSM (dim=3, well-conditioned)

DiffSinkhornLEDHFilter hyperparameters are tuned per-model via grid search
on a short prefix of the evaluation data before the full comparison run.

Tracked metrics per filter × scenario:
    RMSE, OMAT, ESS, avg_flow_cond, avg_ekf_S_cond, avg_ekf_P_cond,
    log_likelihood, wall-clock time, peak memory
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

from StateSpaceModels.stochastic_vol import StochasticVolatilityModel
from StateSpaceModels.multivar_stochastic_vol import MultivariateStochasticVolatilityModel
from FilterModules.ParticleFilters.ledh_particle import PFPF_LEDHFilter
from FilterModules.DifferentiableFilters.diff_ledh import DiffSinkhornLEDHFilter

tf.get_logger().setLevel('ERROR')

# ── Experiment constants ───────────────────────────────────────────────────────
T_EVAL    = 1000    # evaluation time-series length
T_TUNE    = 300     # shorter prefix used for hyperparameter tuning
N_EVAL    = 100     # particles for final evaluation
N_TUNE    = 50      # particles for tuning (faster)
NUM_STEPS = 20      # shared LEDH flow integration steps

# Baseline LEDH config
LEDH_STEPS  = NUM_STEPS
LEDH_THRESH = 0.5

# Tuning grids for DiffSinkhornLEDHFilter
TUNE_OT_EPSILON   = [0.05, 0.1, 0.3, 0.5, 1.0]
TUNE_OT_N_ITER    = [10, 20]
TUNE_NUM_STEPS    = [10, 20]


# ── SSM builders ──────────────────────────────────────────────────────────────

def build_svssm() -> StochasticVolatilityModel:
    return StochasticVolatilityModel(alpha=0.91, sigma=1.0, beta=0.5)


def build_msvssm(dim: int = 3) -> MultivariateStochasticVolatilityModel:
    return MultivariateStochasticVolatilityModel(
        p=dim,
        phi=np.ones(dim, dtype=np.float32) * 0.90,
        sigma_eta=np.eye(dim, dtype=np.float32) * 0.5,
        sigma_eps=np.eye(dim, dtype=np.float32) * 1.0,
        beta=np.ones(dim, dtype=np.float32) * 0.5,
    )


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_diff_sinkhorn_ledh(
    ssm,
    obs:         tf.Tensor,
    states_eval: tf.Tensor,
    n_particles: int = N_TUNE,
    t_tune:      int = T_TUNE,
) -> Tuple[Dict, List[Dict]]:
    """
    Grid search over (ot_epsilon, ot_n_iter, num_steps) for DiffSinkhornLEDHFilter.
    Uses only the first `t_tune` timesteps to keep tuning cost low.
    Criterion: RMSE (lower is better); NaN runs are skipped.

    Returns
    -------
    best_params : dict with keys ot_epsilon, ot_n_iter, num_steps
    all_records : list of dicts with all (params, rmse) combinations
    """
    obs_t    = obs[:t_tune]
    states_t = states_eval[:t_tune] if states_eval is not None else None

    best_rmse   = float('inf')
    best_params = {'ot_epsilon': 0.5, 'ot_n_iter': 20, 'num_steps': NUM_STEPS}
    all_records: List[Dict] = []

    total   = len(TUNE_OT_EPSILON) * len(TUNE_OT_N_ITER) * len(TUNE_NUM_STEPS)
    checked = 0

    print(f"  [Tuning DiffSinkhornLEDH — {total} combinations, "
          f"N={n_particles}, T={t_tune}]")

    for eps, n_it, ns in product(TUNE_OT_EPSILON, TUNE_OT_N_ITER, TUNE_NUM_STEPS):
        checked += 1
        record = {'ot_epsilon': eps, 'ot_n_iter': n_it, 'num_steps': ns, 'rmse': float('inf')}
        try:
            filt = DiffSinkhornLEDHFilter(
                num_particles=n_particles, num_steps=ns,
                ot_epsilon=eps, ot_n_iter=n_it, label="_tune_"
            )
            filt.load_ssm(ssm)
            m = filt.run_filter(obs_t, states_t)

            rmse = float(m['rmse'])
            if np.isnan(rmse) or np.isinf(rmse):
                rmse = float('inf')
            record['rmse'] = rmse

            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = {'ot_epsilon': eps, 'ot_n_iter': n_it, 'num_steps': ns}

        except Exception as exc:
            record['error'] = str(exc)

        all_records.append(record)
        if checked % 10 == 0 or checked == total:
            print(f"    {checked}/{total} done  |  best so far RMSE={best_rmse:.4f} "
                  f"@ ε={best_params['ot_epsilon']}, n_iter={best_params['ot_n_iter']}, "
                  f"steps={best_params['num_steps']}")

    print(f"  → Best: ε={best_params['ot_epsilon']}, "
          f"n_iter={best_params['ot_n_iter']}, "
          f"steps={best_params['num_steps']}  (RMSE={best_rmse:.4f})\n")
    return best_params, all_records


# ── Single filter run ─────────────────────────────────────────────────────────

def run_filter_safe(filt, obs: tf.Tensor, states_eval: Optional[tf.Tensor]) -> Optional[Dict]:
    """Runs run_filter and returns None on failure."""
    try:
        filt.load_ssm(filt._ssm_ref)          # reload so state is fresh
    except AttributeError:
        pass
    try:
        m = filt.run_filter(obs, states_eval)
        if np.any(np.isnan(m['estimates'].numpy())):
            return None
        return m
    except Exception as exc:
        print(f"    [FAILED] {filt.label}: {exc}")
        return None


# ── Full scenario runner ───────────────────────────────────────────────────────

def run_scenario(
    scenario_name: str,
    ssm,
    n_particles:   int = N_EVAL,
    best_params:   Optional[Dict] = None,
) -> Dict:
    """
    Simulates the SSM, then runs both filters for `T_EVAL` steps.
    Returns a results dict with all tracked metrics.
    """
    print(f"\n{'─'*60}")
    print(f"  Scenario: {scenario_name}  |  N={n_particles}  |  T={T_EVAL}")
    print(f"{'─'*60}")

    states, obs = ssm.simulate(T_EVAL)
    obs_t = tf.convert_to_tensor(obs, dtype=tf.float32)

    # For univariate SV, true state has shape (T,) → expand to (T, 1) for OMAT
    dim = states.shape[-1] if len(states.shape) > 1 else 1
    states_eval = (
        tf.expand_dims(states, -1) if dim == 1 else states
    )

    # ── Tune DiffSinkhorn if not already provided ──────────────────────────
    if best_params is None:
        best_params, _ = tune_diff_sinkhorn_ledh(ssm, obs_t, states_eval)

    # ── Build both filters ────────────────────────────────────────────────
    baseline = PFPF_LEDHFilter(
        num_particles=n_particles,
        num_steps=LEDH_STEPS,
        resample_threshold_ratio=LEDH_THRESH,
        label=f"LEDH (steps={LEDH_STEPS})",
    )
    diff_sinkhorn = DiffSinkhornLEDHFilter(
        num_particles=n_particles,
        num_steps=best_params['num_steps'],
        ot_epsilon=best_params['ot_epsilon'],
        ot_n_iter=best_params['ot_n_iter'],
        label=f"Diff-Sinkhorn LEDH (ε={best_params['ot_epsilon']}, "
              f"it={best_params['ot_n_iter']}, steps={best_params['num_steps']})",
    )

    scenario_results = {
        'name':        scenario_name,
        'dim':         dim,
        'states':      states_eval.numpy(),
        'obs':         obs.numpy(),
        'best_params': best_params,
        'metrics':     {},
    }

    for filt in [baseline, diff_sinkhorn]:
        filt.load_ssm(ssm)
        print(f"  Running: {filt.label}")
        try:
            m = filt.run_filter(obs_t, states_eval)
            if np.any(np.isnan(m['estimates'].numpy())):
                raise ValueError("NaN in estimates")
            scenario_results['metrics'][filt.label] = m
            _print_filter_line(scenario_name, filt.label, m)
        except Exception as exc:
            scenario_results['metrics'][filt.label] = None
            print(f"    FAILED: {exc}")

    return scenario_results


def _print_filter_line(scenario: str, label: str, m: Dict):
    rmse  = f"{m.get('rmse', 0):.4f}"
    omat  = f"{m.get('omat', 0):.4f}"
    ess   = f"{m.get('ess_avg', 0):.1f}"
    ll    = f"{m.get('log_likelihood', 0):.1f}"
    fcond = f"{m.get('avg_flow_cond', 0):.2e}"
    t_s   = f"{m.get('time', 0):.2f}"
    mem   = f"{m.get('mem', 0) / 1e6:.1f}"
    print(f"    RMSE={rmse}  OMAT={omat}  ESS={ess}  LL={ll}  "
          f"FlowCond={fcond}  t={t_s}s  mem={mem}MB")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(all_results: List[Dict]):
    col = 120
    print("\n" + "=" * col)
    print(" FULL COMPARISON SUMMARY ")
    print("=" * col)
    hdr = (f"{'Scenario':<20} | {'Filter':<50} | {'RMSE':>7} | {'OMAT':>7} | "
           f"{'ESS':>7} | {'LL':>10} | {'FlowCond':>10} | {'t(s)':>6} | {'mem(MB)':>7}")
    print(hdr)
    print("─" * col)

    for res in all_results:
        for label, m in res['metrics'].items():
            if m is None:
                print(f"{res['name']:<20} | {label:<50} | {'FAILED':>7}")
                continue
            print(
                f"{res['name']:<20} | {label:<50} | "
                f"{m.get('rmse', 0):>7.4f} | {m.get('omat', 0):>7.4f} | "
                f"{m.get('ess_avg', 0):>7.1f} | {m.get('log_likelihood', 0):>10.1f} | "
                f"{m.get('avg_flow_cond', 0):>10.2e} | "
                f"{m.get('time', 0):>6.2f} | {m.get('mem', 0)/1e6:>7.1f}"
            )
        print("─" * col)


# ── Plotting ──────────────────────────────────────────────────────────────────

_COLORS = {'LEDH': 'steelblue', 'Diff-Sinkhorn': 'crimson', 'True': 'black'}
_METRIC_SPECS = [
    ('rmse',           'RMSE ↓',           False),
    ('omat',           'OMAT ↓',           False),
    ('ess_avg',        'Mean ESS ↑',       False),
    ('log_likelihood', 'Log-Likelihood ↑', False),
    ('avg_flow_cond',  'Avg Flow Cond ↓',  True),   # log scale
    ('time',           'Wall-clock (s) ↓', False),
]


def _filter_color(label: str) -> str:
    if 'Diff' in label:
        return _COLORS['Diff-Sinkhorn']
    return _COLORS['LEDH']


def plot_tracking(all_results: List[Dict]):
    """Overlaid state estimates vs true state for each scenario (1D projection)."""
    n_rows = len(all_results)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows), squeeze=False)

    for row, res in enumerate(all_results):
        ax     = axes[row, 0]
        states = res['states']
        T      = len(states)
        x      = np.arange(T)

        # True state (first dimension if multivariate)
        true_1d = states[:, 0] if states.ndim > 1 else states
        ax.plot(x, true_1d, color=_COLORS['True'], linewidth=2.0,
                label='True state', zorder=5)

        linestyles = ['--', '-.']
        for (label, m), ls in zip(res['metrics'].items(), linestyles):
            if m is None:
                continue
            est = m['estimates'].numpy()
            est_1d = est[:, 0] if est.ndim > 1 else est
            ax.plot(x, est_1d, ls, color=_filter_color(label),
                    alpha=0.85, linewidth=1.4,
                    label=label.split('(')[0].strip())

        ax.set_title(f"{res['name']} — dim={res['dim']}", fontsize=10)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('State dim 1')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

    plt.suptitle('State Tracking: LEDH vs Diff-Sinkhorn LEDH',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ledh_tracking.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metrics_bar(all_results: List[Dict]):
    """Side-by-side bar charts for all tracked metrics across both scenarios."""
    n_metrics  = len(_METRIC_SPECS)
    n_scenarios = len(all_results)

    fig, axes = plt.subplots(
        n_metrics, n_scenarios,
        figsize=(6.5 * n_scenarios, 3.5 * n_metrics),
        squeeze=False,
    )

    for col, res in enumerate(all_results):
        labels  = [lbl for lbl, m in res['metrics'].items() if m is not None]
        colors  = [_filter_color(lbl) for lbl in labels]

        for row, (key, ylabel, log_scale) in enumerate(_METRIC_SPECS):
            ax   = axes[row, col]
            vals = [
                res['metrics'][lbl].get(key, 0)
                for lbl in labels
            ]
            short = [lbl.split('(')[0].strip() for lbl in labels]
            ax.bar(short, vals, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_title(f"{res['name']}  |  {ylabel}", fontsize=9)
            ax.set_xticks(range(len(short)))
            ax.set_xticklabels(short, rotation=15, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            if log_scale and all(v > 0 for v in vals):
                ax.set_yscale('log')

    plt.suptitle('Metric Comparison: LEDH vs Diff-Sinkhorn LEDH',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ledh_metrics_bar.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_tuning_heatmap(all_results: List[Dict], all_tune_records: Dict[str, List[Dict]]):
    """
    For each scenario: RMSE heatmap over (ot_epsilon × ot_n_iter),
    one panel per num_steps value found in the records.
    """
    for res in all_results:
        name    = res['name']
        records = all_tune_records.get(name, [])
        if not records:
            continue

        ns_values  = sorted(set(r['num_steps']  for r in records))
        eps_values = sorted(set(r['ot_epsilon'] for r in records))
        it_values  = sorted(set(r['ot_n_iter']  for r in records))

        n_panels = len(ns_values)
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                                  squeeze=False)

        for col, ns in enumerate(ns_values):
            ax      = axes[0, col]
            grid    = np.full((len(it_values), len(eps_values)), np.nan)
            sub     = [r for r in records if r['num_steps'] == ns]

            for r in sub:
                ei = eps_values.index(r['ot_epsilon'])
                ii = it_values.index(r['ot_n_iter'])
                grid[ii, ei] = r['rmse'] if np.isfinite(r['rmse']) else np.nan

            im = ax.imshow(grid, aspect='auto', cmap='RdYlGn_r',
                           origin='lower', vmin=np.nanmin(grid),
                           vmax=np.nanpercentile(grid, 95))
            ax.set_xticks(range(len(eps_values)))
            ax.set_xticklabels([str(e) for e in eps_values], fontsize=8)
            ax.set_yticks(range(len(it_values)))
            ax.set_yticklabels([str(i) for i in it_values], fontsize=8)
            ax.set_xlabel('ot_epsilon', fontsize=9)
            ax.set_ylabel('ot_n_iter',  fontsize=9)
            ax.set_title(f"{name}  |  num_steps={ns}", fontsize=9)
            plt.colorbar(im, ax=ax, label='RMSE')

            # Mark the best cell
            best_r = min(
                (r for r in sub if np.isfinite(r['rmse'])),
                key=lambda r: r['rmse'], default=None
            )
            if best_r is not None:
                ei = eps_values.index(best_r['ot_epsilon'])
                ii = it_values.index(best_r['ot_n_iter'])
                ax.add_patch(plt.Rectangle(
                    (ei - 0.5, ii - 0.5), 1, 1,
                    fill=False, edgecolor='gold', linewidth=2.5,
                ))

        fig.suptitle(f'DiffSinkhornLEDH Tuning Heatmap — {name}',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'ledh_tuning_{name.lower().replace(" ", "_")}.png',
                    dpi=150, bbox_inches='tight')
        plt.show()


def plot_ekf_stability(all_results: List[Dict]):
    """Bar charts of EKF condition numbers (S and P matrices) per scenario."""
    fig, axes = plt.subplots(
        2, len(all_results),
        figsize=(6 * len(all_results), 5),
        squeeze=False,
    )
    ekf_keys = [
        ('avg_ekf_S_cond', 'Avg EKF S cond ↓'),
        ('avg_ekf_P_cond', 'Avg EKF P cond ↓'),
    ]
    for col, res in enumerate(all_results):
        labels = [lbl for lbl, m in res['metrics'].items() if m is not None]
        colors = [_filter_color(lbl) for lbl in labels]
        short  = [lbl.split('(')[0].strip() for lbl in labels]
        for row, (key, ylabel) in enumerate(ekf_keys):
            ax   = axes[row, col]
            vals = [res['metrics'][lbl].get(key, 0) for lbl in labels]
            ax.bar(short, vals, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_title(f"{res['name']}  |  {ylabel}", fontsize=9)
            ax.set_xticks(range(len(short)))
            ax.set_xticklabels(short, rotation=15, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            if all(v > 0 for v in vals):
                ax.set_yscale('log')

    plt.suptitle('EKF Stability: Condition Numbers', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ledh_ekf_stability.png', dpi=150, bbox_inches='tight')
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tf.random.set_seed(0)
    np.random.seed(0)

    scenarios = [
        ('SVSSM  (dim=1)',  build_svssm()),
        ('MSVSSM (dim=3)',  build_msvssm(dim=3)),
    ]

    all_results:      List[Dict]        = []
    all_tune_records: Dict[str, List]   = {}

    print("=" * 70)
    print("  LEDH vs Diff-Sinkhorn LEDH — Hyperparameter Tuning Phase")
    print("=" * 70)

    tuned_params: Dict[str, Dict] = {}
    for name, ssm in scenarios:
        print(f"\nTuning for: {name}")
        states_tmp, obs_tmp = ssm.simulate(T_TUNE + 50)
        obs_t_tmp = tf.convert_to_tensor(obs_tmp[:T_TUNE], dtype=tf.float32)
        dim_tmp   = states_tmp.shape[-1] if len(states_tmp.shape) > 1 else 1
        st_tmp    = (
            tf.expand_dims(states_tmp[:T_TUNE], -1)
            if dim_tmp == 1 else states_tmp[:T_TUNE]
        )
        best, records   = tune_diff_sinkhorn_ledh(ssm, obs_t_tmp, st_tmp)
        tuned_params[name]    = best
        all_tune_records[name] = records

    print("\n" + "=" * 70)
    print("  LEDH vs Diff-Sinkhorn LEDH — Evaluation Phase")
    print("=" * 70)

    for name, ssm in scenarios:
        res = run_scenario(
            scenario_name=name,
            ssm=ssm,
            n_particles=N_EVAL,
            best_params=tuned_params[name],
        )
        all_results.append(res)

    print_summary_table(all_results)

    plot_tracking(all_results)
    plot_metrics_bar(all_results)
    plot_ekf_stability(all_results)
    plot_tuning_heatmap(all_results, all_tune_records)
