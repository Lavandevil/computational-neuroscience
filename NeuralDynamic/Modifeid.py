from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

try:
    import hh_model_modified as hh
except ImportError:
    hh = None

FIG_DIR = Path(__file__).with_name('figures')
FIG_DIR.mkdir(exist_ok=True)

capacitance = 1.0
g_ca = 4.0
g_k = 8.0
g_leak = 2.0

e_ca = 120.0
e_k = -84.0
e_leak = -60.0

v1, v2, v3, v4 = -1.2, 18.0, 2.0, 30.0
v5, v6, v7, v8 = -40.0, 10.0, 0.4, 18.0

def m_gate_inf(v):
    
    return 0.5 * (1.0 + np.tanh((v - v1) / v2))

def w_gate_inf(v):
    
    return 0.5 * (1.0 + np.tanh((v - v5) / v6))

def w_tau(v):
    
    return 1.0 / np.cosh((v - v7) / (2.0 * v8))

def ml_rhs(times, states, current=0.0):
    v, w = states
    dv_dt = (current - g_ca * m_gate_inf(v) * (v - e_ca) - g_k * w * (v - e_k) - g_leak * (v - e_leak)) / capacitance
    dw_dt = (w_gate_inf(v) - w) / w_tau(v)
    return np.array([dv_dt, dw_dt])

def integrate_model(current=0.0, initial_state=(-60, 0), end_time=200, time_step=0.03):
    
    times = np.arange(0, end_time + 0.5 * time_step, time_step)
    solution = solve_ivp(
        fun=lambda tt, xx: ml_rhs(tt, xx, current),
        t_span=(0, end_time),
        y0=initial_state,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
        method='DOP853'
    )
    return solution.times, solution.y.T

def v_nullcline(v, current=0.0):
    
    return (current - g_ca * m_gate_inf(v) * (v - e_ca) - g_leak * (v - e_leak)) / (g_k * (v - e_k))

def equilibrium_residual(v, current):
    
    return ml_rhs(0, [v, w_gate_inf(v)], current)[0]

def find_equilibria(current=0.0):
    
    voltage_grid = np.linspace(-100, 80, 12001)
    y = equilibrium_residual(voltage_grid, current)
    root_values = []

    for a, b, fa, fb in zip(voltage_grid[:-1], voltage_grid[1:], y[:-1], y[1:]):
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0:
            real_parts = brentq(lambda z: equilibrium_residual(z, current), a, b)
            if not root_values or abs(real_parts - root_values[-1]) > 1e-5:
                root_values.append(real_parts)

    return [(real_parts, float(w_gate_inf(real_parts))) for real_parts in root_values]

def numerical_jacobian(states, current=0.0, epsilon=1e-5):
    
    states = np.asarray(states, float)
    jac = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2)
        d[j] = epsilon
        jac[:, j] = (ml_rhs(0, states + d, current) - ml_rhs(0, states - d, current)) / (2 * epsilon)
    return jac

def classify_point(eigenvalues):
    
    real_parts = np.real(eigenvalues)
    if np.prod(real_parts) < 0:
        return 'saddle'
    
    complex_mode = np.any(np.abs(np.imag(eigenvalues)) > 1e-8)
    if np.all(real_parts < 0):
        return 'stable spiral' if complex_mode else 'stable node'
    else:
        return 'unstable spiral' if complex_mode else 'unstable node'

def run_analysis():
    analysis_results = {}

    times, states = integrate_model(current=0.0, initial_state=(-40, 0.3), end_time=120)
    
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(states[:, 0], states[:, 1], color='crimson', lw=1.5, label='Trajectory')
    axes[0].scatter([states[0, 0]], [states[0, 1]], color='black', zorder=5, label='start')
    axes[0].set(xlabel='V (mV)', ylabel='w', title='Morris-Lecar phase portrait')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend()

    if hh is not None:
        rest = hh.settled_state()
        t2, x2 = hh.simulate(hh.step_current(40, 5, 45), rest, t_end=60)
        axes[1].plot(x2[:, 0], x2[:, 1], color='navy', lw=1.5, label='Trajectory')
        axes[1].scatter([x2[0, 0]], [x2[0, 1]], color='black', zorder=5, label='start')
        axes[1].set(xlabel='V (mV)', ylabel='n', title='Hodgkin-Huxley projection')
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend()
    else:
        axes[1].set_title('Hodgkin-Huxley projection (hh_model not found)')

    figure.tight_layout()
    figure.savefig(FIG_DIR / 'A11_phase_comparison.png', dpi=180)
    plt.close(figure)
    voltage_values = np.linspace(-80, 60, 1200)
    valid_mask = np.abs(voltage_values - e_k) > 1.0
    equilibrium_points = find_equilibria(current=0.0)

    figure, axes = plt.subplots(figsize=(7, 5.5))
    axes.plot(voltage_values, w_gate_inf(voltage_values), color='teal', lw=2, label='w-nullcline')
    axes.plot(voltage_values[valid_mask], v_nullcline(voltage_values[valid_mask]), color='orange', lw=2, label='V-nullcline')
    
    for point in equilibrium_points:
        axes.scatter(*point, color='black', s=60, zorder=5)

    axes.set(
        xlim=(-80, 60),
        ylim=(-0.1, 1.0),
        xlabel='V (mV)',
        ylabel='w',
        title='Nullclines and find_equilibria at current=0'
    )
    axes.grid(True, linestyle=':', alpha=0.6)
    axes.legend()
    figure.tight_layout()
    figure.savefig(FIG_DIR / 'A12_nullclines.png', dpi=180)
    plt.close(figure)

    equilibrium_data = []
    for point in equilibrium_points:
        jac = numerical_jacobian(point)
        eigenvalues, eigenvectors = np.linalg.eigenvalues(jac)
        equilibrium_data.append({
            'V': point[0],
            'w': point[1],
            'jac': jac.tolist(),
            'eigenvalues': [[z.real, z.imag] for z in eigenvalues],
            'eigenvectors': [[[z.real, z.imag] for z in row] for row in eigenvectors],
            'type': classify_point(eigenvalues)
        })
    analysis_results['I0_equilibria'] = equilibrium_data

    input_currents = np.linspace(-20, 300, 161)
    stability_points = []

    figure, axes = plt.subplots(figsize=(8, 5))
    for current in input_currents:
        for point in find_equilibria(float(current)):
            eigenvalues = np.linalg.eigvals(numerical_jacobian(point, float(current)))
            point_type = classify_point(eigenvalues)
            stability_points.append([float(current), point[0], point_type])
            
            stable = ('stable' in point_type) and ('unstable' not in point_type)
            plot_marker = 'o' if stable else 'states'
            color = 'forestgreen' if stable else 'firebrick'
            axes.plot(current, point[0], plot_marker, color=color, ms=3.5, alpha=0.8)

    axes.set(
        xlabel='Applied current (µA/cm²)',
        ylabel='Equilibrium V (mV)',
        title='Equilibrium branches and local stability'
    )
    axes.grid(True, linestyle=':', alpha=0.6)
    figure.tight_layout()
    figure.savefig(FIG_DIR / 'A13_bifurcation.png', dpi=180)
    plt.close(figure)
    analysis_results['stability_points'] = stability_points
    (FIG_DIR / 'ml_results.json').write_text(json.dumps(analysis_results, indent=2), encoding='utf-8')
    print(json.dumps(equilibrium_data, indent=2))

if __name__ == '__main__':
    run_analysis()
