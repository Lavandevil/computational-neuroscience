import json
from dataclasses import dataclass, replace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

FIG_DIR = Path(__file__).with_name("figures")
FIG_DIR.mkdir(exist_ok=True)

@dataclass(frozen=True)
class HHParams:
    capacitance: float = 1.0       # C (uF/cm^2)
    g_na: float = 120.0          # gNa (mS/cm^2)
    g_k: float = 36.0            # gK (mS/cm^2)
    g_leak: float = 0.3          # gL (mS/cm^2)
    e_na: float = 50.0              # ENa (mV)
    e_k: float = -77.0              # EK (mV)
    e_leak: float = -54.4           # EL (mV)

PARAMS = HHParams()
INITIAL_STATE = np.array([-65.0, 0.32, 0.05, 0.60], dtype=float)  # [V, n, m, h]

def safe_rate(dv, scale, factor, sign):
    """Evaluate a rate expression safely near zero."""
    scalar_input = np.ndim(dv) == 0
    delta = np.asarray(dv, dtype=float)
    
    exp_arg = np.clip(sign * delta / scale, -700.0, 700.0)
    denominator = -np.expm1(exp_arg)
    limit_value = -factor * scale / sign
    
    rates = np.empty_like(delta)
    near_zero = np.abs(delta) < 1e-8
    rates[near_zero] = limit_value
    np.divide(factor * delta, denominator, out=rates, where=~near_zero)
    
    return float(rates) if scalar_input else rates

def alpha_n(vm): return safe_rate(vm - 25.0, 9.0, 0.02, -1.0)
def beta_n(vm):  return safe_rate(vm - 25.0, 9.0, -0.002, +1.0)
def alpha_m(vm): return safe_rate(vm + 35.0, 9.0, 0.182, -1.0)
def beta_m(vm):  return safe_rate(vm + 35.0, 9.0, -0.124, +1.0)
def alpha_h(vm): return 0.25 * np.exp(np.clip(-(vm + 90.0) / 12.0, -700.0, 700.0))
def beta_h(vm):  return 0.25 * np.exp(np.clip((vm + 62.0) / 6.0 - (vm + 90.0) / 12.0, -700.0, 700.0))

def step_current(amplitude, start=5.0, stop=45.0):
    """Create a square current pulse."""
    return lambda time_now: float(amplitude if start <= time_now <= stop else 0.0)

def hh_rhs(time, state, input_current, p=PARAMS):
    """Hodgkin-Huxley differential equations."""
    v, n, m, h = state
    
    i_na = p.g_na * (m ** 3) * h * (v - p.e_na)
    i_k = p.g_k * (n ** 4) * (v - p.e_k)
    i_leak = p.g_leak * (v - p.e_leak)
    
    v_dot = (input_current(time) - i_na - i_k - i_leak) / p.capacitance
    n_dot = alpha_n(v) * (1.0 - n) - beta_n(v) * n
    m_dot = alpha_m(v) * (1.0 - m) - beta_m(v) * m
    h_dot = alpha_h(v) * (1.0 - h) - beta_h(v) * h
    
    return np.array([v_dot, n_dot, m_dot, h_dot])

def simulate(input_current, x0=INITIAL_STATE, t_end=100.0, dt=0.02, p=PARAMS):
    """Integrate the model with LSODA."""
    t_grid = np.arange(0.0, t_end + 0.5 * dt, dt)
    solution = solve_ivp(
        fun=lambda tau, state: hh_rhs(tau, state, input_current, p),
        t_span=(0.0, t_end),
        y0=x0,
        t_eval=t_grid,
        method="LSODA",
        rtol=1e-7,
        atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.t, solution.y.T

def settled_state(p=PARAMS):
    """Estimate the resting state without input."""
    _, states = simulate(lambda t: 0.0, t_end=250.0, dt=0.05, p=p)
    return states[-1]

def spike_times(times, voltage, threshold=0.0):
    """Find upward threshold crossings."""
    crossings = np.flatnonzero((voltage[:-1] < threshold) & (voltage[1:] >= threshold))
    return times[crossings]

def find_threshold(pulse_duration=40.0, p=PARAMS):
    """Estimate rheobase using bisection."""
    rest_state = settled_state(p)
    
    def has_spike(current):
        stimulus = step_current(current, 5.0, 5.0 + pulse_duration)
        _, states = simulate(stimulus, rest_state, t_end=55.0, p=p)
        return np.max(states[:, 0]) >= (rest_state[0] + 40.0)
    
    low, high = 0.0, 50.0
    while not has_spike(high):
        high *= 1.5
        
    for _ in range(28):
        mid = 0.5 * (low + high)
        if has_spike(mid):
            high = mid
        else:
            low = mid
            
    return rest_state, high

def firing_rate(current_level, p=PARAMS, window=200.0):
    """Compute firing rate in Hz."""
    rest = settled_state(p)
    stimulus = step_current(current_level, 5.0, 5.0 + window)
    timeseries_t, trajectory = simulate(stimulus, rest, t_end=window + 10.0, p=p)
    spikes = spike_times(timeseries_t, trajectory[:, 0])
    return len(spikes) / (window / 1000.0)

def run_analysis():
    resting_state, threshold_current = find_threshold()
    results = {
        "resting_state": resting_state.tolist(),
        "threshold_current": float(threshold_current)
    }

   
    t1, x1 = simulate(step_current(10.0, 10.0, 40.0), INITIAL_STATE, t_end=80.0)
    fig1, ax1 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1[0].plot(t1, x1[:, 0])
    ax1[0].axhline(resting_state[0], ls="--", lw=1)
    ax1[0].set_ylabel("V (mV)")
    ax1[0].set_title("Response from the stated initial condition")
    
    ax1[1].plot(t1, x1[:, 1], label="n")
    ax1[1].plot(t1, x1[:, 2], label="m")
    ax1[1].plot(t1, x1[:, 3], label="h")
    ax1[1].set_xlabel("Time (ms)")
    ax1[1].set_ylabel("Gate value")
    ax1[1].legend()
    fig1.tight_layout()
    fig1.savefig(FIG_DIR / "A1_initial_settling.png", dpi=180)
    plt.close(fig1)

    
    fig_a2, ax_a2 = plt.subplots(1, 2, figsize=(12, 4.6))
    for amp in [10, 20, 30, 40, 55]:
        ts, xs = simulate(step_current(amp, 5.0, 55.0), resting_state, t_end=90.0)
        ax_a2[0].plot(ts, xs[:, 0], label=f"{amp:g}")
    ax_a2[0].set(title="Amplitude sweep (50 ms pulse)", xlabel="Time (ms)", ylabel="V (mV)")
    ax_a2[0].legend(title="I")

    for pulse_ms in [5, 20, 50, 100]:
        ts, xs = simulate(step_current(35.0, 5.0, 5.0 + pulse_ms), resting_state, t_end=130.0)
        ax_a2[1].plot(ts, xs[:, 0], label=f"{pulse_ms:g} ms")
    ax_a2[1].set(title="Duration sweep (I=35)", xlabel="Time (ms)", ylabel="V (mV)")
    ax_a2[1].legend()
    fig_a2.tight_layout()
    fig_a2.savefig(FIG_DIR / "A2_amplitude_duration.png", dpi=180)
    plt.close(fig_a2)

    
    currents = np.linspace(0.0, 60.0, 16)
    frequencies = np.array([firing_rate(i_val) for i_val in currents])
    results["fi_amplitudes"] = currents.tolist()
    results["fi_rates"] = frequencies.tolist()

    fig_a3, ax_a3 = plt.subplots(figsize=(6.2, 4.5))
    ax_a3.plot(currents, frequencies, "o-")
    ax_a3.set(xlabel="Current (uA/cm2)", ylabel="Firing rate (Hz)", title="Frequency-current relation")
    fig_a3.tight_layout()
    fig_a3.savefig(FIG_DIR / "A3_fi_curve.png", dpi=180)
    plt.close(fig_a3)

   
    initial_conditions = {
        "stated": INITIAL_STATE,
        "settled": resting_state,
        "depolarized": np.array([-30.0, 0.32, 0.05, 0.60]),
        "hyperpolarized": np.array([-90.0, 0.32, 0.05, 0.60]),
    }
    fig_a4, ax_a4 = plt.subplots(figsize=(8, 4.8))
    for label, state0 in initial_conditions.items():
        ti, xi = simulate(step_current(10.0, 10.0, 40.0), state0, t_end=65.0)
        ax_a4.plot(ti, xi[:, 0], label=label)
    ax_a4.set(xlabel="Time (ms)", ylabel="V (mV)", title="Dependence on initial conditions")
    ax_a4.legend()
    fig_a4.tight_layout()
    fig_a4.savefig(FIG_DIR / "A4_initial_conditions.png", dpi=180)
    plt.close(fig_a4)

   
    fig_a5, ax_a5 = plt.subplots(1, 2, figsize=(12, 4.6))
    for na_value in [60, 120, 180, 240]:
        na_params = replace(PARAMS, g_na=float(na_value))
        equilibrium_state = settled_state(na_params)
        t_na, x_na = simulate(step_current(40.0, 5.0, 45.0), equilibrium_state, t_end=60.0, p=na_params)
        ax_a5[0].plot(t_na, x_na[:, 0], label=str(na_value))
        
    for k_value in [12, 36, 60, 90]:
        k_params = replace(PARAMS, g_k=float(k_value))
        equilibrium_state = settled_state(k_params)
        t_k, x_k = simulate(step_current(40.0, 5.0, 45.0), equilibrium_state, t_end=60.0, p=k_params)
        ax_a5[1].plot(t_k, x_k[:, 0], label=str(k_value))
        
    ax_a5[0].set(title="Sodium conductance", xlabel="Time (ms)", ylabel="V (mV)")
    ax_a5[0].legend(title="gNa")
    ax_a5[1].set(title="Potassium conductance", xlabel="Time (ms)", ylabel="V (mV)")
    ax_a5[1].legend(title="gK")
    fig_a5.tight_layout()
    fig_a5.savefig(FIG_DIR / "A5_conductance_sweeps.png", dpi=180)
    plt.close(fig_a5)

    
    na_range = np.array([60.0, 100.0, 140.0, 180.0, 220.0])
    k_range = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
    frequency_map = np.zeros((len(k_range), len(na_range)))
    
    for row, k_value in enumerate(k_range):
        for col, na_value in enumerate(na_range):
            frequency_map[row, col] = firing_rate(
                45.0, replace(PARAMS, g_na=float(na_value), g_k=float(k_value)), window=150.0
            )
            
    results["conductance_rate_grid"] = frequency_map.tolist()
    fig_a6, ax_a6 = plt.subplots(figsize=(6.5, 5))
    heatmap = ax_a6.imshow(frequency_map, origin="lower", aspect="auto", extent=[na_range[0], na_range[-1], k_range[0], k_range[-1]])
    ax_a6.set(xlabel="gNa", ylabel="gK", title="Firing-rate map")
    fig_a6.colorbar(heatmap, ax=ax_a6, label="Hz")
    fig_a6.tight_layout()
    fig_a6.savefig(FIG_DIR / "A6_conductance_map.png", dpi=180)
    plt.close(fig_a6)

    
    leak_values = np.array([0.1, 0.3, 0.6, 1.0, 2.0])
    rest_voltages = []
    fig7, ax7 = plt.subplots(1, 2, figsize=(12, 4.6))
    
    for leak_value in leak_values:
        leak_params = replace(PARAMS, g_leak=float(leak_value))
        equilibrium = settled_state(leak_params)
        rest_voltages.append(equilibrium[0])
        t_leak, x_leak = simulate(step_current(40.0, 5.0, 45.0), equilibrium, t_end=60.0, p=leak_params)
        ax7[1].plot(t_leak, x_leak[:, 0], label=str(leak_value))
        
    ax7[0].plot(leak_values, rest_voltages, "o-")
    ax7[0].set(xlabel="gL", ylabel="Resting V (mV)", title="Leak conductance and rest")
    ax7[1].set(xlabel="Time (ms)", ylabel="V (mV)", title="Leak conductance and waveform")
    ax7[1].legend(title="gL")
    fig7.tight_layout()
    fig7.savefig(FIG_DIR / "A7_leak.png", dpi=180)
    plt.close(fig7)
    results["leak_resting_voltage"] = rest_voltages

    scales = np.linspace(0.5, 2.0, 9)
    sensitivity = {}
    fig_a8, ax_a8 = plt.subplots(1, 3, figsize=(13, 4.2))
    
    parameters_to_test = [
        ("gNa", "g_na", PARAMS.g_na),
        ("gK", "g_k", PARAMS.g_k),
        ("C", "capacitance", PARAMS.capacitance),
    ]
    
    for sub_ax, (key, field, base) in zip(ax_a8, parameters_to_test):
        values = base * scales
        rates_list = []
        for value in values:
            updated_params = replace(PARAMS, **{field: float(value)})
            rates_list.append(firing_rate(40.0, updated_params, window=150.0))
            
        sensitivity[key] = {"values": values.tolist(), "rates": rates_list}
        sub_ax.plot(values, rates_list, "o-")
        sub_ax.set(title=key, xlabel=key, ylabel="Hz")
        
    fig_a8.tight_layout()
    fig_a8.savefig(FIG_DIR / "A8_sensitivity.png", dpi=180)
    plt.close(fig_a8)
    results["sensitivity"] = sensitivity

    subthreshold_currents = [0.2 * threshold_current, 0.4 * threshold_current, 0.6 * threshold_current, 0.8 * threshold_current]
    fig_a9, ax_a9 = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    
    for sub_current in subthreshold_currents:
        t_sub, x_sub = simulate(step_current(sub_current, 10.0, 60.0), resting_state, t_end=100.0)
        ax_a9[0].plot(t_sub, x_sub[:, 0], label=f"{sub_current:.2f}")
        ax_a9[1].plot(t_sub, x_sub[:, 2], label=f"{sub_current:.2f}")
        
    ax_a9[0].set(ylabel="V (mV)", title="Subthreshold voltage responses")
    ax_a9[0].legend(title="I")
    ax_a9[1].set(xlabel="Time (ms)", ylabel="m", title="Fast sodium activation")
    fig_a9.tight_layout()
    fig_a9.savefig(FIG_DIR / "A9_subthreshold.png", dpi=180)
    plt.close(fig_a9)

    frequencies_in = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    response_amplitudes = []
    
    for freq in frequencies_in:
        sinusoidal_input = lambda time, f=freq: 0.15 * threshold_current * np.sin(2.0 * np.pi * f * time / 1000.0)
        t_res, x_res = simulate(sinusoidal_input, resting_state, t_end=400.0)
        steady_voltage = x_res[t_res > 200.0, 0]
        response_amplitudes.append(np.ptp(steady_voltage))
        
    results["resonance_frequencies"] = frequencies_in.tolist()
    results["resonance_amplitudes"] = response_amplitudes

    fig_a10, ax_a10 = plt.subplots(figsize=(6.2, 4.5))
    ax_a10.plot(frequencies_in, response_amplitudes, "o-")
    ax_a10.set(xlabel="Frequency (Hz)", ylabel="Peak-to-peak V (mV)", title="Subthreshold frequency response")
    fig_a10.tight_layout()
    fig_a10.savefig(FIG_DIR / "A10_resonance.png", dpi=180)
    plt.close(fig_a10)

    (FIG_DIR / "hh_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"rest": resting_state.tolist(), "threshold": threshold_current}, indent=2))

if __name__ == "__main__":
    run_analysis()
