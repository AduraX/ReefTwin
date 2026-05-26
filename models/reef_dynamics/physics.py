"""Physics-based reef thermal stress model.

Encodes known coral bleaching physics as a system of ODEs:
  - Degree Heating Weeks (DHW) accumulates when SST exceeds the bleaching threshold
  - Coral stress follows a logistic response to cumulative thermal exposure
  - Recovery occurs when thermal stress drops, but with hysteresis (slower recovery than onset)

This is the "physics prior" used by the hybrid PIML predictor.
References:
  - NOAA Coral Reef Watch: DHW = sum of HotSpots > 1°C over rolling 12 weeks
  - Liu et al. (2014): Reef-scale thermal stress monitoring
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


# NOAA-standard bleaching threshold (mean of warmest month + 1°C)
DEFAULT_MMM = 28.2  # approximate for GBR


def dhw_accumulation_rate(sst: float, mmm: float = DEFAULT_MMM) -> float:
    """Instantaneous DHW accumulation rate (°C-weeks per week).

    HotSpot = max(0, SST - MMM). DHW accumulates when HotSpot >= 1°C.
    """
    hotspot = max(0.0, sst - mmm)
    return hotspot if hotspot >= 1.0 else 0.0


def reef_stress_ode(
    t: float,
    state: np.ndarray,
    sst_func: callable,
    mmm: float = DEFAULT_MMM,
    stress_rate: float = 0.15,
    recovery_rate: float = 0.03,
    dhw_critical: float = 8.0,
) -> np.ndarray:
    """ODE system for reef thermal stress dynamics.

    State variables:
        state[0] = dhw: cumulative degree heating weeks
        state[1] = stress: coral stress level [0, 1]

    Physics encoded:
        dDHW/dt = dhw_accumulation_rate(SST)  (NOAA formula)
        dStress/dt = onset_term - recovery_term
            onset  = stress_rate * sigmoid(DHW - DHW_critical) * (1 - stress)
            recovery = recovery_rate * stress * (1 - DHW/DHW_critical)^+
    """
    dhw, stress = state
    sst = sst_func(t)

    # DHW accumulation (NOAA physics)
    d_dhw = dhw_accumulation_rate(sst, mmm)

    # Stress dynamics with logistic onset and hysteresis recovery
    dhw_ratio = dhw / dhw_critical
    onset_signal = 1.0 / (1.0 + np.exp(-5.0 * (dhw_ratio - 1.0)))  # sigmoid around DHW_critical
    onset = stress_rate * onset_signal * (1.0 - stress)
    recovery_potential = max(0.0, 1.0 - dhw_ratio)
    recovery = recovery_rate * stress * recovery_potential

    d_stress = onset - recovery

    return np.array([d_dhw, d_stress])


def simulate_reef_stress(
    sst_series: np.ndarray,
    dt_weeks: float = 1.0,
    mmm: float = DEFAULT_MMM,
    initial_dhw: float = 0.0,
    initial_stress: float = 0.0,
    stress_rate: float = 0.15,
    recovery_rate: float = 0.03,
    dhw_critical: float = 8.0,
) -> dict[str, np.ndarray]:
    """Simulate reef stress evolution given an SST time series.

    Args:
        sst_series: Array of sea surface temperatures (one per time step).
        dt_weeks: Time step size in weeks.
        mmm: Maximum Monthly Mean SST (bleaching threshold baseline).
        initial_dhw: Initial cumulative DHW.
        initial_stress: Initial coral stress level [0, 1].

    Returns:
        Dictionary with arrays: dhw, stress, hotspot, bleaching_risk
    """
    n_steps = len(sst_series)
    t_span = (0.0, (n_steps - 1) * dt_weeks)
    t_eval = np.linspace(t_span[0], t_span[1], n_steps)

    # Interpolate SST for the ODE solver
    def sst_func(t: float) -> float:
        idx = min(int(t / dt_weeks), n_steps - 1)
        return float(sst_series[idx])

    sol = solve_ivp(
        reef_stress_ode,
        t_span,
        y0=np.array([initial_dhw, initial_stress]),
        t_eval=t_eval,
        args=(sst_func, mmm, stress_rate, recovery_rate, dhw_critical),
        method="RK45",
        max_step=dt_weeks / 2,
    )

    dhw = sol.y[0]
    stress = np.clip(sol.y[1], 0.0, 1.0)
    hotspot = np.array([max(0.0, sst - mmm) for sst in sst_series])

    # Bleaching risk combines stress level with DHW exceedance
    bleaching_risk = np.clip(
        0.6 * stress + 0.4 * np.clip(dhw / dhw_critical, 0, 1),
        0.0,
        1.0,
    )

    return {
        "time_weeks": t_eval,
        "dhw": dhw,
        "stress": stress,
        "hotspot": hotspot,
        "bleaching_risk": bleaching_risk,
    }
