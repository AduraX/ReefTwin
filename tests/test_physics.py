import numpy as np
import pytest

from models.reef_dynamics.physics import (
    dhw_accumulation_rate,
    simulate_reef_stress,
)


def test_dhw_no_accumulation_below_threshold():
    # SST at or below MMM should not accumulate DHW
    assert dhw_accumulation_rate(28.0, mmm=28.2) == 0.0
    assert dhw_accumulation_rate(28.5, mmm=28.2) == 0.0  # hotspot < 1


def test_dhw_accumulates_above_threshold():
    # SST 2°C above MMM → hotspot=2, should accumulate
    rate = dhw_accumulation_rate(30.2, mmm=28.2)
    assert rate == pytest.approx(2.0, abs=0.01)


def test_simulate_stable_conditions():
    # Cool water should produce low stress and low bleaching risk
    sst_cool = np.full(12, 27.0)
    result = simulate_reef_stress(sst_cool)
    assert result["bleaching_risk"][-1] < 0.2
    assert result["stress"][-1] < 0.1


def test_simulate_heat_stress():
    # Sustained high SST should produce elevated DHW and bleaching risk
    sst_hot = np.full(12, 31.0)
    result = simulate_reef_stress(sst_hot)
    assert result["dhw"][-1] > 4.0
    assert result["bleaching_risk"][-1] > 0.3


def test_simulate_output_shape():
    n = 8
    sst = np.full(n, 29.0)
    result = simulate_reef_stress(sst)
    assert len(result["time_weeks"]) == n
    assert len(result["dhw"]) == n
    assert len(result["stress"]) == n
    assert len(result["bleaching_risk"]) == n


def test_stress_bounded_zero_one():
    sst = np.linspace(27, 33, 20)
    result = simulate_reef_stress(sst)
    assert np.all(result["stress"] >= 0.0)
    assert np.all(result["stress"] <= 1.0)
    assert np.all(result["bleaching_risk"] >= 0.0)
    assert np.all(result["bleaching_risk"] <= 1.0)
