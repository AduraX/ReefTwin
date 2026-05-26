from infrastructure.settings import settings


def test_settings_loads_reef_ids():
    assert len(settings.reef_ids) == 3
    assert "gbr_heron_reef" in settings.reef_ids


def test_settings_loads_features():
    assert "water_temperature_c" in settings.features
    assert "degree_heating_weeks" in settings.features
    assert len(settings.features) == 9


def test_risk_thresholds_consistent():
    t = settings.risk_thresholds
    assert t.alert > t.warning > t.watch > t.normal
    assert t.alert == 0.85
    assert t.warning == 0.70
    assert t.watch == 0.50


def test_bleaching_thresholds_positive():
    assert settings.bleaching_dhw_threshold > 0
    assert settings.bleaching_temp_threshold > 0
    assert settings.bleaching_hotspot_threshold > 0


def test_simulation_weights_positive():
    assert settings.sim_temperature_weight > 0
    assert settings.sim_duration_weight > 0
    assert settings.sim_turbidity_weight > 0
    assert settings.sim_acidification_weight > 0
