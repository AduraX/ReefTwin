from models.stress_scoring import ReefStressModel, StressWeights


def test_low_stress_conditions():
    model = ReefStressModel()
    result = model.score({
        "sst_anomaly_c": -0.5,
        "hotspot_c": 0.0,
        "turbidity_ntu": 0.3,
        "ph": 8.1,
        "dissolved_oxygen_mg_l": 7.0,
        "degree_heating_weeks": 0.0,
    })
    assert result.total_score < 0.3
    assert result.dominant_stressor in ("thermal", "water_quality", "biological", "cumulative")


def test_high_thermal_stress():
    model = ReefStressModel()
    result = model.score({
        "sst_anomaly_c": 3.0,
        "hotspot_c": 2.5,
        "turbidity_ntu": 0.5,
        "ph": 8.1,
        "dissolved_oxygen_mg_l": 6.5,
        "degree_heating_weeks": 10.0,
    })
    assert result.total_score > 0.5
    assert result.cumulative_score > 0.5


def test_custom_weights():
    weights = StressWeights(thermal=0.0, water_quality=0.0, biological=0.0, cumulative=1.0)
    model = ReefStressModel(weights)
    result = model.score({
        "sst_anomaly_c": 3.0,
        "hotspot_c": 2.5,
        "degree_heating_weeks": 10.0,
    })
    # With only cumulative weight, total should equal cumulative score
    assert abs(result.total_score - result.cumulative_score) < 0.01


def test_breakdown_fields_present():
    model = ReefStressModel()
    result = model.score({"water_temperature_c": 29.0})
    assert hasattr(result, "thermal_score")
    assert hasattr(result, "water_quality_score")
    assert hasattr(result, "biological_score")
    assert hasattr(result, "cumulative_score")
    assert hasattr(result, "dominant_stressor")
    assert hasattr(result, "weights_used")


def test_score_bounded():
    model = ReefStressModel()
    # Extreme stress
    result = model.score({
        "sst_anomaly_c": 10.0, "hotspot_c": 10.0,
        "turbidity_ntu": 50.0, "ph": 6.5,
        "dissolved_oxygen_mg_l": 1.0, "degree_heating_weeks": 20.0,
    })
    assert 0.0 <= result.total_score <= 1.0
