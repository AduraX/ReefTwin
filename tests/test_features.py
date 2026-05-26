from pipelines.ingest_noaa_crw import generate_noaa_sample
from pipelines.simulate_iot_stream import generate_readings
from pipelines.build_features import build_features


def test_generators_return_data(tmp_path):
    iot_path = tmp_path / "iot.csv"
    noaa_path = tmp_path / "noaa.csv"
    generate_readings(100).to_csv(iot_path, index=False)
    generate_noaa_sample(10).to_csv(noaa_path, index=False)
    df = build_features(str(iot_path), str(noaa_path))
    assert not df.empty
    assert "bleaching_label" in df.columns
    assert "degree_heating_weeks" in df.columns
