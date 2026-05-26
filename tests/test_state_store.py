
from infrastructure.db.base import JsonStateStore


def test_json_state_store_roundtrip(tmp_path):
    store = JsonStateStore(tmp_path / "state.json")
    payload = {
        "generated_at": "2026-05-07T00:00:00Z",
        "states": [{"reef_id": "test_reef", "bleaching_risk_score": 0.75}],
    }
    store.save(payload)
    loaded = store.load()
    assert loaded["generated_at"] == "2026-05-07T00:00:00Z"
    assert len(loaded["states"]) == 1
    assert loaded["states"][0]["reef_id"] == "test_reef"


def test_json_state_store_load_missing(tmp_path):
    store = JsonStateStore(tmp_path / "nonexistent.json")
    result = store.load()
    assert result["states"] == []
    assert result["generated_at"] is None


def test_json_state_store_load_states(tmp_path):
    store = JsonStateStore(tmp_path / "state.json")
    store.save({"states": [{"reef_id": "a"}, {"reef_id": "b"}]})
    states = store.load_states()
    assert len(states) == 2
    assert states[0]["reef_id"] == "a"


def test_json_state_store_creates_parent_dirs(tmp_path):
    store = JsonStateStore(tmp_path / "nested" / "deep" / "state.json")
    store.save({"states": []})
    assert store.path.exists()
