.PHONY: test lint generate-sample-data ingest-noaa build-features train-model train-hybrid update-twin run-api run-dashboard run-frontend run-experiments

test:
	pytest -q

lint:
	ruff check .

generate-sample-data:
	python -m pipelines.simulate_iot_stream --output data/bronze/iot_readings.csv --rows 5000

ingest-noaa:
	python -m pipelines.ingest_noaa_crw --output data/bronze/noaa_crw_sample.csv

ingest-noaa-real:
	python -m pipelines.ingest_noaa_real --output data/bronze/noaa_crw_sample.csv

ingest-netcdf:
	python -m pipelines.ingest_netcdf --input $(INPUT) --output data/bronze/noaa_crw_sample.csv

compile-kfp:
	python -m pipelines.kfp.pipeline

build-features:
	python -m pipelines.build_features --iot data/bronze/iot_readings.csv --noaa data/bronze/noaa_crw_sample.csv --output data/gold/reef_features.parquet

train-model:
	python -m models.bleaching_risk.train --features data/gold/reef_features.parquet --model-out models/bleaching_risk/model.joblib

train-hybrid:
	python -m models.reef_dynamics.train_hybrid --features data/gold/reef_features.parquet --model-out models/reef_dynamics/hybrid_model.joblib

update-twin:
	python -m pipelines.update_twin_state --features data/gold/reef_features.parquet --model models/bleaching_risk/model.joblib --output data/gold/reef_state.json

generate-geotiff:
	python -m pipelines.generate_geotiff --features data/gold/reef_features.parquet

run-api:
	uvicorn services.twin_api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run services/dashboard/app.py --server.port 8501

run-frontend:
	cd frontend && npm run dev

run-experiments:
	python scripts/run_experiments.py
