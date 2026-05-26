# Experiment Plan

## Experiment 1: Pipeline latency reduction

Goal: reduce reef-state update latency by 35%.

Baseline:
- batch-only update
- compute latency from data arrival to state update

Optimization:
- streaming ingestion
- async update pipeline
- compact feature updates

Metrics:
- p50 and p95 state update latency
- ingestion-to-state lag

## Experiment 2: Inference cost reduction

Goal: cut inference cost by 22%.

Baseline:
- run model on every event

Optimization:
- cache unchanged reef states
- batch predictions
- trigger inference only when feature drift exceeds threshold

Metrics:
- model invocations per hour
- skipped predictions
- cost per 1,000 state updates

## Experiment 3: Reliability improvement

Goal: improve pipeline reliability from 97% to 99.9%.

Failure cases:
- missing sensor values
- delayed NOAA updates
- malformed records
- consumer crash

Mechanisms:
- schema validation
- retries
- dead-letter queue
- health checks
- checkpointing

Metrics:
- successful pipeline runs / total pipeline runs
- mean time to recovery
- failed records quarantined
