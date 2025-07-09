# Performance Benchmarks

This document summarizes benchmark results for the end-to-end data ingestion and training pipeline.

## Data Pipeline

The synthetic data generation and feature engineering pipeline completes in **under 2 ms** on the test environment.

## Training Loop

A single PPO training iteration using the mocked trainer completes in **under 3 ms**.

These numbers come from the `pytest-benchmark` results in `tests/integration/test_end_to_end_pipeline.py`.

--

For legal and safety notes see the [project disclaimer](disclaimer.md).
