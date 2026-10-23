# Pytest tests mapped to the QRD

These tests currently target the two Python stage scripts present in this project:

- `stage_1_sample.py` — Stage 1 ingestion
- `preprocess.py` — Stage 2 validation and preparation

They use QRD identifiers in test names/docstrings where practical. Tests requiring live GCP resources are intentionally not run as unit tests.

## Run

From the project root, with `pytest` installed:

```bash
python -m pytest tests -v
```

Optional Parquet test skips if `pandas` or `pyarrow` is unavailable.

## Known coverage gaps against the QRD

The QRD includes requirements that the current scripts do not implement yet. These are not silently represented as passing tests:

- BigQuery ingestion and direct GCS connector integration tests
- Explicit image decoding/processing (current ingestion only preserves image references)
- Ingestion configuration for schema and task selection
- Imputation, fitted feature extraction, and train/test leakage checks
- Anomaly detection and model-level OOD detection
- Pseudo-label review/accept/reject policy and label provenance propagation into metrics
- Training, inference, evaluator registration, BigQuery metrics persistence
- Dashboard quality/anomaly/lineage presentation and access-control tests
- Workflow branching, retries, run-level status, artifact publication gates, and end-to-end GCS/BigQuery lineage

Add stage-specific test modules once those implementation files and interfaces are committed. Cloud integration tests should use a dedicated test project/bucket and be marked separately (for example, `@pytest.mark.integration`).
