# Pytest tests mapped to the QRD

Every test ID in `QRD.md` (1.1.a through 12.5.c, 156 in total) has a test function named after it, with the QRD text in its docstring. Tests are grouped by QRD section:

- `test_stage_1_sample.py` — Section 1, data ingestion
- `test_preprocess.py` — Section 2, validation and preparation
- `test_qrd_03_model_selection.py` — Section 3, training and inference
- `test_qrd_03_retrieval_agents.py` — Section 3.8 (RAG) and 3.9 (LangChain agents)
- `test_qrd_04_labeling.py` — Section 4, pseudo-labels
- `test_qrd_05_evaluation.py` — Section 5, metrics
- `test_qrd_06_anomaly_ood.py` — Section 6, anomaly and OOD detection
- `test_qrd_07_dashboard.py` — Section 7, dashboard
- `test_qrd_08_workflow_config.py` — Section 8, workflow configuration
- `test_qrd_09_run_management.py` — Section 9, run records and reporting
- `test_qrd_10_reliability_responsible_use.py` — Section 10, reliability and responsible use
- `test_qrd_11_pipeline_contract.py` — Section 11, stage order and artifact contract
- `test_qrd_12_implementation_decisions.py` — Section 12, implementation decisions

Shared fixtures and stage runners (`run_train`, `run_label`, `run_metrics`, `run_anomaly`, `run_inference_stage`) live in `conftest.py`. `FakeOllamaServer` is a local HTTP stand-in for the Ollama endpoint, so inference and agent tests run without a model.

Retrieval and agent tests run offline, using a bag-of-words encoder and an in-memory vector search in place of BERT and BigQuery `VECTOR_SEARCH`. Agent tests need `langchain-core` and skip without it. Tests marked `requires_bigquery` run only against a live table when you set:

```bash
export RAG_TEST_TABLE=project.dataset.records   # source table
export RAG_TEST_PREFIX=gs://bucket/path         # GCS prefix for 12.6.a
export RAG_TEST_MODEL=sentence-transformers/all-MiniLM-L6-v2  # optional
```

## Run

From the project root, with `pytest` installed:

```bash
python -m pytest tests -v
```

Optional Parquet test skips if `pandas` or `pyarrow` is unavailable.

## Pending tests

A test marked `@pending("reason")` is written but the behavior it checks is not implemented yet. It raises `NotImplementedError` and pytest reports it as `xfail` with the reason. When the behavior lands, replace the body with real assertions and remove the marker.

Pending areas:

- BigQuery and GCS connectors, and cross-storage manifest references
- Cloud Run failure, interruption, retry, and manifest-publication gates (need deployed jobs)
- IAM least privilege, Secret Manager, retention and deletion, and export permissions
- Dashboard rendering, lineage, and access control (needs a Streamlit AppTest harness and fixture run artifacts)
- Detectors and evaluators beyond LOF anomaly detection and classification/regression metrics, plus extension registries
- Model comparison, artifact reuse, consolidated run summaries, and label-source propagation into training and metrics

Cloud integration tests should use a dedicated test project and bucket and be marked separately (for example, `@pytest.mark.integration`).
