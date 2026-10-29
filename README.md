# AI-engineering-base

A set of reusable building blocks for AI and ML projects. Nothing here is tied to one dataset, model, or business problem. The code covers the parts most AI projects end up rebuilding: loading data, checking its quality, labeling it, training or running a model, measuring results, detecting anomalies, retrieving context for a language model, and reporting what happened. Point it at a new dataset with a config file and most of it works unchanged.

## What it does

The repository is a seven-stage pipeline plus supporting modules. Each stage reads the previous stage's output and writes its own output and a manifest to a local directory or a `gs://` bucket, so runs can be inspected, repeated, or compared.

| Stage | Script | Purpose |
|---|---|---|
| 1. Ingestion | `pipeline/sample.py` | Reads CSV, JSON, JSONL, or Parquet from a local path, `gs://`, or `bq://`. Keeps the original file and writes a canonical JSONL copy plus a manifest. |
| 2. Preparation | `pipeline/preprocess.py` | Profiles the data and reports missing values, duplicates, type problems, and unexpected categories. It does not silently drop, impute, or coerce anything unless a config rule asks for it. |
| 3. Anomaly detection | `pipeline/anomaly_detection.py` | Finds density-based outliers with Local Outlier Factor, optionally against a reference dataset that represents normal behavior. |
| 4. Labeling | `pipeline/label.py` | Generates pseudo-labels with an expert model when ground truth is missing. Each label records its source and confidence so it stays distinct from ground truth. |
| 5. Training | `pipeline/train.py` | Trains a configurable scikit-learn classifier or regressor, splits the data, handles numeric and categorical features, and saves the model with its preprocessing. |
| 6. Inference | `pipeline/inference.py` | Runs a model over prepared data through a common interface. Optionally runs the multi-agent analysis in `agents.py` on the predictions. |
| 7. Metrics | `pipeline/compute_metrics.py` | Computes classification or regression metrics from labeled records. It keeps ground-truth results separate from pseudo-label results. |

Supporting modules:

- **`pipeline/common.py`**: shared read and write helpers. Local paths and `gs://` URIs behave the same way in every stage.
- **`pipeline/rag.py`**: retrieval-augmented generation. It builds a vector index from BigQuery rows and GCS documents, embeds them with a BERT-family encoder, and retrieves relevant chunks to ground a language model's answers.
- **`pipeline/agents.py`**: a LangChain supervisor that delegates to specialist agents. Each specialist can call only the read-only tools it is allowed. The agents write every model reply and tool call to a trace and cannot change source data or other stages' outputs.
- **`pipeline/dashboard.py`**: a Streamlit dashboard for inspecting a run. It shows dataset characteristics, quality findings, metrics, anomalies, and individual records. It labels model-generated text as such.
- **`pipeline/workflow.yaml`**: a Google Cloud Workflows definition that runs the stages in order as Cloud Run Jobs. Stages 4 (labeling) and 5 (training) can be turned on or off per run.

## Design principles

- **Configuration, not code changes.** Domain-specific behavior lives in JSON files under `pipeline/config/`: labeling, training, inference, and metrics. Adding a dataset or model usually means editing config.
- **Stages can be skipped or run alone.** `make run-<stage>` runs one stage. `RUN_LABELING` and `RUN_TRAINING` switch the optional stages on and off for a full run.
- **Nothing is silently changed.** Stages preserve the original input, record their settings in manifests, and report problems instead of hiding them.
- **Generated text and pseudo-labels are labeled.** Language-model output and model-generated labels stay distinct from verified results throughout the pipeline and the dashboard.
- **Requirements are traceable.** `PRD.md` states the product requirements. `QRD.md` states the quality requirements, each with a test ID, and `tests/` maps every test to its QRD entry. `ERD.md` covers the data model.

## Repository layout

```
pipeline/            stage scripts, shared helpers, dashboard, workflow, Dockerfile
pipeline/config/     JSON config for labeling, training, inference, and metrics
tests/               pytest suite mapped to the QRD (see tests/README.md)
smoke/               small sample dataset for an end-to-end check
Makefile             setup, IAM, image build, Cloud Run deploys, workflow runs, monitoring
PRD.md, QRD.md, ERD.md   requirements and data model
```

## Getting started

### Run the tests

The tests run offline. They use a bag-of-words encoder in place of BERT and a fake Ollama server in place of a real model, so they need no cloud access or model downloads.

```bash
python -m pytest tests -v
# or
make test
```

Tests that need live BigQuery are marked `requires_bigquery`. They run only when you set `RAG_TEST_TABLE`, `RAG_TEST_PREFIX`, and optionally `RAG_TEST_MODEL`. Details are in [tests/README.md](tests/README.md).

### Run the dashboard locally

```bash
make dashboard-local
```

### Run the pipeline on Google Cloud

The Makefile deploys the pipeline to Cloud Run Jobs and Workflows. The variables you are most likely to set are:

| Variable | Meaning |
|---|---|
| `PROJECT_ID`, `REGION` | Google Cloud project and region |
| `ARTIFACT_BUCKET` | Bucket where run outputs are written |
| `SOURCE_URI` | Input data, such as `gs://bucket/input/data.csv` |
| `RUN_ID` | Name for this run's output folder |
| `RUN_LABELING`, `RUN_TRAINING` | Turn the optional stages on or off |

A typical first-time sequence:

```bash
make setup                 # enable APIs and create the Artifact Registry repository
make iam                   # grant the pipeline service account its roles
make build push            # build and push the container image
make deploy                # deploy the Cloud Run Jobs and the workflow
make run-workflow SOURCE_URI=gs://bucket/input/data.csv RUN_ID=run-001
```

Run `make help` to see every target. Use `make status`, `make executions`, and `make logs` to monitor jobs.

## Adapting it to a new project

1. Put the new data in a file or table and set `SOURCE_URI`.
2. Edit the JSON files in `pipeline/config/` to name the target field, the task (classification or regression), the features, and the metrics you need.
3. Run the stages you need with `make run-<stage>`. Skip the ones you don't need.
4. Add project-specific checks or tools in the stage scripts only where the config can't express them. The shared engine stays unchanged.

## Status

The core stages, the RAG and agent modules, the dashboard, and the test suite are in place, and the pipeline has run end to end on a sample input. Some requirements are still pending: BigQuery and GCS connectors beyond the current ones, Cloud Run failure and retry handling, IAM and retention checks, dashboard rendering tests, and some detectors and evaluators. Those tests are marked `@pending` in the suite, so they show as expected failures until the behavior exists. See the "Pending tests" section of [tests/README.md](tests/README.md) for the full list.
