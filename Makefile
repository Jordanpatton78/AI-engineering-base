
# ============================================================
# Configuration
# ============================================================

PROJECT_ID ?= ai-engineering-512517
REGION ?= us-central1
AR_REPOSITORY ?= ml-pipeline
IMAGE_NAME ?= ml-pipeline
IMAGE_TAG ?= latest

SERVICE_ACCOUNT ?= ml-pipeline@$(PROJECT_ID).iam.gserviceaccount.com
ARTIFACT_BUCKET ?= $(PROJECT_ID)-artifacts

WORKFLOW_NAME ?= ml-pipeline
WORKFLOW_FILE ?= pipeline/workflow.yaml

IMAGE_URI := $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(AR_REPOSITORY)/$(IMAGE_NAME):$(IMAGE_TAG)

# These configuration files must exist inside the container image.
LABEL_CONFIG ?= /app/config/label_config.json
TRAIN_CONFIG ?= /app/config/train_config.json
INFERENCE_CONFIG ?= /app/config/inference_config.json
METRICS_CONFIG ?= /app/config/metrics_config.json

RUN_ID ?= run-001
SOURCE_URI ?= gs://$(ARTIFACT_BUCKET)/input/data.csv

BASE_URI = gs://$(ARTIFACT_BUCKET)/runs/$(RUN_ID)
STAGE_1_URI = $(BASE_URI)/stage_1
STAGE_2_URI = $(BASE_URI)/stage_2
STAGE_3_URI = $(BASE_URI)/stage_3
STAGE_4_URI = $(BASE_URI)/stage_4
STAGE_5_URI = $(BASE_URI)/stage_5
STAGE_6_URI = $(BASE_URI)/stage_6
STAGE_7_URI = $(BASE_URI)/stage_7

RUN_LABELING ?= false
RUN_TRAINING ?= true

# Training reads labeled data only when labeling ran, matching workflow.yaml.
TRAIN_INPUT = $(if $(filter true,$(RUN_LABELING)),$(STAGE_4_URI)/labeled.jsonl,$(STAGE_2_URI)/prepared.jsonl)
TRAIN_INPUT_MANIFEST = $(if $(filter true,$(RUN_LABELING)),$(STAGE_4_URI)/manifest.json,$(STAGE_2_URI)/manifest.json)

# Fixed model location shared by training and inference (matches inference_config.json).
MODEL_URI ?= gs://$(ARTIFACT_BUCKET)/models/smoke/model.joblib

# ============================================================
# Job names — must match workflow.yaml
# ============================================================

JOB_SAMPLING = ml-stage-1-ingestion
JOB_PREPROCESSING = ml-stage-2-preprocess
JOB_ANOMALY = ml-stage-3-anomaly-detection
JOB_LABELING = ml-stage-4-labeling
JOB_TRAINING = ml-stage-5-training
JOB_INFERENCE = ml-stage-6-inference
JOB_METRICS = ml-stage-7-compute-metrics

JOBS = \
	$(JOB_SAMPLING) \
	$(JOB_PREPROCESSING) \
	$(JOB_ANOMALY) \
	$(JOB_LABELING) \
	$(JOB_TRAINING) \
	$(JOB_INFERENCE) \
	$(JOB_METRICS)

.PHONY: help test setup iam configure build push deploy-all \
	deploy-sampling deploy-preprocessing deploy-anomaly \
	deploy-labeling deploy-training deploy-inference deploy-metrics \
	deploy-workflow run-workflow \
	run-sampling run-preprocessing run-anomaly run-labeling \
	run-training run-inference run-metrics \
	dashboard-local deploy-dashboard \
	status executions logs clean-jobs

help:
	@echo "Setup:"
	@echo "  make setup"
	@echo "  make iam          # grant the service account its roles"
	@echo "  make build push deploy-all deploy-workflow"
	@echo ""
	@echo "Run the full pipeline:"
	@echo "  make run-workflow SOURCE_URI=gs://bucket/input/data.csv RUN_ID=run-001"
	@echo ""
	@echo "Run individual stages:"
	@echo "  make run-sampling"
	@echo "  make run-preprocessing"
	@echo "  make run-anomaly"
	@echo "  make run-labeling"
	@echo "  make run-training"
	@echo "  make run-inference"
	@echo "  make run-metrics"
	@echo ""
	@echo "Dashboard:"
	@echo "  make dashboard-local"
	@echo "  make deploy-dashboard"

# ============================================================
# Run tests
# ============================================================

test:
	python -m pytest tests -q

# ============================================================
# Setup and image
# ============================================================

configure:
	gcloud config set project $(PROJECT_ID)

setup: configure
	gcloud services enable \
		run.googleapis.com \
		workflows.googleapis.com \
		artifactregistry.googleapis.com \
		cloudbuild.googleapis.com \
		storage.googleapis.com \
		--project=$(PROJECT_ID)
	# Workflows creates its service agent lazily; creating it here avoids
	# FAILED_PRECONDITION ("Workflows service agent does not exist") on deploy.
	gcloud beta services identity create \
		--service=workflows.googleapis.com \
		--project=$(PROJECT_ID)
	gcloud artifacts repositories describe $(AR_REPOSITORY) \
		--location=$(REGION) \
		--project=$(PROJECT_ID) >/dev/null 2>&1 || \
	gcloud artifacts repositories create $(AR_REPOSITORY) \
		--repository-format=docker \
		--location=$(REGION) \
		--project=$(PROJECT_ID)
	@echo "Ensure the artifact bucket exists: gs://$(ARTIFACT_BUCKET)"
	@echo "Ensure the service account exists, then run: make iam"

# ============================================================
# IAM
#
# Grants the pipeline service account the roles its stages use. Run once per
# project, after make setup. Project-wide roles cover BigQuery and Cloud Run;
# narrow them to specific datasets if the project holds other data.
# The deployer (DEPLOYER, or the active gcloud account) is granted
# serviceAccountUser so jobs and workflows can run as the service account.
# ============================================================

PIPELINE_PROJECT_ROLES = \
	roles/bigquery.jobUser \
	roles/bigquery.dataEditor \
	roles/run.developer \
	roles/secretmanager.secretAccessor \
	roles/logging.logWriter \
	roles/artifactregistry.reader

iam:
	@for role in $(PIPELINE_PROJECT_ROLES); do \
		gcloud projects add-iam-policy-binding $(PROJECT_ID) \
			--member=serviceAccount:$(SERVICE_ACCOUNT) \
			--role=$$role \
			--condition=None >/dev/null || exit 1; \
	done
	gcloud storage buckets add-iam-policy-binding gs://$(ARTIFACT_BUCKET) \
		--member=serviceAccount:$(SERVICE_ACCOUNT) \
		--role=roles/storage.objectAdmin
	@deployer=$${DEPLOYER:-$$(gcloud config get-value account 2>/dev/null)}; \
	gcloud iam service-accounts add-iam-policy-binding $(SERVICE_ACCOUNT) \
		--member="user:$$deployer" \
		--role=roles/iam.serviceAccountUser \
		--project=$(PROJECT_ID) >/dev/null
	@echo "Granted pipeline roles to $(SERVICE_ACCOUNT)."

build:
	gcloud builds submit \
		--tag=$(IMAGE_URI) \
		--project=$(PROJECT_ID) pipeline

push: build
	@echo "Cloud Build builds and pushes the image to Artifact Registry."

# ============================================================
# Deploy Cloud Run Jobs
#
# One shared image, one Python script per job.
# ENTRYPOINT/CMD must permit the script name to be overridden.
# ============================================================

define DEPLOY_JOB
gcloud run jobs deploy $(1) \
	--image=$(IMAGE_URI) \
	--region=$(REGION) \
	--project=$(PROJECT_ID) \
	--command=python \
	--args=$(2) \
	--service-account=$(SERVICE_ACCOUNT) \
	--set-env-vars=ARTIFACT_BUCKET=$(ARTIFACT_BUCKET) \
	--max-retries=1 \
	--task-timeout=3600s
endef

deploy-sampling:
	$(call DEPLOY_JOB,$(JOB_SAMPLING),sample.py)

deploy-preprocessing:
	$(call DEPLOY_JOB,$(JOB_PREPROCESSING),preprocess.py)

deploy-anomaly:
	$(call DEPLOY_JOB,$(JOB_ANOMALY),anomaly_detection.py)

deploy-labeling:
	$(call DEPLOY_JOB,$(JOB_LABELING),label.py)

deploy-training:
	$(call DEPLOY_JOB,$(JOB_TRAINING),train.py)

deploy-inference:
	$(call DEPLOY_JOB,$(JOB_INFERENCE),inference.py)

deploy-metrics:
	$(call DEPLOY_JOB,$(JOB_METRICS),compute_metrics.py)

deploy-all: deploy-sampling deploy-preprocessing deploy-anomaly \
	deploy-labeling deploy-training deploy-inference deploy-metrics

# ============================================================
# Deploy Workflows
# ============================================================

deploy-workflow:
	gcloud workflows deploy $(WORKFLOW_NAME) \
		--source=$(WORKFLOW_FILE) \
		--location=$(REGION) \
		--project=$(PROJECT_ID) \
		--service-account=$(SERVICE_ACCOUNT)

deploy: deploy-all deploy-workflow

run-workflow:
	gcloud workflows run $(WORKFLOW_NAME) \
		--location=$(REGION) \
		--project=$(PROJECT_ID) \
		--data='{"project_id":"$(PROJECT_ID)","region":"$(REGION)","bucket":"$(ARTIFACT_BUCKET)","source_uri":"$(SOURCE_URI)","run_id":"$(RUN_ID)","run_labeling":$(RUN_LABELING),"run_training":$(RUN_TRAINING),"label_config":"$(LABEL_CONFIG)","train_config":"$(TRAIN_CONFIG)","inference_config":"$(INFERENCE_CONFIG)","metrics_config":"$(METRICS_CONFIG)"}'

# ============================================================
# Individual stage execution
#
# These targets override each job's default arguments.
# The script-specific CLI flags must match the Python files.
# ============================================================

run-sampling:
	gcloud run jobs execute $(JOB_SAMPLING) \
		--region=$(REGION) --project=$(PROJECT_ID) --wait \
		--args=sample.py,--source,$(SOURCE_URI),--output-prefix,$(STAGE_1_URI)

run-preprocessing:
	gcloud run jobs execute $(JOB_PREPROCESSING) \
		--region=$(REGION) --project=$(PROJECT_ID) --wait \
		--args=preprocess.py,--input,$(STAGE_1_URI)/canonical.jsonl,--input-manifest,$(STAGE_1_URI)/manifest.json,--output-dir,$(STAGE_2_URI)

run-anomaly:
	gcloud run jobs execute $(JOB_ANOMALY) \
		--region=$(REGION) --project=$(PROJECT_ID) --wait \
		--args=anomaly_detection.py,--input,$(STAGE_2_URI)/prepared.jsonl,--output-prefix,$(STAGE_3_URI)

run-labeling:
	gcloud run jobs execute $(JOB_LABELING) \
		--region=$(REGION) --project=$(PROJECT_ID) --wait \
		--args=label.py,--input,$(STAGE_2_URI)/prepared.jsonl,--input-manifest,$(STAGE_2_URI)/manifest.json,--config,$(LABEL_CONFIG),--output-prefix,$(STAGE_4_URI)

run-training:
	gcloud run jobs execute $(JOB_TRAINING) \
		--region=$(REGION) --project=$(PROJECT_ID) --wait \
		--args=train.py,--input,$(TRAIN_INPUT),--input-manifest,$(TRAIN_INPUT_MANIFEST),--config,$(TRAIN_CONFIG),--output-prefix,$(STAGE_5_URI),--model-path,$(MODEL_URI)

run-inference:
	gcloud run jobs execute $(JOB_INFERENCE) \
		--region=$(REGION) --project=$(PROJECT_ID) --wait \
		--args=inference.py,--input,$(STAGE_2_URI)/prepared.jsonl,--input-manifest,$(STAGE_2_URI)/manifest.json,--config,$(INFERENCE_CONFIG),--output-prefix,$(STAGE_6_URI)

run-metrics:
	gcloud run jobs execute $(JOB_METRICS) \
		--region=$(REGION) --project=$(PROJECT_ID) --wait \
		--args=compute_metrics.py,--input,$(STAGE_6_URI)/predictions.jsonl,--config,$(METRICS_CONFIG),--output-prefix,$(STAGE_7_URI),--run-id,$(RUN_ID)

# ============================================================
# Dashboard — separate Cloud Run Service, not a workflow stage
# ============================================================

dashboard-local:
	streamlit run dashboard.py

deploy-dashboard:
	gcloud run deploy ml-pipeline-dashboard \
		--image=$(IMAGE_URI) \
		--region=$(REGION) \
		--project=$(PROJECT_ID) \
		--command=streamlit \
		--args=run,dashboard.py,--server.port,8080,--server.address,0.0.0.0 \
		--service-account=$(SERVICE_ACCOUNT) \
		--port=8080 \
		--set-env-vars=ARTIFACT_BUCKET=$(ARTIFACT_BUCKET) \
		--no-allow-unauthenticated

# ============================================================
# Monitoring and cleanup
# ============================================================

status:
	gcloud run jobs list \
		--region=$(REGION) \
		--project=$(PROJECT_ID)

executions:
	@for job in $(JOBS); do \
		echo "=== $$job ==="; \
		gcloud run jobs executions list \
			--job=$$job \
			--region=$(REGION) \
			--project=$(PROJECT_ID); \
	done

logs:
	gcloud logging read \
		'resource.type="cloud_run_job" AND resource.labels.location="$(REGION)"' \
		--project=$(PROJECT_ID) \
		--limit=100 \
		--format="table(timestamp,severity,textPayload)"

clean-jobs:
	@for job in $(JOBS); do \
		gcloud run jobs delete $$job \
			--region=$(REGION) \
			--project=$(PROJECT_ID) \
			--quiet || true; \
	done