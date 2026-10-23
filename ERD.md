Engineering Requirements Document (ERD)
Modular AI Intelligence Framework · GCP · Version 1.2

1. Data Ingestion
1.1. Import datasets from common data sources and file formats.

The ingestion job shall support BigQuery tables and GCS objects. It shall use source-specific connectors to read supported formats, including CSV, JSONL, and Parquet, and validate that the selected format can be parsed.

1.2. Support structured and unstructured data.

The ingestion interface shall identify the input data type and preserve records in a suitable representation. Structured data shall support tabular processing; text, images, and other unstructured data shall be stored or referenced in GCS and passed to compatible downstream processors.

1.3. Configure how incoming data is interpreted and used.

A run configuration shall specify the source location, file format or table, schema overrides, record identifiers, relevant columns, and intended task. The configuration shall be stored with the run artifacts.

1.4. Provide a consistent workflow regardless of source or domain.

A common ingestion interface shall return a standardized dataset reference and metadata object regardless of the source connector. Google Cloud Workflows shall invoke the same downstream stages using those standardized references.

1.5. Preserve original data.

The system shall treat source datasets as read-only. Any required snapshots shall be written to a separate, run-specific GCS location or BigQuery table, with source references and version information recorded in the manifest.

2. Data Validation and Preparation
2.1. Evaluate completeness, consistency, validity, and quality.

The data preparation job shall run configurable checks for missing values, schema conformity, valid ranges, data types, and consistency constraints. Results shall be saved as a structured quality report.

2.2. Identify missing values, duplicates, unexpected formats, and invalid entries.

The job shall execute reusable validation rules appropriate to the detected data type. Each finding shall include the rule violated, affected field or record, and count of affected records where measurable.

2.3. Identify issues that could affect model performance or analysis.

The job shall flag issues such as class imbalance, high missingness, constant features, inconsistent units, unexpected categories, and possible target leakage when detectable. Dataset-specific checks shall be configurable.

2.4. Summarize issues and their potential impact.

The job shall generate a machine-readable quality report containing issue severity, prevalence, affected fields, and recommended remediation. The dashboard shall display the report.

2.5. Configure how quality issues are handled.

The run configuration shall define per-rule actions, such as warn, reject, drop, impute, or transform. Rules shall support safe defaults, and the job shall record which actions were applied.

2.6. Prepare data while preserving traceability.

The preparation job shall apply configured transformations and feature extraction, then save the prepared data and transformation metadata as stage artifacts. Record identifiers and source-to-output mappings shall be retained where feasible. Fitted transformations shall be learned from training data only when required to avoid evaluation leakage.

3. Model Selection and Execution
3.1. Support existing pretrained models without additional training.

The model interface shall accept a configured model reference, such as a GCS model artifact or supported Vertex AI model identifier. The pipeline shall load the model and its required preprocessing dependencies without invoking the training stage.

3.2. Support training new models on user-provided datasets.

The training Cloud Run Job shall accept prepared data, eligible labels, and a training configuration. It shall execute the selected training adapter and persist the resulting model and training metadata.

3.3. Select or configure models based on data type and task.

The run configuration shall specify the task and model identifier. A model registry of supported adapters shall declare each model's supported input types, prediction tasks, label requirements, and inference interface. The framework may recommend compatible models, but shall not assume a suitable model exists for every dataset.

3.4. Support multiple model types and prediction tasks through a common interface.

Each model adapter shall implement standardized operations for loading, validating inputs, training when supported, and generating predictions. Task-specific behavior shall remain inside the adapter rather than the Workflows orchestration definition.

3.5. Identify incompatible datasets or tasks.

Before executing a model, the applicable stage shall compare the dataset schema, feature representation, label requirements, and task definition against the adapter's declared requirements. Incompatible configurations shall produce a clear validation error and stop dependent stages.

3.6. Configure model settings without changing the core framework.

Model parameters, thresholds, training options, and inference settings shall be supplied through versioned configuration. New supported models shall be integrated through an adapter registration mechanism rather than modifications to the central orchestration logic.

3.7. Record model configurations and execution results.

Every model execution shall record the model identifier and version, configuration, preprocessing version, input artifact references, code version, and execution outcome. Training and inference manifests shall reference these records.

3.8. Retrieval-augmented generation (RAG) for language models.

Retrieval is an optional step that runs before a language-model adapter generates output, in the inference stage and for agents. It is configured by a retrieval block and is disabled by default. Retrieval applies only to language-model adapters; a retrieval block on any other adapter shall fail validation under 3.5.

Sources. The retrieval block shall list the sources to index: BigQuery tables, read through an explicit column list, and GCS prefixes containing .txt, .md, .html, .pdf, or .jsonl objects. Each BigQuery row becomes one document built from its configured text fields and keeps its record identifier. GCS objects are parsed with the existing text, pypdf, and BeautifulSoup dependencies. Unsupported or unparseable objects shall be listed as skipped in the index manifest, not dropped silently.

Chunking. Documents are split into chunks of a configured token length (default 256) with a configured overlap (default 32). A chunk ID is derived from source URI, record identifier or byte range, chunk position, and a content hash, so re-indexing unchanged content produces the same IDs.

Embeddings. Chunks and queries are embedded with the same BERT-family encoder, configured as a model reference: a sentence-embedding model loaded with Hugging Face transformers, or a Vertex AI endpoint. Embeddings are the mean-pooled last hidden state, L2-normalized. The index manifest records the model reference, model version, and output dimension. A query whose encoder version or dimension differs from the index shall be refused.

Vector store. The index build is an optional part of the data preparation stage (Stage 2 in 11.1). It writes one row per chunk to a BigQuery table that serves as the vector database, with columns chunk_id, run_id, source_uri, record_id, chunk_text, chunk_position, embedding (`ARRAY<FLOAT64>`), embedding_model, and indexed_at. Large tables shall also receive a BigQuery vector index. A JSONL copy of the chunks and the index manifest are written to GCS under the run prefix, so the index can be rebuilt or audited without BigQuery.

Retrieval. The query text is embedded with the same encoder. The stage retrieves the top k chunks (default 5) by cosine distance using BigQuery VECTOR_SEARCH, optionally filtered by source or run identifier, and discards chunks below a configured minimum similarity. Exact (brute-force) search is used when no vector index exists. Each retrieval records the query hash, k, similarity scores, chunk IDs, and index version.

Context assembly. Retrieved chunks are inserted into a versioned prompt template in descending score order until the configured context token budget is reached. Each inserted chunk carries its chunk ID so the model can cite it. Chunks dropped for budget are counted in the manifest. If no chunk passes the similarity threshold, the record follows the configured policy: when require_context is true (the default), it is marked failed and is not sent to the model; when false, the model answers the question alone and a warning is recorded. Context shall never be fabricated.

Access. Retrieval reads only the sources named in its block. The service account needs read access to those sources and write access to the index dataset only.

3.9. Multi-agent analysis with LangChain.

Multi-agent analysis lets a configured set of agents perform several analysis tasks over one dataset context in a single run. It is implemented with langchain-core: the chat-model wrapper, messages, prompt templates, and tools. The supervisor and specialist loops are plain Python in agents.py, so LangGraph is not required. Each agent's chat model wraps a language-model adapter from the registry in 3.4, so any registered language-model adapter can serve an agent.

Roles. A supervisor agent receives the objective and a summary of the run manifest, breaks the objective into subtasks, and routes each subtask to a specialist agent. Specialists are declared in configuration with a name, instruction reference, model reference, tool allowlist, and execution limits. Example specialists for an industry profile are a data-quality reviewer, an anomaly explainer, a metrics reviewer, and a report writer.

Tools. Tools are registered Python functions with typed input schemas. The initial set is read-only: bigquery_select (SELECT statements only, restricted to allowed datasets and row-limited), read_run_artifact (reads manifests and reports under the run prefix), retrieve_context (3.8), and describe_table (descriptive statistics). No tool writes to source datasets. The only write is the agent findings artifact.

Limits. Each agent has a maximum step count, a maximum tool-call count, and a wall-clock timeout. Reaching a limit stops that agent with status limit_reached and keeps its partial trace.

Trace. Every model reply and tool call (name, argument hash, result summary) shall be appended to a JSONL agent trace. The stage writes the trace next to its findings, and the inference manifest references both. Model output is not guaranteed to replay exactly, so the trace is the audit record.

Findings. Agents write a findings artifact in which each finding carries its text, a provenance of model_generated, and the artifacts and chunk IDs it cites. Citations that do not match a tool result are listed as unverified. No confidence value is recorded, because the model adapters do not report one consistently. Recommendations follow 9.5 and are marked as recommendations.

Execution. Agents run inside the inference stage (Stage 6 in 11.1), after predictions are written and only when an agents block is configured. A failure is recorded in the manifest and does not alter the predictions. New specialists and tools are added through the extension interfaces in 8.4 and selected by configuration.

4. Automated Labeling and Pseudo-Label Generation
4.1. Support labeled and unlabeled datasets.

The data preparation stage shall detect configured label fields and determine whether usable labels are present. The workflow configuration shall distinguish labeled, partially labeled, and unlabeled datasets so that downstream stages can select appropriate behavior.

4.2. Generate pseudo-labels using an expert model when labels are unavailable.

When pseudo-labeling is enabled and appropriate, a Cloud Run Job shall load the configured expert model, process eligible records, and write generated labels to a separate artifact. The workflow shall skip this stage when pseudo-labeling is unnecessary or disabled.

4.3. Configure the expert model.

The run configuration shall specify the expert model reference, version, input format, output format, task, and applicable generation parameters. A common expert-model adapter shall normalize outputs into a consistent labeling schema.

4.4. Record pseudo-label source and confidence.

Each generated label shall include its record identifier, predicted label, expert model identifier and version, and confidence or uncertainty information when the model provides it. The system shall not invent confidence scores for models that do not provide a meaningful estimate.

4.5. Distinguish ground-truth labels from pseudo-labels.

Labels shall include a provenance field identifying their source, such as ground_truth, pseudo_label, or human_reviewed. Ground-truth labels shall be preserved separately and shall not be silently overwritten by generated labels.

4.6. Review, filter, or exclude pseudo-labels.

The system shall support configurable confidence thresholds, filtering rules, and an optional review workflow. Eligible pseudo-labels shall be explicitly selected for training or evaluation through configuration. The dashboard shall summarize accepted, rejected, and unreviewed labels.

4.7. Communicate limitations of pseudo-label-based evaluation.

The metrics stage shall identify whether its reference labels are ground truth or pseudo-labels. Results measured against pseudo-labels shall be reported as model agreement or pseudo-label-based estimates, not verified real-world accuracy.

5. Model Evaluation and Metrics
5.1. Calculate appropriate metrics for the task and available labels.

The metrics job shall use a task-specific evaluator selected through configuration. Each evaluator shall declare its required inputs and supported metrics and shall calculate only metrics whose prerequisites are satisfied.

5.2. Support classification, regression, anomaly detection, and other configurable tasks.

The evaluation framework shall provide registered metric implementations for supported tasks. New tasks shall be added through evaluator modules that conform to the shared interface rather than by rewriting the metrics stage.

5.3. Compare performance against baselines, previous versions, or alternative models.

The evaluator shall accept predictions and evaluation results from multiple model runs, align them using consistent evaluation data where possible, and calculate comparable metrics. Comparisons shall record differences in datasets, labels, and evaluation conditions that could affect interpretation.

5.4. Identify common errors and performance weaknesses.

Where labels and predictions are available, the evaluator shall generate error summaries, such as confusion matrices, residual distributions, error categories, and performance breakdowns by configured groups. Supported explanations shall be stored as evaluation artifacts.

5.5. Distinguish ground-truth-based from pseudo-label-based evaluation.

Every evaluation result shall include the reference-label source, evaluation dataset identifier, and metric methodology. The dashboard shall visually distinguish verified-label evaluation from comparisons against model-generated labels.

5.6. Report insufficient data or labels.

The evaluator shall check sample counts, label availability, class coverage, and metric-specific requirements. When a metric cannot be calculated reliably or at all, the system shall report the limitation rather than returning a misleading score.

5.7. Retain evaluation results for comparison over time.

Each metrics job shall write a versioned evaluation artifact and persist queryable results in BigQuery. Records shall be linked to the run, model version, evaluation dataset, and configuration.

6. Anomaly and Distribution-Shift Detection
6.1. Identify anomalous observations within a dataset.

The detection job shall support registered anomaly detection methods, including statistical rules and unsupervised models. It shall produce record-level scores or flags when the selected method supports them.

6.2. Identify datasets or records that differ from a reference distribution.

The job shall compare the target dataset against a configured reference using appropriate feature-level statistical tests, distribution distances, or learned representations. Dataset-level summaries and record-level results shall be produced where supported.

6.3. Detect potential out-of-distribution inputs.

The system shall support OOD detectors compatible with the selected model and input representation. Detection results shall identify the method used and its applicable reference distribution. The framework shall not claim OOD detection is available for every model by default.

6.4. Configure reference datasets.

The run configuration shall accept a reference dataset or artifact identifier, such as training data or a designated baseline. The detection job shall verify schema and representation compatibility before comparing distributions.

6.5. Summarize anomalies, distribution differences, and significance.

The detection stage shall produce structured summaries containing anomaly counts, score distributions, affected features, detected shifts, and supporting evidence when available. These summaries shall be included in the dashboard artifact.

6.6. Configure detection methods and thresholds.

The run configuration shall select the detector, relevant parameters, and alert thresholds. Each detector adapter shall validate its settings and record them in the stage manifest.

6.7. Distinguish statistical deviations from confirmed domain-specific events.

Anomaly outputs shall be identified as statistical or model-generated findings. The system shall not automatically classify a deviation as fraud, a confirmed defect, or a security incident without suitable domain-specific evidence and validation.

6.8. Communicate missing references or unsupported detection.

Before execution, the detection job shall verify that the required reference data and model components are available. If the selected method cannot run, the stage shall record a warning or a skipped status with the reason, rather than fabricate a result.

7. Dashboard and Results Visualization
7.1. Provide a dashboard summarizing each workflow run.

The dashboard-generation job shall consume the stage manifests and outputs and produce a consolidated view of the run. It shall include overall status, key results, warnings, and links or references to detailed artifacts.

7.2. Display dataset characteristics, quality findings, and processing status.

The dashboard shall read dataset metadata and quality reports to display row counts, data types, missingness, validation issues, and stage completion status. It shall not require direct access to raw source data to display summary information.

7.3. Display model outputs, evaluation metrics, and comparisons.

The dashboard shall load inference and evaluation artifacts and present task-appropriate predictions, metrics, and baseline comparisons. Unavailable results shall be identified as unavailable, not displayed as zero.

7.4. Display anomalies, OOD findings, and supporting evidence.

The dashboard shall display anomaly summaries, distribution comparisons, relevant feature-level findings, and record-level results when available. Detailed results shall reference the corresponding stored artifacts.

7.5. Distinguish findings, predictions, pseudo-labels, and verified results.

The dashboard data model shall preserve output provenance and status. Visual labels and descriptions shall distinguish predictions, generated labels, ground-truth-based metrics, and statistical warnings.

7.6. Allow inspection of individual records and results.

The dashboard shall support filtering and viewing individual records by stable record identifier. Record detail views shall retrieve associated source references, predictions, labels, and anomaly results, subject to access permissions.

7.7. Adapt metrics and visualizations to the task and available data.

The dashboard shall use registered visualization templates selected according to the task and available artifacts. Task-specific modules may provide additional charts, but missing optional artifacts shall not prevent the rest of the dashboard from rendering.

7.8. Communicate errors, warnings, and limitations to nontechnical users.

The dashboard shall translate stage errors and validation findings into concise descriptions of what happened, its likely impact, and the next action where known. Technical logs shall remain available separately for debugging.

8. Workflow Configuration and Modularity
8.1. Configure workflows without modifying the core framework.

Each run shall be driven by a validated configuration file or equivalent configuration record. It shall specify source data, task type, model choices, stage settings, and output destinations. Google Cloud Workflows shall interpret the configuration and pass relevant parameters to Cloud Run Jobs.

8.2. Enable, disable, or replace individual workflow components.

Each stage shall have an explicit execution setting and component identifier. The orchestrator shall support skipping optional stages and selecting alternative registered adapters. Required downstream dependencies shall be validated before skipped stages are bypassed.

8.3. Support domain-specific configurations while preserving shared functionality.

Company- or industry-specific behavior shall be defined in separate configuration profiles and adapter modules. Shared ingestion, artifact management, orchestration, metrics infrastructure, and dashboard framework shall remain reusable across profiles.

8.4. Add data sources, model types, evaluation methods, and detection methods without redesigning the system.

The framework shall expose documented extension interfaces for source connectors, preparation adapters, model adapters, evaluators, and detectors. Each extension shall declare its capabilities, inputs, outputs, and configuration schema. The core workflow shall invoke registered components through these interfaces.

8.5. Execute only relevant workflow stages.

The configuration shall allow the user to define the intended workflow objective and enabled stages. Workflows shall execute only the necessary stages while retaining the required order and validating all dependencies.

8.6. Skip unnecessary or unsupported stages.

Each stage shall return an explicit status of completed, skipped, or failed, along with a reason where applicable. Downstream stages shall determine whether they can proceed based on their required inputs. The system shall not treat a skipped stage as a successful computation.

9. Run Management and Reporting
9.1. Track each execution and its inputs, configurations, and outputs.

Every run shall receive a unique run identifier. A run record in BigQuery shall track the configuration version, source references, stage statuses, timestamps, model identifiers, and artifact manifests.

9.2. Communicate progress, completion, and failures.

Google Cloud Workflows shall track the execution sequence and update the run record after each stage. Cloud Run Job status and Cloud Logging shall provide execution details. The dashboard or run summary shall display the current state and any failure information.

9.3. Reproduce or compare previous runs.

The system shall retain versioned configuration, code identifiers, source references, preprocessing artifacts, model references, and stage outputs. A rerun shall use the recorded inputs and configuration where those resources remain available. Comparisons shall identify differences between runs that could affect results.

9.4. Export results, metrics, and findings.

The system shall provide exportable reports and structured outputs. GCS shall store downloadable artifacts, such as JSON reports, CSV files, and supported dashboard exports. BigQuery tables shall remain available for querying and downstream analysis according to access permissions.

9.5. Provide a consolidated run summary.

The dashboard-generation job shall assemble the run status, major data quality findings, model results, anomaly findings, evaluation metrics, and warnings into a single report. Recommendations shall be generated only where supported by the configured rules or models and shall be distinguished from verified findings.

10. Reliability and Responsible Use
10.1. Prevent invalid inputs from silently producing misleading results.

Each stage shall validate its inputs against its declared requirements before processing. Invalid configurations, missing required fields, incompatible feature representations, and unavailable model dependencies shall produce explicit errors or warnings. The workflow shall stop dependent stages when their prerequisites are not satisfied.

10.2. Communicate uncertainty and model limitations.

Model outputs shall retain available confidence or uncertainty estimates, and evaluation results shall record relevant limitations. The dashboard shall communicate when results are based on small samples, weak reference data, pseudo-labels, unsupported assumptions, or incomplete evaluation.

10.3. Maintain traceability across the full pipeline.

Every stage manifest shall reference its input artifacts and output artifacts. Stable record identifiers, label provenance, model versions, preprocessing configurations, and evaluation dataset references shall be preserved where applicable. These references shall allow users to trace a reported result back to its source.

10.4. Handle failures without corrupting data or falsely reporting success.

Stage jobs shall write outputs to run-specific locations and publish their manifests only after outputs have been successfully persisted and validated. Workflows shall verify job completion before advancing. Retries shall be designed to avoid duplicate logical results or overwriting completed outputs.

10.5. Protect user data and control access.

GCP IAM shall restrict access to source datasets, model artifacts, and reports using least-privilege service accounts. Sensitive credentials shall be stored in Secret Manager. BigQuery permissions, GCS access controls, logging policies, and retention settings shall be configured to meet the data owner's requirements.

10.6. Distinguish statistical findings from verified explanations or events.

The framework shall label model predictions, anomaly scores, correlations, and generated explanations according to their actual evidentiary status. Any causal claim or confirmed domain-specific conclusion shall require appropriate validation beyond a statistical deviation or model output.

11. Pipeline Execution and Artifact Contract
This section defines how the numbered requirements above are assembled into the requested eight-stage GCP workflow. Two stages have optional components: the vector index in Stage 2 and agent analysis in Stage 6. It supplements the PRD traceability sections rather than introducing additional product requirements.
11.1. Required execution order
Order	Stage	Required output
1	Sample data	Sampled dataset and sampling manifest
2	Data preparation / feature extraction (optional vector index)	Prepared data, feature schema, transformations; vector table, chunk JSONL, and index manifest when enabled
3	Pseudo-labeling (conditional)	Generated labels and provenance, or explicit skip result
4	Anomaly / OOD detection (conditional)	Scores, flags, distribution analysis, or skip result
5	Model training (conditional)	Model artifact or selected existing-model reference, plus training metadata or skip result
6	Inference (retrieval-augmented and agent analysis when configured)	Predictions, inference metadata, retrieval records, and agent findings and trace when enabled
7	Metrics	Evaluation report and structured metrics
8	Dashboard generation	Dashboard summary, visualization outputs, and references to supporting artifacts

11.2. Artifact persistence contract
Every stage shall follow the same execution pattern:
1. Receive the run ID, configuration, and upstream artifact references.
2. Validate the required inputs.
3. Execute the configured stage implementation.
4. Persist the stage outputs to GCS and/or BigQuery.
5. Validate that the outputs were successfully written.
6. Write the stage manifest with output references, status, warnings, and execution metadata.
7. Return the status and manifest reference to Google Cloud Workflows.
The next stage shall not begin until the preceding stage has returned a valid completion or an explicitly permitted skip status. If a required stage fails, dependent stages shall not run.
11.3. Important execution rules
- Stage 3: Pseudo-labeling is conditional on the availability and suitability of labels and the workflow configuration.
- Stage 4: Anomaly/OOD detection is conditional on the availability of a compatible method and any required reference data.
- Stage 2: The vector index build runs only when a retrieval block is configured. Otherwise it records a skip. The index manifest records the embedding model version and output dimension.
- Stage 5: Training is conditional. When disabled, the workflow shall select a compatible existing primary model instead.
- Stage 6: Inference that requires retrieval shall not send records to the model when the index is failed or skipped; those records are marked failed (3.8). Agent analysis runs only when an agents block is configured. Its failure does not alter the predictions.
- Stage 7: Metrics shall be computed only when their prerequisites are satisfied. The stage shall still produce a report explaining unavailable metrics.
- Stage 8: Dashboard generation shall use all available stage artifacts, including explicit skipped-stage and failure information.
Acceptance criterion: A successful run must leave a traceable artifact or explicit skip record for every stage, with each completed stage's outputs persisted before the workflow advances. This is the central engineering contract that makes the framework reproducible and modular across different datasets and industries.