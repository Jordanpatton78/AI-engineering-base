Quality Requirements Document (QRD)
Modular AI Intelligence Framework · GCP Implementation · Version 1.1
1. Data Ingestion
1.1. Import datasets from common data sources and file formats
- Test 1.1.a — Supported source import: Import a known BigQuery table and supported GCS files (CSV, JSONL, Parquet). Verify that the expected records and fields are read correctly.
- Test 1.1.b — Unsupported format: Attempt to import an unsupported or malformed file. Verify that the system returns a clear error without reporting a successful import.
1.2. Support structured and unstructured data
- Test 1.2.a — Data type compatibility: Import a tabular dataset, a text dataset, and a dataset containing image references. Verify that each is represented correctly and routed to compatible downstream processing.
- Test 1.2.b — Unsupported data type: Provide an unregistered data type and verify that the system identifies the incompatibility.
1.3. Configure how incoming data is interpreted and used
- Test 1.3.a — Configuration application: Configure a dataset's schema, identifier field, and task. Verify that ingestion and subsequent stages use the configured values.
- Test 1.3.b — Invalid configuration: Provide a configuration with missing required fields or invalid settings. Verify that validation identifies the problem before processing begins.
1.4. Provide a consistent workflow regardless of source or domain
- Test 1.4.a — Connector equivalence: Load equivalent datasets from BigQuery and GCS. Verify that both produce compatible standardized dataset references and metadata.
- Test 1.4.b — Cross-domain workflow: Run two supported datasets from different domains through the shared orchestration interface. Verify that no domain-specific changes to the orchestrator are required.
1.5. Preserve original data
- Test 1.5.a — Source immutability: Record the source dataset's contents before execution, run ingestion, and verify that the source remains unchanged.
- Test 1.5.b — Snapshot traceability: Verify that any snapshot is stored separately and includes a reference to the original source and its version or identifying metadata.
2. Data Validation and Preparation
2.1. Evaluate completeness, consistency, validity, and quality
- Test 2.1.a — Quality rule coverage: Use a test dataset containing missing values, invalid types, inconsistent fields, and out-of-range values. Verify that the corresponding checks detect each seeded issue.
- Test 2.1.b — Clean dataset: Run validation against a dataset known to satisfy its rules. Verify that no false validation failures are reported.
2.2. Identify missing values, duplicates, unexpected formats, and invalid entries
- Test 2.2.a — Defect detection: Introduce known missing values, duplicate records, unexpected formats, and invalid entries. Verify that the report identifies the affected fields or records and correct counts.
2.3. Identify issues that could affect model performance or analysis
- Test 2.3.a — ML data risks: Test datasets containing class imbalance, constant features, high missingness, unexpected categories, and a deliberately introduced target-leakage pattern. Verify that applicable checks flag detectable issues.
- Test 2.3.b — Domain-specific checks: Add a dataset-specific validation rule and verify that it runs without changing the shared preparation engine.
2.4. Summarize issues and their potential impact
- Test 2.4.a — Quality report accuracy: Verify that the report contains the expected issue types, severity, prevalence, affected fields, and remediation suggestions.
- Test 2.4.b — Dashboard consistency: Verify that the dashboard's quality summaries agree with the saved validation report.
2.5. Configure how quality issues are handled
- Test 2.5.a — Rule actions: Configure the same validation rule with warn, reject, drop, and impute actions where applicable. Verify that each action produces its intended behavior.
- Test 2.5.b — Action traceability: Verify that applied transformations and rejected or removed records are recorded and that the original data remains unchanged.
2.6. Prepare data while preserving traceability
- Test 2.6.a — Transformation correctness: Apply known cleaning and feature extraction transformations. Compare the prepared output against expected results.
- Test 2.6.b — Record lineage: Verify that prepared records can be traced to their original records wherever the transformation permits.
- Test 2.6.c — Leakage prevention: Use a dataset with an extreme value or category appearing only in the test partition. Verify that fitted preprocessing parameters are learned from training data only when required by the evaluation design.
3. Model Selection and Execution
3.1. Support existing pretrained models without additional training
- Test 3.1.a — Inference-only mode: Configure an existing compatible model and disable training. Verify that inference can run without invoking model training.
- Test 3.1.b — Model loading failure: Configure an invalid or inaccessible model reference. Verify that the system reports the failure and does not proceed with an uninitialized model.
3.2. Support training new models on user-provided datasets
- Test 3.2.a — Training execution: Train a supported model on a known labeled dataset. Verify that training completes and produces a model artifact and training metadata.
- Test 3.2.b — Invalid training data: Provide insufficient or incompatible training data. Verify that the stage fails with an actionable explanation rather than saving an apparently valid model.
3.3. Select or configure models based on data type and task
- Test 3.3.a — Compatible selection: Select supported models for classification and regression tasks. Verify that the appropriate adapter and task configuration are used.
- Test 3.3.b — Unsupported combination: Select a model that does not support the specified task or input type. Verify that the mismatch is identified before execution.
3.4. Support multiple model types and prediction tasks through a common interface
- Test 3.4.a — Adapter conformance: Run two different model adapters through the same standardized interface. Verify that both return outputs conforming to the framework's prediction contract.
- Test 3.4.b — Regression testing: Add a new model adapter and verify that existing adapters and workflows still pass their regression tests.
3.5. Identify incompatible datasets or tasks
- Test 3.5.a — Schema mismatch: Supply a missing required feature, an incompatible data type, or an invalid target definition. Verify that execution is blocked with a clear compatibility error.
3.6. Configure model settings without changing the core framework
- Test 3.6.a — Configuration-only change: Change model parameters or select a different registered adapter through configuration. Verify that the intended change takes effect without modifying orchestration code.
- Test 3.6.b — Invalid parameter: Supply an unsupported parameter or out-of-range value. Verify that configuration validation rejects it or reports the supported constraints.
3.7. Record model configurations and execution results
- Test 3.7.a — Model metadata completeness: Verify that each model run records the model version, preprocessing version, configuration, input artifact references, code version, and execution status.
- Test 3.7.b — Reproducibility: Repeat a run using identical data, configuration, and model versions. Verify that outputs match within the defined tolerance for deterministic or stochastic execution.
3.8. Retrieval-augmented generation
- Test 3.8.a — Relevant retrieval: Index a corpus containing known passages and query with a passage's content. Verify that the expected chunk appears in the top-k results.
- Test 3.8.b — BigQuery vector store build: Build an index from a BigQuery table with known rows. Verify that every eligible row produces chunks with embeddings of the configured dimension and that the index records the embedding model version.
- Test 3.8.c — GCS document ingestion: Index text, Markdown, HTML, PDF, and JSONL objects from a GCS prefix, including one unsupported object. Verify that supported objects are parsed, chunked, and embedded, and that the unsupported object is listed as skipped in the index manifest.
- Test 3.8.d — Embedding model consistency: Query an index with a different encoder version or embedding dimension. Verify that retrieval is refused with a compatibility error.
- Test 3.8.e — Context budget: Retrieve more chunks than the configured context token budget allows. Verify that the prompt stays within budget, chunks are dropped in score order, and the number dropped is recorded.
- Test 3.8.f — Citation provenance: Verify that each chunk supplied to the model carries its chunk ID, source URI, and record identifier, and that these appear in the inference manifest.
- Test 3.8.g — Missing context policy: Run retrieval against an empty index with require_context set to true, then to false. Verify that the first case marks the affected record failed without contacting the model, and the second answers the question alone with a recorded warning and no fabricated context.
- Test 3.8.h — Retrieval disabled: Run inference with no retrieval block. Verify that its outputs match a baseline run and that no retrieval record is written.
- Test 3.8.i — Retrieval determinism: Repeat a retrieval with the same index, query, and k. Verify that the same chunk IDs are returned in the same order.
- Test 3.8.j — Source restriction: Configure retrieval over one permitted table and attempt a query against an unlisted table or prefix. Verify that access is refused and no content from the unlisted source is returned.
- Test 3.8.k — Non-language-model adapter: Attach a retrieval block to a classification adapter. Verify that the compatibility check rejects the configuration before execution.
3.9. Multi-agent analysis with LangChain
- Test 3.9.a — Supervisor routing: Give a supervisor agent an objective that requires data-quality and anomaly work. Verify that it delegates to the configured specialist agents and that each subtask is recorded with its assigned agent.
- Test 3.9.b — Tool allowlist: Have a specialist call a tool outside its allowlist. Verify that the call is refused, the refusal is recorded, and the agent stops or continues according to policy.
- Test 3.9.c — Read-only tools: Have an agent request a write, or a non-SELECT statement against BigQuery or a write to GCS. Verify that the request is rejected and source data is unchanged.
- Test 3.9.d — Execution limits: Configure a low step limit and a task that would otherwise loop. Verify that the agent stops with status limit_reached and that its partial trace is saved.
- Test 3.9.e — Trace completeness: Run a multi-agent analysis. Verify that every model reply, tool call with argument hash, and tool result summary appears in the agent trace, and that the inference manifest references the trace.
- Test 3.9.f — Findings provenance: Verify that each agent finding is labeled model_generated, lists the artifacts or chunk IDs it cites, and that every cited reference exists.
- Test 3.9.g — Model endpoint unavailable: Point the agent stage at an unreachable model endpoint. Verify that the stage fails or skips with a recorded reason and produces no findings.
- Test 3.9.h — Extension by configuration: Register a new specialist agent and tool through the extension interface and configuration. Verify that the new agent runs without changes to the supervisor code.
- Test 3.9.i — Failure isolation: Force the agent stage to fail. Verify that the outputs of earlier stages are unchanged and that the dashboard displays the failure.
4. Automated Labeling and Pseudo-Label Generation
4.1. Support labeled and unlabeled datasets
- Test 4.1.a — Label-state detection: Test fully labeled, partially labeled, and unlabeled datasets. Verify that each is assigned the correct label state and follows the configured workflow.
4.2. Generate pseudo-labels using an expert model when labels are unavailable
- Test 4.2.a — Pseudo-label generation: Run an unlabeled dataset through a configured expert model. Verify that labels are generated and linked to the correct records.
- Test 4.2.b — Conditional skipping: Disable pseudo-labeling or provide a dataset where it is unnecessary. Verify that the stage is explicitly marked as skipped and dependent stages receive valid inputs when available.
4.3. Configure the expert model
- Test 4.3.a — Expert model configuration: Change the expert model identifier and supported generation settings. Verify that the selected model and parameters are used.
- Test 4.3.b — Expert model incompatibility: Configure an unavailable model or incompatible input format. Verify that the stage fails or skips according to policy and records the reason.
4.4. Record pseudo-label source and confidence
- Test 4.4.a — Label provenance: Verify that each pseudo-label includes its record identifier, source model, version, and generated label.
- Test 4.4.b — Confidence availability: Test an expert model that provides confidence and one that does not. Verify that confidence is preserved when available and is not fabricated when absent.
4.5. Distinguish ground-truth labels from pseudo-labels
- Test 4.5.a — Ground-truth preservation: Provide ground-truth labels and conflicting expert predictions. Verify that original labels remain unchanged and both sources are distinguishable.
- Test 4.5.b — Provenance propagation: Verify that label source remains identifiable in training data, metrics, and dashboard outputs.
4.6. Review, filter, or exclude pseudo-labels
- Test 4.6.a — Confidence filtering: Configure a confidence threshold and verify that labels above and below the threshold are handled as specified.
- Test 4.6.b — Human review: Mark selected labels as reviewed, accepted, or rejected. Verify that training includes only labels permitted by the configured policy.
- Test 4.6.c — Filtering summary: Verify that counts of accepted, rejected, and unreviewed labels match the saved label artifact.
4.7. Communicate limitations of pseudo-label-based evaluation
- Test 4.7.a — Evaluation provenance: Evaluate predictions against pseudo-labels. Verify that results are identified as agreement with the expert model, not verified ground-truth accuracy.
- Test 4.7.b — Dashboard warning: Verify that the dashboard displays the appropriate qualification whenever evaluation relies on pseudo-labels.
5. Model Evaluation and Metrics
5.1. Calculate appropriate metrics for the task and available labels
- Test 5.1.a — Metric correctness: Run each registered evaluator on a small dataset with hand-calculated expected results. Verify the computed values within numerical tolerance.
- Test 5.1.b — Metric prerequisites: Remove a required label or prediction field. Verify that the metric is marked unavailable rather than calculated from invalid inputs.
5.2. Support classification, regression, anomaly detection, and other configurable tasks
- Test 5.2.a — Task coverage: Run representative test cases for each supported task and verify that its registered evaluator produces the expected metrics.
- Test 5.2.b — Extension compatibility: Register a new evaluator and verify that it works through the shared evaluation interface without modifying unrelated evaluators.
5.3. Compare performance against baselines, previous versions, or alternative models
- Test 5.3.a — Comparison correctness: Evaluate two models on the same dataset and compare their results against independently calculated metric differences.
- Test 5.3.b — Incomparable evaluations: Supply results from different evaluation datasets or label sources. Verify that the system identifies the difference and warns against unsupported comparisons.
5.4. Identify common errors and performance weaknesses
- Test 5.4.a — Error analysis: Use a labeled dataset with known misclassifications or regression errors. Verify that error summaries identify the expected error patterns.
- Test 5.4.b — Group breakdowns: Evaluate performance across configured subgroups and verify that the reported group-level metrics match independently calculated results.
5.5. Distinguish ground-truth-based from pseudo-label-based evaluation
- Test 5.5.a — Evaluation source labeling: Run evaluations against ground-truth and pseudo-label datasets. Verify that results preserve the correct source designation.
- Test 5.5.b — Display correctness: Verify that the dashboard distinguishes verified-label metrics from pseudo-label agreement.
5.6. Report insufficient data or labels
- Test 5.6.a — Insufficient sample: Run evaluation with too few records or insufficient class coverage for a selected metric. Verify that the metric reports its limitation.
- Test 5.6.b — Missing labels: Run evaluation without valid labels. Verify that unavailable metrics are explained and not reported as zero.
5.7. Retain evaluation results over time
- Test 5.7.a — Persistence: Complete multiple evaluations and verify that all results remain queryable in BigQuery with correct run and model identifiers.
- Test 5.7.b — Historical comparison: Retrieve results from earlier runs and verify that comparisons use the correct stored versions and evaluation conditions.
6. Anomaly and Out-of-Distribution Detection
6.1. Identify anomalous observations within a dataset
- Test 6.1.a — Known anomalies: Create a dataset containing deliberately seeded anomalies. Verify that the configured detector identifies the expected observations or ranks them appropriately according to its defined behavior.
- Test 6.1.b — Normal observations: Run a representative normal dataset through the detector and measure false-positive rates against an established test baseline.
6.2. Identify datasets or records that differ from a reference distribution
- Test 6.2.a — Known distribution shift: Create a target dataset with a controlled shift in one or more features. Verify that the detector identifies the affected features and reports a difference consistent with the selected method.
- Test 6.2.b — Stable distribution: Compare equivalent samples from the same distribution and verify that the system does not report a substantial shift beyond the configured tolerance without supporting evidence.
6.3. Detect potential out-of-distribution inputs
- Test 6.3.a — OOD inputs: Evaluate known in-distribution and deliberately shifted inputs using a compatible OOD detector. Verify that its scores or flags follow the expected behavior for the chosen test cases.
- Test 6.3.b — Unsupported OOD method: Configure an incompatible detector or omit required reference information. Verify that the stage reports the limitation rather than claiming successful OOD detection.
6.4. Configure reference datasets
- Test 6.4.a — Reference selection: Configure a known reference dataset and verify that the detector uses the correct artifact.
- Test 6.4.b — Incompatible reference: Supply a reference dataset with an incompatible schema or feature representation. Verify that comparison is blocked or handled through an explicitly supported alignment procedure.
6.5. Summarize anomalies, distribution differences, and significance
- Test 6.5.a — Summary correctness: Compare the saved detection summary against the underlying record-level scores, flags, and feature-level results. Verify that counts and reported findings agree.
- Test 6.5.b — Evidence traceability: Select a dashboard finding and verify that it links to the relevant detection artifact and supporting records.
6.6. Configure detection methods and thresholds
- Test 6.6.a — Threshold behavior: Run the same dataset with different configured thresholds. Verify that record flags or alerts change according to the detector's documented threshold behavior.
- Test 6.6.b — Method selection: Switch between two registered detectors using configuration alone. Verify that the selected detector is executed and its identity is recorded.
6.7. Distinguish statistical deviations from confirmed domain-specific events
- Test 6.7.a — Interpretation accuracy: Supply a statistically unusual record without evidence that it represents fraud or a defect. Verify that the output identifies an anomaly without asserting a confirmed real-world event.
- Test 6.7.b — Dashboard wording: Verify that anomaly scores and statistical warnings are labeled as model-generated findings, not verified causal conclusions.
6.8. Communicate missing references or unsupported detection
- Test 6.8.a — Missing reference: Enable a detector that requires reference data but omit the reference. Verify that the stage reports the missing dependency and records a skipped or failed status according to policy.
- Test 6.8.b — Dashboard propagation: Verify that the dashboard explains why detection was unavailable and does not display fabricated results.
7. Dashboard and Results Visualization
7.1. Provide a dashboard summarizing each workflow run
- Test 7.1.a — Dashboard completeness: Execute a complete workflow and verify that the dashboard contains run status, key findings, warnings, and references to detailed artifacts.
- Test 7.1.b — Partial workflow: Execute a workflow with optional stages skipped. Verify that the dashboard still renders and accurately represents the skipped stages.
7.2. Display dataset characteristics, quality findings, and processing status
- Test 7.2.a — Dataset summary accuracy: Compare displayed row counts, field summaries, and quality findings against stored dataset metadata and validation reports.
- Test 7.2.b — Stage status accuracy: Verify that displayed stage statuses match the corresponding workflow and job records.
7.3. Display model outputs, evaluation metrics, and comparisons
- Test 7.3.a — Prediction and metric display: Compare dashboard values against saved inference and metrics artifacts. Verify that displayed results match.
- Test 7.3.b — Missing results: Omit optional metrics or inference artifacts. Verify that the dashboard indicates unavailable results instead of displaying misleading default values.
7.4. Display anomalies, OOD findings, and supporting evidence
- Test 7.4.a — Detection display: Compare anomaly counts, score summaries, and feature-level findings with the underlying detection artifact.
- Test 7.4.b — Evidence links: Open a displayed anomaly and verify that its record identifier and supporting artifact correspond to the correct finding.
7.5. Distinguish findings, predictions, pseudo-labels, and verified results
- Test 7.5.a — Provenance presentation: Load examples of each result type and verify that the dashboard labels each according to its stored provenance.
- Test 7.5.b — Mixed-source data: Use a dataset containing both ground-truth and pseudo-labels. Verify that the interface does not conflate their sources.
7.6. Allow inspection of individual records and results
- Test 7.6.a — Record lookup: Select records by identifier and verify that their source references, predictions, labels, and anomaly results are retrieved correctly.
- Test 7.6.b — Access control: Attempt to inspect records without the required permissions. Verify that restricted data is not disclosed.
7.7. Adapt metrics and visualizations to the task and available data
- Test 7.7.a — Task-specific rendering: Generate dashboards for classification, regression, and anomaly detection. Verify that each uses appropriate charts and metrics.
- Test 7.7.b — Optional artifacts: Remove an optional stage artifact and verify that the remaining visualizations still render correctly.
7.8. Communicate errors, warnings, and limitations to nontechnical users
- Test 7.8.a — Error explanation: Trigger representative ingestion, model, and evaluation failures. Verify that the dashboard provides understandable descriptions of each failure.
- Test 7.8.b — Technical detail separation: Verify that user-facing explanations remain concise while technical details remain available in authorized logs.
7.9. Display retrieved sources and generated text
- Test 7.9.a — Generated-text labeling: Display an inference output and an agent finding. Verify that each is labeled as model-generated and shows the sources it cites.
- Test 7.9.b — Source drill-down: Select a citation in a displayed output. Verify that the referenced chunk and its source URI or record identifier are shown.
8. Workflow Configuration and Modularity
8.1. Configure workflows without modifying the core framework
- Test 8.1.a — Configuration-driven execution: Change the dataset, model, and task through configuration. Verify that the workflow executes the intended changes without modifications to orchestration code.
- Test 8.1.b — Configuration validation: Provide malformed or contradictory configuration values. Verify that the system rejects them with actionable validation errors.
8.2. Enable, disable, or replace individual workflow components
- Test 8.2.a — Stage toggling: Disable each optional stage in turn and verify that the workflow skips it while preserving valid downstream behavior.
- Test 8.2.b — Component replacement: Replace a registered detector or model adapter and verify that the new component executes without unintended effects on unrelated stages.
- Test 8.2.c — Dependency enforcement: Disable a stage whose output is required downstream. Verify that the workflow blocks the dependent operation unless a valid alternative input is configured.
8.3. Support domain-specific configurations while preserving shared functionality
- Test 8.3.a — Multiple profiles: Run two domain configurations against the same framework. Verify that domain-specific behavior changes as configured while shared orchestration and artifact handling remain consistent.
- Test 8.3.b — Configuration isolation: Verify that settings from one project or run do not unintentionally affect another.
8.4. Add extensions without redesigning the system
- Test 8.4.a — Extension registration: Add a new supported data connector, model adapter, evaluator, or detector through its documented extension interface. Verify that it can be selected and executed.
- Test 8.4.b — Regression suite: Run existing supported configurations after registering the extension and verify that their behavior remains unchanged.
8.5. Execute only relevant workflow stages
- Test 8.5.a — Minimal workflow: Configure a workflow requiring data preparation, pretrained-model inference, and reporting. Verify that unnecessary stages do not execute.
- Test 8.5.b — Required stage order: Inspect execution records and verify that all enabled stages execute in the specified order.
8.6. Skip unnecessary or unsupported stages
- Test 8.6.a — Skip status: Verify that every skipped stage records its status and reason in the run record and manifest.
- Test 8.6.b — Downstream compatibility: Verify that downstream stages proceed only when required artifacts are available, including when an upstream stage has been skipped.
9. Run Management and Reporting
9.1. Track each execution and its inputs, configurations, and outputs
- Test 9.1.a — Run record completeness: Execute a workflow and verify that its unique run ID, configuration, source references, stage statuses, timestamps, model identifiers, and artifact manifests are recorded.
- Test 9.1.b — Run isolation: Execute two runs concurrently. Verify that their records and artifacts remain associated with the correct run IDs.
9.2. Communicate progress, completion, and failures
- Test 9.2.a — Progress tracking: Execute a workflow with multiple stages and verify that the run record updates as each stage starts and completes.
- Test 9.2.b — Failure visibility: Force a Cloud Run Job to fail. Verify that the run status, failing stage, and relevant error details are recorded and made available to the dashboard.
9.3. Reproduce or compare previous runs
- Test 9.3.a — Repeat execution: Rerun a workflow using the same source references and configuration. Verify that the system retains both runs and their associated artifacts.
- Test 9.3.b — Configuration changes: Modify one configuration parameter and run again. Verify that the new run records the changed configuration and that comparisons identify relevant differences.
- Test 9.3.c — Artifact reuse: Enable reuse of previous stage outputs and verify that an artifact is reused only when its inputs, code, configuration, and dependencies satisfy the reuse criteria.
9.4. Export results, metrics, and findings
- Test 9.4.a — Export correctness: Export supported results in the configured formats and verify that the exported contents match the corresponding stored artifacts.
- Test 9.4.b — Export permissions: Attempt to export data without sufficient permissions. Verify that access is denied and no restricted information is returned.
9.5. Provide a consolidated run summary
- Test 9.5.a — Summary completeness: Verify that the consolidated report includes run status, key data quality findings, model results, applicable metrics, anomaly findings, and limitations.
- Test 9.5.b — Recommendation grounding: Verify that generated recommendations reference supporting rules or model outputs and are distinguished from verified findings.
10. Reliability and Responsible Use
10.1. Prevent invalid inputs from silently producing misleading results
- Test 10.1.a — Invalid input handling: Submit invalid schemas, incompatible model inputs, and missing required fields. Verify that the appropriate stage reports an error or warning and does not silently produce a successful result.
- Test 10.1.b — Dependent-stage blocking: Verify that a failed required stage prevents downstream stages from executing with invalid or missing artifacts.
10.2. Communicate uncertainty and model limitations
- Test 10.2.a — Uncertainty presentation: Test models with available confidence estimates and models without them. Verify that uncertainty is presented only when supported and that limitations are recorded.
- Test 10.2.b — Small-sample warning: Evaluate a deliberately small dataset and verify that the system reports relevant limitations without overstating the reliability of its results.
10.3. Maintain traceability across the full pipeline
- Test 10.3.a — End-to-end lineage: Select a dashboard metric or prediction and trace it back through the inference, preparation, and source artifacts. Verify that links and identifiers are consistent.
- Test 10.3.b — Label and model lineage: Verify that predictions and evaluation results identify the correct model version, preprocessing artifacts, label sources, and evaluation dataset.
10.4. Handle failures without corrupting data or falsely reporting success
- Test 10.4.a — Interrupted stage: Interrupt a Cloud Run Job during execution. Verify that incomplete outputs are not published as completed artifacts and that the run records the failure.
- Test 10.4.b — Retry safety: Force a transient failure and allow the workflow to retry. Verify that retries do not create duplicate logical outputs or overwrite valid artifacts from prior runs.
- Test 10.4.c — Manifest consistency: Force a failure during output persistence. Verify that the stage does not publish a completed manifest until required outputs have been persisted and validated.
10.5. Protect user data and control access
- Test 10.5.a — Least-privilege access: Verify that each service account can access only the required datasets, buckets, models, and other resources.
- Test 10.5.b — Secret protection: Verify that credentials are retrieved through Secret Manager or approved identity mechanisms and are not exposed in source code, logs, or error messages.
- Test 10.5.c — Data retention and deletion: Execute configured retention and deletion procedures in the test environment. Verify that covered data and artifacts are handled according to policy.
10.6. Distinguish statistical findings from verified explanations or events
- Test 10.6.a — Unsupported causal claims: Provide correlated but noncausal features and verify that the system does not present them as proven causes.
- Test 10.6.b — Event classification: Provide an anomaly without independent evidence of fraud, a defect, or a security incident. Verify that the output remains a potential finding rather than a confirmed event.
10.7. Label generated language and agent output
- Test 10.7.a — Unsupported generated claims: Ask a language model a question whose answer is absent from the retrieved context. Verify that the output is flagged as not supported by the context or as uncertain, and is not presented as a sourced fact.
- Test 10.7.b — Agent action boundaries: Verify that agents act only through tools in their allowlists and that no agent output alters a manifest or artifact written by another stage.
11. Pipeline Execution and Artifact Contract
11.1. Execute stages in the required order
- Test 11.1.a — Sequential execution: Run a complete workflow and verify that the eight Cloud Run Jobs execute in the specified order, with each stage starting only after the preceding stage completes successfully or returns an explicitly permitted skip status.
- Test 11.1.b — Artifact completeness: Verify that every stage produces its required output artifact or an explicit skip record, and that all artifacts are associated with the correct run ID.
- Test 11.1.c — Dashboard completion: Verify that the final stage produces the dashboard summary and references artifacts generated by all applicable upstream stages.
11.2. Persist and validate artifacts before advancing
- Test 11.2.a — Artifact contract: For every stage, verify that its inputs, outputs, configuration, status, and manifest conform to the documented artifact contract.
- Test 11.2.b — Persistence gate: Simulate an artifact write failure. Verify that the stage does not report successful completion and that Workflows does not advance to the next dependent stage.
- Test 11.2.c — Artifact integrity: Verify that stored artifact references resolve to readable objects or tables and that recorded metadata matches the persisted outputs.
11.3. Enforce conditional stage behavior
- Test 11.3.a — Pseudo-labeling condition: Test labeled and unlabeled datasets with pseudo-labeling enabled and disabled. Verify that the stage executes or skips according to configuration and label availability.
- Test 11.3.b — Training condition: Test training-enabled and inference-only configurations. Verify that training executes only when required and that inference uses the correct trained or existing model.
- Test 11.3.c — Metrics condition: Test datasets with valid ground-truth labels, pseudo-labels only, and no labels. Verify that the metrics stage calculates only valid metrics and documents unavailable ones.
- Test 11.3.d — Dashboard resilience: Skip an optional stage and verify that dashboard generation succeeds while accurately representing missing outputs.
- Test 11.3.e — Index requirement: Run inference with and without a retrieval block, with no index built. Verify that inference runs without an index when no retrieval block is set, and that a configured retrieval with no index fails the stage without writing predictions.
- Test 11.3.f — Agent stage condition: Run inference with and without an agents block. Verify that agent findings and trace artifacts exist only when the block is configured, and that the dashboard renders in both cases.
12. Recommended Implementation Decisions
12.1. Use one Cloud Run Job per stage initially
- Test 12.1.a — Stage isolation: Execute each stage independently using its declared inputs and configuration. Verify that it completes without relying on in-memory state from another job.
- Test 12.1.b — Independent deployment: Update one stage's implementation and verify that other stages do not require code changes when their interfaces remain unchanged.
12.2. Use GCS as the canonical artifact store and BigQuery for queryable tables
- Test 12.2.a — Storage routing: Run a workflow and verify that model files, non-tabular artifacts, reports, and manifests are saved in GCS, while configured queryable outputs are saved in BigQuery.
- Test 12.2.b — Cross-storage references: Verify that manifests correctly reference both GCS objects and BigQuery tables and that authorized downstream jobs can retrieve them.
12.3. Define a common artifact contract
- Test 12.3.a — Contract conformance: Run every registered stage and validate its manifest against the common artifact schema.
- Test 12.3.b — Interface compatibility: Replace a stage with another conforming implementation and verify that downstream stages continue to consume its outputs successfully.
- Test 12.3.c — Invalid artifact: Provide an artifact with missing required metadata or incompatible output fields. Verify that the receiving stage rejects it with a clear error.
12.4. Make training and pseudo-labeling conditional while preserving stage order
- Test 12.4.a — Conditional training: Run with training enabled and disabled. Verify that the workflow follows the configured mode while preserving the required stage sequence.
- Test 12.4.b — Conditional pseudo-labeling: Run with and without a need for pseudo-labels. Verify that the stage is executed or explicitly skipped as appropriate.
- Test 12.4.c — Model availability: Disable training without providing a compatible existing model. Verify that the workflow stops before inference rather than proceeding without a primary model.
12.5. Separate data-level anomalies from model-level OOD detection
- Test 12.5.a — Detection distinction: Use a dataset containing an unusual record that remains within the primary model's expected input distribution. Verify that data-level anomaly detection and model-level OOD detection are recorded as separate results.
- Test 12.5.b — Independent execution: Configure data-level anomaly detection without a model-specific OOD detector. Verify that the supported anomaly analysis can run independently.
- Test 12.5.c — Result interpretation: Verify that the dashboard distinguishes data anomalies, distribution shifts, and model-level OOD findings rather than merging them into a single unqualified status.
12.6. Use BigQuery as the vector store and LangChain for agent orchestration
- Test 12.6.a — Vector store routing: Build an index and run retrieval. Verify that chunk vectors are stored in BigQuery, that the chunk JSONL and index manifest are in GCS, and that retrieval does not depend on a local index file.
- Test 12.6.b — Framework isolation: Verify that LangChain is imported only by the agent and retrieval modules, so that the orchestration logic and the non-agent stages run without it installed.