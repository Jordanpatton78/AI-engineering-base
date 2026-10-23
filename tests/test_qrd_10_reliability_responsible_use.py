"""QRD tests for Section 10: Reliability and Responsible Use.

Traceability: QRD 10.1 through 10.6. Fault injection (interrupts, retries, write
failures), IAM, Secret Manager, retention, and dashboard lineage need the deployed
Cloud Run environment or the dashboard, so those tests are pending.
"""
from __future__ import annotations

import json
import re

import rag
from conftest import (
    ROOT,
    TRAIN,
    FakeOllamaServer,
    find_value,
    labeled_rows,
    ollama_config,
    pending,
    read_jsonl,
    run_anomaly,
    run_inference_stage,
    run_train,
    write_jsonl,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_10_1_a_invalid_input_reports_an_error_without_a_success_manifest(tmp_path, run_cli):
    """QRD 10.1.a: a schema with a missing required field produces an error and no successful result."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "invalid_schema", "--target", "absent_column")

    assert result.returncode != 0
    manifest_path = out / "manifest.json"
    assert not manifest_path.exists() or read_json(manifest_path)["status"] != "succeeded"


def test_10_1_b_failed_upstream_manifest_blocks_training(tmp_path, run_cli):
    """QRD 10.1.b: training does not run on the artifacts of a failed upstream stage."""
    upstream = tmp_path / "upstream_manifest.json"
    upstream.write_text(json.dumps({"status": "failed", "outputs": {}}), encoding="utf-8")
    input_path = write_jsonl(tmp_path / "blocked_input.jsonl", labeled_rows(40))
    out = tmp_path / "blocked"
    out.mkdir()

    result = run_cli(
        TRAIN, "--input", input_path, "--input-manifest", upstream, "--target", "label", "--task", "classification",
        "--model-path", out / "model.joblib", "--metrics", out / "metrics.json",
        "--predictions", out / "predictions.jsonl", "--manifest", out / "manifest.json",
    )

    assert result.returncode != 0
    assert not (out / "model.joblib").exists()


@pending("Uncertainty presentation is a dashboard behavior and the dashboard is not implemented")
def test_10_2_a_uncertainty_is_shown_only_when_the_model_provides_it():
    """QRD 10.2.a: uncertainty appears for models that supply it and is absent for those that do not; limitations are recorded."""
    raise NotImplementedError


def test_10_2_b_small_sample_training_records_a_limitation(tmp_path, run_cli):
    """QRD 10.2.b: training on a deliberately small dataset records a limitation about reliability."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(8), "small_sample")

    assert result.returncode == 0, result.stderr
    assert find_value(read_json(out / "manifest.json"), "limitations")


@pending("End-to-end lineage needs a dashboard and a full multi-stage run, neither of which runs locally")
def test_10_3_a_dashboard_metric_traces_back_to_source_artifacts():
    """QRD 10.3.a: a dashboard metric or prediction traces through inference, preparation, and source artifacts with consistent identifiers."""
    raise NotImplementedError


def test_10_3_b_training_manifest_identifies_its_input_dataset(tmp_path, run_cli):
    """QRD 10.3.b: the training manifest identifies the input dataset it was trained on."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "lineage", "--run-id", "qrd-10-3-b")

    assert result.returncode == 0, result.stderr
    manifest = read_json(out / "manifest.json")
    assert "lineage_input.jsonl" in json.dumps(manifest["input"])
    assert manifest["run_id"] == "qrd-10-3-b"


@pending("Interrupting a Cloud Run Job requires a deployed job")
def test_10_4_a_interrupted_stage_publishes_no_completed_artifacts():
    """QRD 10.4.a: an interrupted stage publishes no completed artifacts and the run records the failure."""
    raise NotImplementedError


@pending("Retry behavior depends on Cloud Run retries, which cannot run locally")
def test_10_4_b_retries_do_not_duplicate_or_overwrite_valid_artifacts():
    """QRD 10.4.b: a retried transient failure creates no duplicate logical outputs and overwrites no valid prior artifacts."""
    raise NotImplementedError


@pending("Output-persistence fault injection is not supported by the stage scripts")
def test_10_4_c_manifest_is_not_published_before_outputs_are_persisted():
    """QRD 10.4.c: a failure during output persistence leaves no completed manifest."""
    raise NotImplementedError


@pending("Service-account least-privilege is configured in GCP IAM and cannot be checked locally")
def test_10_5_a_each_service_account_reaches_only_its_required_resources():
    """QRD 10.5.a: each service account can access only the datasets, buckets, models, and resources it needs."""
    raise NotImplementedError


def test_10_5_b_no_literal_credentials_in_pipeline_source():
    """QRD 10.5.b: the pipeline source contains no hard-coded credentials; they come from the environment or a secret store."""
    secret_pattern = re.compile(r"""(api[_-]?key|secret|token|password)\s*[:=]\s*["'][A-Za-z0-9_\-]{12,}["']""", re.IGNORECASE)
    offenders = [
        path.name
        for path in (ROOT / "pipeline").glob("*.py")
        if secret_pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


@pending("Retention and deletion procedures are not implemented")
def test_10_5_c_retention_and_deletion_follow_policy_in_test_environment():
    """QRD 10.5.c: configured retention and deletion procedures remove covered data and artifacts according to policy."""
    raise NotImplementedError


@pending("Causal-claim checks need an explanation output that does not exist yet")
def test_10_6_a_correlated_features_are_not_presented_as_causes():
    """QRD 10.6.a: correlated but noncausal features are not presented as proven causes."""
    raise NotImplementedError


def test_10_6_b_anomaly_without_evidence_is_not_reported_as_a_confirmed_event(tmp_path, run_cli):
    """QRD 10.6.b: an anomaly with no independent evidence is described as a potential finding, not a confirmed event."""
    rows = [{"id": i, "x": (i % 6) * 0.1, "y": (i // 6) * 0.1} for i in range(30)] + [{"id": 30, "x": 100.0, "y": 100.0}]

    result, out = run_anomaly(run_cli, tmp_path, rows, "no_evidence")

    assert result.returncode == 0, result.stderr
    text = json.dumps(read_json(out / "anomaly_report.json")).lower()
    assert "confirmed" not in text
    assert "security incident" not in text


def test_10_7_a_prompt_requires_grounding_and_admits_missing_answers():
    """QRD 10.7.a (prompt level): the prompt tells the model to use only the context and to say when the answer is absent."""
    prompt = rag.render_prompt("What is the churn rate?", [{"chunk_id": "c1", "chunk_text": "revenue grew"}])

    assert "using only the context" in prompt
    assert "say so" in prompt
    assert "[c1] revenue grew" in prompt


@pending("Whether a live model stays within retrieved context needs a labeled evaluation set and a model endpoint")
def test_10_7_a_model_answers_stay_within_retrieved_context():
    """QRD 10.7.a (behavior): on questions with no supporting context, answers are flagged as unsupported or uncertain."""
    raise NotImplementedError


def test_10_7_b_agent_run_writes_only_its_own_artifacts_and_leaves_predictions_unchanged(tmp_path):
    """QRD 10.7.b: agents write only their findings and trace; the prediction output is unchanged by the agent stage."""
    with FakeOllamaServer(["prediction answer", '{"next": "FINISH", "summary": "Nothing to flag."}']) as server:
        config = ollama_config(server, agents={
            "objective": "Summarize.",
            "specialists": [{"name": "quality", "instructions": "Read.", "tools": ["read_artifact"]}],
        })
        code, out = run_inference_stage(tmp_path, config, [{"id": "q1", "text": "question"}])

    assert code == 0
    predictions = read_jsonl(out / "predictions.jsonl")
    assert predictions[0]["prediction"]["text"] == "prediction answer"
    written = {path.name for path in out.iterdir()}
    assert written == {
        "predictions.jsonl", "inference_report.json", "manifest.json",
        "agent_findings.json", "agent_trace.jsonl",
    }
