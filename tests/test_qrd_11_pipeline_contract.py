"""QRD tests for Section 11: Pipeline Execution and Artifact Contract.

Traceability: QRD 11.1 through 11.3. Stage ordering and conditions are read from
pipeline/workflow.yaml. Manifest and output checks run the real train, anomaly,
and metrics scripts. Fault injection and the dashboard stage are pending.

Note: the QRD says eight Cloud Run Jobs. pipeline/workflow.yaml calls seven stage
jobs; the dashboard is deployed as a separate Cloud Run Service, not a workflow step.
"""
from __future__ import annotations

import json
from pathlib import Path

from conftest import (
    FakeOllamaServer,
    find_value,
    labeled_rows,
    ollama_config,
    pending,
    run_anomaly,
    run_inference_stage,
    run_metrics,
    run_train,
    workflow_condition_for,
    workflow_jobs,
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def output_paths(node):
    """Yield every value stored under a 'path' key in a manifest's outputs section."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "path" and isinstance(value, str):
                yield value
            else:
                yield from output_paths(value)
    elif isinstance(node, list):
        for item in node:
            yield from output_paths(item)


def test_11_1_a_seven_stage_jobs_run_in_the_required_order():
    """QRD 11.1.a: the stage jobs run in the required order, each after its predecessor."""
    assert workflow_jobs() == [
        "ml-stage-1-ingestion",
        "ml-stage-2-preprocess",
        "ml-stage-3-anomaly-detection",
        "ml-stage-4-labeling",
        "ml-stage-5-training",
        "ml-stage-6-inference",
        "ml-stage-7-compute-metrics",
    ]


@pending("Artifact completeness across every stage needs a full local or deployed run")
def test_11_1_b_every_stage_produces_its_output_or_a_skip_record():
    """QRD 11.1.b: every stage writes its required output or an explicit skip record, tagged with the run ID."""
    raise NotImplementedError


@pending("The final dashboard stage is deployed as a separate service and cannot be checked locally")
def test_11_1_c_dashboard_references_artifacts_from_upstream_stages():
    """QRD 11.1.c: the final dashboard summary references artifacts from all applicable upstream stages."""
    raise NotImplementedError


def test_11_2_a_stage_manifests_share_the_contract_fields(tmp_path, run_cli):
    """QRD 11.2.a: manifests from different stages carry the shared contract fields."""
    train, train_out = run_train(run_cli, tmp_path, labeled_rows(40), "contract_train", "--run-id", "contract-1")
    anomaly, anomaly_out = run_anomaly(run_cli, tmp_path, [{"id": i, "x": float(i), "y": float(i)} for i in range(12)], "contract_anomaly")

    assert train.returncode == 0, train.stderr
    assert anomaly.returncode == 0, anomaly.stderr
    contract = {"run_id", "status", "stage"}
    assert contract <= set(read_json(train_out / "manifest.json"))
    assert contract <= set(read_json(anomaly_out / "manifest.json"))


@pending("A stage that fails persistence validation can only be simulated with fault injection")
def test_11_2_b_failed_artifact_write_prevents_success_and_advancement():
    """QRD 11.2.b: a simulated artifact write failure stops the stage from reporting success and blocks the next stage."""
    raise NotImplementedError


def test_11_2_c_manifest_output_paths_resolve_to_readable_files(tmp_path, run_cli):
    """QRD 11.2.c: every output path recorded in the manifest exists and is readable."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "integrity")

    assert result.returncode == 0, result.stderr
    paths = list(output_paths(read_json(out / "manifest.json")["outputs"]))
    assert paths
    for path in paths:
        assert Path(path).is_file(), path


def test_11_3_a_pseudo_labeling_runs_only_when_the_flag_is_set():
    """QRD 11.3.a: the labeling stage is gated by run_labeling in the workflow."""
    assert workflow_condition_for("ml-stage-4-labeling") == "run_labeling"


def test_11_3_b_training_runs_only_when_the_flag_is_set():
    """QRD 11.3.b: the training stage is gated by run_training in the workflow."""
    assert workflow_condition_for("ml-stage-5-training") == "run_training"


def test_11_3_c_metrics_without_labels_document_what_was_not_computed(tmp_path, run_cli):
    """QRD 11.3.c: with no ground-truth labels, metrics are not computed and the limitation is documented."""
    rows = [{"id": i, "prediction": "yes"} for i in range(4)]

    result, out = run_metrics(run_cli, tmp_path, rows, "no_labels")

    assert result.returncode == 0, result.stderr
    metrics = read_json(out / "metrics.json")
    assert find_value(metrics, "accuracy") is None
    assert find_value(metrics, "limitations")


@pending("Dashboard generation with a skipped optional stage needs the dashboard harness")
def test_11_3_d_dashboard_renders_when_an_optional_stage_is_skipped():
    """QRD 11.3.d: skipping an optional stage still lets dashboard generation succeed and shows the gap accurately."""
    raise NotImplementedError


def test_11_3_e_index_is_required_only_when_retrieval_is_configured(tmp_path):
    """QRD 11.3.e: inference runs with no index when no retrieval block is set; with one and no index, it fails."""
    with FakeOllamaServer(["answer"]) as server:
        code, _ = run_inference_stage(tmp_path, ollama_config(server), [{"id": "q1", "text": "hi"}], "no_index")
    assert code == 0

    with FakeOllamaServer() as server:
        config = ollama_config(server, retrieval={
            "vector_table": "my-project.my_dataset.chunk_vectors",
            "index_manifest": str(tmp_path / "never_built" / "index_manifest.json"),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        })
        code, out = run_inference_stage(tmp_path, config, [{"id": "q1", "text": "hi"}], "missing_index")
    assert code == 1
    assert not (out / "predictions.jsonl").exists()


def test_11_3_f_agent_artifacts_exist_only_when_agents_are_configured(tmp_path):
    """QRD 11.3.f: without an agents block there are no agent artifacts; with one, the findings and trace are written."""
    with FakeOllamaServer(["answer"]) as server:
        code, out = run_inference_stage(tmp_path, ollama_config(server), [{"id": "q1", "text": "hi"}], "no_agents")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0
    assert manifest.get("agents") is None
    assert not (out / "agent_findings.json").exists()

    with FakeOllamaServer(["answer", '{"next": "FINISH", "summary": "ok"}']) as server:
        config = ollama_config(server, agents={
            "objective": "Summarize.",
            "specialists": [{"name": "quality", "instructions": "Read.", "tools": ["read_artifact"]}],
        })
        code, out = run_inference_stage(tmp_path, config, [{"id": "q1", "text": "hi"}], "with_agents")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0
    assert (out / "agent_findings.json").exists()
    assert (out / "agent_trace.jsonl").exists()
    assert manifest["artifacts"]["agent_findings"].endswith("agent_findings.json")
