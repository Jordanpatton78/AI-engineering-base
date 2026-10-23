"""QRD tests for Section 12: Recommended Implementation Decisions.

Traceability: QRD 12.1 through 12.5. Stage isolation, inference guards, and the
anomaly-without-OOD path run the real scripts. Cloud deployment, GCS/BigQuery
routing, and the common artifact schema are pending.
"""
from __future__ import annotations

import json

from conftest import ANOMALY, INFERENCE, pending, workflow_condition_for, workflow_jobs, write_jsonl


ANOMALY_ROWS = [{"id": i, "x": (i % 6) * 0.1, "y": (i // 6) * 0.1} for i in range(30)] + [{"id": 30, "x": 100.0, "y": 100.0}]


def test_12_1_a_stage_completes_from_its_declared_inputs_alone(tmp_path, run_cli):
    """QRD 12.1.a: a stage run in an empty working directory, using only its declared inputs, completes."""
    input_path = write_jsonl(tmp_path / "declared_input.jsonl", ANOMALY_ROWS)
    isolated = tmp_path / "fresh_workdir"
    isolated.mkdir()
    out = tmp_path / "out"
    out.mkdir()

    result = run_cli(
        ANOMALY, "--input", input_path, "--feature-columns", "x", "y", "--n-neighbors", "5",
        "--output", out / "anomaly_results.jsonl", "--report", out / "anomaly_report.json",
        "--manifest", out / "manifest.json",
        cwd=isolated,
    )

    assert result.returncode == 0, result.stderr


@pending("Independent deployment of one Cloud Run Job cannot be checked without gcloud and deployed jobs")
def test_12_1_b_updating_one_stage_leaves_other_stages_unchanged():
    """QRD 12.1.b: updating one stage's implementation requires no code changes to other stages when interfaces hold."""
    raise NotImplementedError


@pending("GCS and BigQuery routing is not implemented; stages read and write local paths")
def test_12_2_a_models_reports_and_manifests_go_to_gcs_and_tables_go_to_bigquery():
    """QRD 12.2.a: model files, reports, and manifests are saved to GCS; configured queryable outputs go to BigQuery."""
    raise NotImplementedError


@pending("Cross-storage manifest references need the GCS and BigQuery integration")
def test_12_2_b_manifests_reference_gcs_objects_and_bigquery_tables():
    """QRD 12.2.b: manifests reference both GCS objects and BigQuery tables, and downstream jobs can retrieve them."""
    raise NotImplementedError


@pending("No common artifact schema file exists to validate manifests against")
def test_12_3_a_every_stage_manifest_conforms_to_the_common_schema():
    """QRD 12.3.a: every registered stage's manifest validates against the common artifact schema."""
    raise NotImplementedError


@pending("No second conforming stage implementation exists to swap in")
def test_12_3_b_replacing_a_stage_keeps_downstream_stages_working():
    """QRD 12.3.b: a replaced stage with the same interface produces outputs that downstream stages consume."""
    raise NotImplementedError


def test_12_3_c_inference_rejects_an_upstream_manifest_with_failed_status(tmp_path, run_cli):
    """QRD 12.3.c: a receiving stage rejects an upstream artifact whose manifest is not a success."""
    upstream = tmp_path / "failed_manifest.json"
    upstream.write_text(json.dumps({"status": "failed"}), encoding="utf-8")
    input_path = write_jsonl(tmp_path / "rejected_input.jsonl", [{"id": 1, "x": 1.0}])
    config = tmp_path / "inference.json"
    config.write_text(json.dumps({"type": "sklearn", "model_path": str(tmp_path / "m.joblib"), "feature_columns": ["x"]}), encoding="utf-8")
    out = tmp_path / "out"

    result = run_cli(INFERENCE, "--input", input_path, "--input-manifest", upstream, "--config", config, "--output-prefix", out)

    assert result.returncode != 0
    assert not (out / "predictions.jsonl").exists()


def test_12_4_a_training_is_conditional_and_keeps_stage_order():
    """QRD 12.4.a: training is gated by run_training, and the stage sequence is unchanged."""
    assert workflow_condition_for("ml-stage-5-training") == "run_training"
    assert workflow_jobs().index("ml-stage-5-training") > workflow_jobs().index("ml-stage-4-labeling")


def test_12_4_b_pseudo_labeling_is_conditional_and_keeps_stage_order():
    """QRD 12.4.b: pseudo-labeling is gated by run_labeling and sits before training in the sequence."""
    assert workflow_condition_for("ml-stage-4-labeling") == "run_labeling"
    assert workflow_jobs().index("ml-stage-4-labeling") < workflow_jobs().index("ml-stage-5-training")


def test_12_4_c_inference_stops_when_no_model_is_configured(tmp_path, run_cli):
    """QRD 12.4.c: with training disabled and no model reference, inference stops before producing predictions."""
    input_path = write_jsonl(tmp_path / "no_model_input.jsonl", [{"id": 1, "x": 1.0}])
    config = tmp_path / "no_model.json"
    config.write_text(json.dumps({"type": "sklearn", "feature_columns": ["x"]}), encoding="utf-8")
    out = tmp_path / "out"

    result = run_cli(INFERENCE, "--input", input_path, "--config", config, "--output-prefix", out)

    assert result.returncode != 0
    assert not (out / "predictions.jsonl").exists()


@pending("No OOD detector exists, so data-level and model-level results cannot be recorded separately")
def test_12_5_a_data_anomaly_and_model_ood_are_recorded_as_separate_results():
    """QRD 12.5.a: a record inside the model's input distribution is still recorded as a separate data-level result."""
    raise NotImplementedError


def test_12_5_b_data_anomaly_detection_runs_without_an_ood_detector(tmp_path, run_cli):
    """QRD 12.5.b: data-level anomaly detection runs on its own, with no model-specific OOD detector configured."""
    input_path = write_jsonl(tmp_path / "independent_input.jsonl", ANOMALY_ROWS)
    out = tmp_path / "out"
    out.mkdir()

    result = run_cli(
        ANOMALY, "--input", input_path, "--feature-columns", "x", "y", "--n-neighbors", "5",
        "--output", out / "anomaly_results.jsonl", "--report", out / "anomaly_report.json",
        "--manifest", out / "manifest.json",
    )

    assert result.returncode == 0, result.stderr
    assert (out / "anomaly_results.jsonl").exists()


@pending("Dashboard does not exist yet, so its separation of finding types cannot be checked")
def test_12_5_c_dashboard_separates_data_anomalies_shifts_and_model_ood_findings():
    """QRD 12.5.c: the dashboard shows data anomalies, distribution shifts, and model-level OOD findings separately."""
    raise NotImplementedError
