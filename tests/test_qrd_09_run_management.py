"""QRD tests for Section 9: Run Management and Reporting.

Traceability: QRD 9.1 through 9.5. Run records come from the stage manifests that
pipeline/train.py writes. Progress, failure visibility, artifact reuse, export, and
consolidated summaries need the orchestrator, a dashboard, or an export command,
none of which exist locally yet, so those tests are pending.
"""
from __future__ import annotations

import json
import subprocess
import sys

from conftest import ROOT, TRAIN, TRAIN_BASE_ARGS, labeled_rows, pending, run_train, write_jsonl


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_9_1_a_run_record_has_identity_configuration_and_outputs(tmp_path, run_cli):
    """QRD 9.1.a: a run records its run ID, configuration, source references, status, timestamps, and artifacts."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "record", "--run-id", "qrd-9-1-a")

    assert result.returncode == 0, result.stderr
    manifest = read_json(out / "manifest.json")
    assert {"run_id", "configuration", "status", "outputs", "input"} <= set(manifest)
    assert "created_at_unix" in manifest or "duration_seconds" in manifest
    assert manifest["run_id"] == "qrd-9-1-a"


def test_9_1_b_concurrent_runs_keep_their_own_records(tmp_path):
    """QRD 9.1.b: two runs executing at the same time keep their records and artifacts under their own run IDs."""
    launched = []
    for name in ("concurrent_a", "concurrent_b"):
        input_path = write_jsonl(tmp_path / f"{name}_input.jsonl", labeled_rows(40))
        out = tmp_path / name
        out.mkdir()
        command = [
            sys.executable, str(TRAIN), "--input", str(input_path), *TRAIN_BASE_ARGS,
            "--model-path", str(out / "model.joblib"),
            "--metrics", str(out / "metrics.json"),
            "--predictions", str(out / "predictions.jsonl"),
            "--manifest", str(out / "manifest.json"),
            "--run-id", name,
        ]
        launched.append((name, out, subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))

    for name, out, process in launched:
        _, stderr = process.communicate(timeout=600)
        assert process.returncode == 0, stderr
        assert read_json(out / "manifest.json")["run_id"] == name


@pending("Stage-by-stage run-record updates come from the orchestrator, which cannot run locally")
def test_9_2_a_run_record_updates_as_each_stage_starts_and_completes():
    """QRD 9.2.a: the run record is updated as each stage of a multi-stage workflow starts and completes."""
    raise NotImplementedError


@pending("Forcing a Cloud Run Job failure requires deployed jobs")
def test_9_2_b_failed_cloud_run_job_records_status_stage_and_error():
    """QRD 9.2.b: a failed Cloud Run Job records run status, failing stage, and error details for the dashboard."""
    raise NotImplementedError


def test_9_3_a_repeated_runs_are_both_retained(tmp_path, run_cli):
    """QRD 9.3.a: rerunning with the same source and configuration keeps both runs and their artifacts."""
    first, first_out = run_train(run_cli, tmp_path, labeled_rows(40), "repeat_1", "--run-id", "repeat-1")
    second, second_out = run_train(run_cli, tmp_path, labeled_rows(40), "repeat_2", "--run-id", "repeat-2")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert read_json(first_out / "manifest.json")["run_id"] == "repeat-1"
    assert read_json(second_out / "manifest.json")["run_id"] == "repeat-2"
    assert (first_out / "model.joblib").exists() and (second_out / "model.joblib").exists()


def test_9_3_b_changed_parameter_is_recorded_and_visible_in_comparison(tmp_path, run_cli):
    """QRD 9.3.b: changing one parameter produces a run whose recorded configuration differs in that parameter."""
    base, base_out = run_train(run_cli, tmp_path, labeled_rows(40), "base", "--random-state", "1")
    changed, changed_out = run_train(run_cli, tmp_path, labeled_rows(40), "changed", "--random-state", "2")

    assert base.returncode == 0, base.stderr
    assert changed.returncode == 0, changed.stderr
    base_config = read_json(base_out / "manifest.json")["configuration"]
    changed_config = read_json(changed_out / "manifest.json")["configuration"]
    differing = {key for key in set(base_config) | set(changed_config) if base_config.get(key) != changed_config.get(key)}
    assert "random_state" in differing


@pending("Artifact reuse based on input, code, configuration, and dependency hashes is not implemented")
def test_9_3_c_previous_stage_output_is_reused_only_when_criteria_match():
    """QRD 9.3.c: a previous stage output is reused only when its inputs, code, configuration, and dependencies match."""
    raise NotImplementedError


@pending("No export command exists for results, metrics, or findings")
def test_9_4_a_exported_contents_match_stored_artifacts():
    """QRD 9.4.a: exported results in the configured formats match the corresponding stored artifacts."""
    raise NotImplementedError


@pending("No export command exists, so export permissions cannot be checked")
def test_9_4_b_export_without_permission_is_denied():
    """QRD 9.4.b: export without sufficient permissions is denied and returns no restricted information."""
    raise NotImplementedError


@pending("No consolidated run summary stage exists yet")
def test_9_5_a_consolidated_summary_includes_all_required_sections():
    """QRD 9.5.a: the run summary covers status, data quality findings, model results, metrics, anomalies, and limitations."""
    raise NotImplementedError


@pending("No consolidated run summary or recommendation generator exists yet")
def test_9_5_b_recommendations_cite_supporting_rules_or_outputs():
    """QRD 9.5.b: generated recommendations reference supporting rules or model outputs and are marked as non-verified."""
    raise NotImplementedError
