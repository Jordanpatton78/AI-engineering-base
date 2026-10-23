"""QRD tests for Section 8: Workflow Configuration and Modularity.

Traceability: QRD 8.1 through 8.6. Configuration tests run pipeline/train.py with
JSON config files. Workflow-structure tests read pipeline/workflow.yaml as text,
since the Workflows runtime is not available locally. Extension-registration and
orchestrator skip-status behavior are pending.
"""
from __future__ import annotations

import json

from conftest import TRAIN, labeled_rows, pending, regression_rows, run_train, workflow_condition_for, workflow_jobs, workflow_text, write_jsonl


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_8_1_a_task_and_target_come_from_configuration_alone(tmp_path, run_cli):
    """QRD 8.1.a: the target and task are changed through configuration, with no command-line overrides."""
    input_path = write_jsonl(tmp_path / "config_only_input.jsonl", regression_rows(40))
    config = tmp_path / "config_only.json"
    config.write_text(json.dumps({"target": "y", "task": "regression", "random_state": 0}), encoding="utf-8")
    out = tmp_path / "config_only"
    out.mkdir()

    result = run_cli(
        TRAIN, "--input", input_path, "--config", config,
        "--model-path", out / "model.joblib",
        "--metrics", out / "metrics.json",
        "--predictions", out / "predictions.jsonl",
        "--manifest", out / "manifest.json",
    )

    assert result.returncode == 0, result.stderr
    assert "rmse" in json.dumps(read_json(out / "metrics.json"))


def test_8_1_b_malformed_configuration_is_rejected_with_actionable_error(tmp_path, run_cli):
    """QRD 8.1.b: a malformed configuration file is rejected with a message that points at the problem."""
    config = tmp_path / "malformed.json"
    config.write_text("{\"task\": \"classification\",", encoding="utf-8")
    input_path = write_jsonl(tmp_path / "malformed_input.jsonl", labeled_rows(20))

    result = run_cli(TRAIN, "--input", input_path, "--config", config, "--target", "label", "--manifest", tmp_path / "m.json")

    assert result.returncode != 0
    assert result.stderr.strip()


def test_8_1_b_unsupported_task_in_configuration_is_rejected(tmp_path, run_cli):
    """QRD 8.1.b: an unsupported task value in configuration is rejected rather than silently used."""
    config = tmp_path / "contradictory.json"
    config.write_text(json.dumps({"task": "ranking"}), encoding="utf-8")
    input_path = write_jsonl(tmp_path / "contradictory_input.jsonl", labeled_rows(40))

    result = run_cli(TRAIN, "--input", input_path, "--config", config, "--target", "label")

    assert result.returncode != 0


def test_8_2_a_optional_training_and_labeling_stages_are_toggleable(tmp_path):
    """QRD 8.2.a: the optional labeling and training stages each have a run flag, so each can be skipped."""
    assert workflow_condition_for("ml-stage-4-labeling") == "run_labeling"
    assert workflow_condition_for("ml-stage-5-training") == "run_training"


@pending("No detector or model adapter registry for replacing a component by name")
def test_8_2_b_replacing_a_component_does_not_affect_unrelated_stages():
    """QRD 8.2.b: replacing a registered detector or adapter executes the new one without side effects elsewhere."""
    raise NotImplementedError


def test_8_2_c_inference_falls_back_to_preprocessed_input_when_labeling_is_disabled():
    """QRD 8.2.c: training reads labeled output only when labeling runs; otherwise it reads preprocessed output."""
    assert 'if(run_labeling, labeling_uri + "/labeled.jsonl", preprocessing_uri + "/prepared.jsonl")' in workflow_text()


def test_8_3_a_two_domain_profiles_run_on_the_same_framework(tmp_path, run_cli):
    """QRD 8.3.a: two domain configurations change behavior while the manifest structure stays the same."""
    cls, cls_out = run_train(run_cli, tmp_path, labeled_rows(40), "profile_cls", "--config", _profile(tmp_path, "cls", {"model": "random_forest"}))
    reg, reg_out = run_train(
        run_cli, tmp_path, regression_rows(40), "profile_reg",
        "--config", _profile(tmp_path, "reg", {"model": "hist_gradient_boosting"}), "--target", "y", "--task", "regression",
    )

    assert cls.returncode == 0, cls.stderr
    assert reg.returncode == 0, reg.stderr
    cls_manifest = read_json(cls_out / "manifest.json")
    reg_manifest = read_json(reg_out / "manifest.json")
    assert set(cls_manifest) == set(reg_manifest)
    assert cls_manifest["configuration"] != reg_manifest["configuration"]


def test_8_3_b_a_later_run_does_not_change_an_earlier_runs_configuration(tmp_path, run_cli):
    """QRD 8.3.b: settings from one run do not leak into another run's recorded configuration."""
    first, first_out = run_train(run_cli, tmp_path, labeled_rows(40), "isolated_a", "--random-state", "1")
    before = read_json(first_out / "manifest.json")["configuration"]

    second, _ = run_train(run_cli, tmp_path, labeled_rows(40), "isolated_b", "--random-state", "99")

    assert first.returncode == second.returncode == 0
    assert read_json(first_out / "manifest.json")["configuration"] == before


@pending("Extension interface for connectors, adapters, evaluators, and detectors is not documented or registered")
def test_8_4_a_registered_extension_can_be_selected_and_executed():
    """QRD 8.4.a: a new connector, adapter, evaluator, or detector registered through its interface can be selected and run."""
    raise NotImplementedError


@pending("Extension interface is not defined, so there is no extension to regression-test against")
def test_8_4_b_existing_configurations_unchanged_after_registering_extension():
    """QRD 8.4.b: existing supported configurations behave the same after an extension is registered."""
    raise NotImplementedError


def test_8_5_a_only_the_two_optional_stages_can_be_switched_off():
    """QRD 8.5.a: the workflow gates only labeling and training, so a minimal run can omit them."""
    gated = [job for job in workflow_jobs() if workflow_condition_for(job)]
    assert gated == ["ml-stage-4-labeling", "ml-stage-5-training"]


def test_8_5_b_enabled_stages_run_in_the_specified_order():
    """QRD 8.5.b: the workflow calls the stages in the specified order."""
    assert workflow_jobs() == [
        "ml-stage-1-ingestion",
        "ml-stage-2-preprocess",
        "ml-stage-3-anomaly-detection",
        "ml-stage-4-labeling",
        "ml-stage-5-training",
        "ml-stage-6-inference",
        "ml-stage-7-compute-metrics",
    ]


@pending("Skip status is written by the Workflows runtime and cannot be exercised locally")
def test_8_6_a_skipped_stage_records_status_and_reason_in_run_record():
    """QRD 8.6.a: every skipped stage records its status and reason in the run record and manifest."""
    raise NotImplementedError


@pending("Downstream start conditions depend on the Workflows runtime, which cannot run locally")
def test_8_6_b_downstream_stages_proceed_only_when_inputs_exist():
    """QRD 8.6.b: downstream stages start only when their required artifacts are available, including after an upstream skip."""
    raise NotImplementedError


def _profile(tmp_path, name: str, values: dict) -> str:
    path = tmp_path / f"profile_{name}.json"
    path.write_text(json.dumps({"random_state": 0, **values}), encoding="utf-8")
    return str(path)
