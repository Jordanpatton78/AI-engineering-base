"""QRD tests for Section 3: Model Selection and Execution.

Traceability: QRD 3.1 through 3.7. Training and inference run the real stage
scripts (pipeline/train.py, pipeline/inference.py). Adapter-registration and
regression-harness behavior is pending.
"""
from __future__ import annotations

import json

import pytest

from conftest import INFERENCE, labeled_rows, pending, read_jsonl, regression_rows, run_train, write_jsonl


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_3_1_a_inference_runs_with_existing_model_without_training(tmp_path, run_cli):
    """QRD 3.1.a: an existing compatible model is used for inference with training disabled."""
    joblib = pytest.importorskip("joblib")
    linear_model = pytest.importorskip("sklearn.linear_model")
    rows = labeled_rows(40)
    model = linear_model.LogisticRegression().fit([[r["x"]] for r in rows], [r["label"] for r in rows])
    model_path = tmp_path / "existing_model.joblib"
    joblib.dump(model, model_path)
    input_path = write_jsonl(tmp_path / "unlabeled.jsonl", [{"id": 1, "x": 3.0}, {"id": 2, "x": 30.0}])
    config = tmp_path / "inference.json"
    config.write_text(json.dumps({
        "type": "sklearn",
        "model_path": str(model_path),
        "feature_columns": ["x"],
        "task": "classification",
    }), encoding="utf-8")
    out = tmp_path / "out"

    result = run_cli(INFERENCE, "--input", input_path, "--config", config, "--output-prefix", out, "--run-id", "qrd-3-1-a")

    assert result.returncode == 0, result.stderr
    assert len(read_jsonl(out / "predictions.jsonl")) == 2
    assert read_json(out / "manifest.json")["status"] == "succeeded"
    assert not (out / "model.joblib").exists()  # inference must not write or retrain a model


def test_3_1_b_missing_model_reference_stops_before_inference(tmp_path, run_cli):
    """QRD 3.1.b: an inaccessible model reference is reported and inference does not proceed."""
    input_path = write_jsonl(tmp_path / "in.jsonl", [{"id": 1, "x": 1.0}])
    config = tmp_path / "inference.json"
    config.write_text(json.dumps({
        "type": "sklearn",
        "model_path": str(tmp_path / "missing.joblib"),
        "feature_columns": ["x"],
    }), encoding="utf-8")
    out = tmp_path / "out"

    result = run_cli(INFERENCE, "--input", input_path, "--config", config, "--output-prefix", out)

    assert result.returncode != 0
    assert result.stderr.strip()
    assert not (out / "predictions.jsonl").exists()


def test_3_2_a_training_produces_model_artifact_and_metadata(tmp_path, run_cli):
    """QRD 3.2.a: training on a labeled dataset completes and writes a model artifact and training metadata."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "train_ok")

    assert result.returncode == 0, result.stderr
    assert (out / "model.joblib").exists()
    assert read_json(out / "metrics.json")
    assert read_json(out / "manifest.json")["status"] == "succeeded"


def test_3_2_b_insufficient_training_data_fails_without_saving_a_model(tmp_path, run_cli):
    """QRD 3.2.b: insufficient training data fails with an explanation and saves no model."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(3), "too_small")

    assert result.returncode != 0
    assert result.stderr.strip()
    assert not (out / "model.joblib").exists()


def test_3_3_a_classification_and_regression_use_matching_task_configuration(tmp_path, run_cli):
    """QRD 3.3.a: supported models for classification and regression run with the matching task configuration."""
    cls_result, cls_out = run_train(run_cli, tmp_path, labeled_rows(40), "classification")
    reg_result, reg_out = run_train(
        run_cli, tmp_path, regression_rows(40), "regression",
        "--target", "y", "--task", "regression", "--model", "hist_gradient_boosting",
    )

    assert cls_result.returncode == 0, cls_result.stderr
    assert reg_result.returncode == 0, reg_result.stderr
    assert "accuracy" in json.dumps(read_json(cls_out / "metrics.json"))
    assert "rmse" in json.dumps(read_json(reg_out / "metrics.json"))


def test_3_3_b_regression_on_categorical_target_is_rejected_before_training(tmp_path, run_cli):
    """QRD 3.3.b: a model/task mismatch is identified before execution."""
    rows = [{"id": i, "x": float(i), "label": "yes" if i % 2 else "no"} for i in range(40)]

    result, out = run_train(run_cli, tmp_path, rows, "mismatch", "--task", "regression")

    assert result.returncode != 0
    assert not (out / "model.joblib").exists()


@pending("No adapter registry with a second concrete adapter and shared prediction schema exists yet")
def test_3_4_a_two_adapters_return_the_framework_prediction_contract():
    """QRD 3.4.a: two different adapters return outputs conforming to the same prediction contract."""
    raise NotImplementedError


@pending("No adapter regression harness exists yet")
def test_3_4_b_new_adapter_leaves_existing_adapter_regression_tests_passing():
    """QRD 3.4.b: adding a new adapter leaves existing adapters and workflows passing their regression tests."""
    raise NotImplementedError


def test_3_5_a_missing_target_column_blocks_execution(tmp_path, run_cli):
    """QRD 3.5.a: a missing required target blocks execution with a clear compatibility error."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "missing_target", "--target", "outcome")

    assert result.returncode != 0
    assert "outcome" in result.stderr
    assert not (out / "model.joblib").exists()


def test_3_6_a_model_parameters_change_through_configuration_only(tmp_path, run_cli):
    """QRD 3.6.a: a parameter change made in configuration reaches the model without code changes."""
    config = tmp_path / "train_config.json"
    config.write_text(json.dumps({"model": "extra_trees", "model_params": {"n_estimators": 7}}), encoding="utf-8")

    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "configured", "--config", config)

    assert result.returncode == 0, result.stderr
    manifest = read_json(out / "manifest.json")
    assert manifest["configuration"]["model_params"]["n_estimators"] == 7


def test_3_6_b_out_of_range_parameter_is_rejected(tmp_path, run_cli):
    """QRD 3.6.b: an out-of-range parameter is rejected and the constraint is reported."""
    config = tmp_path / "bad_config.json"
    config.write_text(json.dumps({"test_size": 1.5}), encoding="utf-8")

    result, _ = run_train(run_cli, tmp_path, labeled_rows(40), "bad_param", "--config", config)

    assert result.returncode != 0
    assert "test_size" in result.stderr


def test_3_7_a_run_manifest_records_model_metadata(tmp_path, run_cli):
    """QRD 3.7.a: each run records run ID, configuration, input references, version fields, and status."""
    result, out = run_train(run_cli, tmp_path, labeled_rows(40), "metadata", "--run-id", "qrd-3-7-a")

    assert result.returncode == 0, result.stderr
    manifest = read_json(out / "manifest.json")
    assert {"run_id", "status", "configuration", "input", "outputs"} <= set(manifest)
    assert manifest["run_id"] == "qrd-3-7-a"
    assert any("version" in key for key in manifest), "manifest records no model, preprocessing, or code version"


def test_3_7_b_repeated_run_with_identical_inputs_reproduces_metrics(tmp_path, run_cli):
    """QRD 3.7.b: identical data, configuration, and seed produce matching test metrics."""
    rows = labeled_rows(40)
    first, first_out = run_train(run_cli, tmp_path, rows, "first", "--random-state", "7")
    second, second_out = run_train(run_cli, tmp_path, rows, "second", "--random-state", "7")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert read_json(first_out / "metrics.json")["test_metrics"] == read_json(second_out / "metrics.json")["test_metrics"]
