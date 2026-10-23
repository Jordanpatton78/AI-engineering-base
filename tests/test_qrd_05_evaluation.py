"""QRD tests for Section 5: Model Evaluation and Metrics.

Traceability: QRD 5.1 through 5.7. Runs pipeline/compute_metrics.py. Evaluator
registration, comparison, subgroup breakdowns, and BigQuery persistence are pending.
"""
from __future__ import annotations

import json
import math

import pytest

from conftest import find_value, pending, run_metrics

REGRESSION_METRICS = {"task": "regression", "target_field": "y", "prediction_field": "prediction"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# y_true = [yes, no, yes, yes], y_pred = [yes, no, no, yes]
HAND_CALCULATED = [
    {"id": 1, "label": "yes", "prediction": "yes"},
    {"id": 2, "label": "no", "prediction": "no"},
    {"id": 3, "label": "yes", "prediction": "no"},
    {"id": 4, "label": "yes", "prediction": "yes"},
]


def test_5_1_a_metric_values_match_hand_calculated_results(tmp_path, run_cli):
    """QRD 5.1.a: classification and regression metrics match hand-calculated values within tolerance."""
    cls_result, cls_out = run_metrics(run_cli, tmp_path, HAND_CALCULATED, "cls")
    reg_rows = [{"id": 1, "y": 1.0, "prediction": 1.0}, {"id": 2, "y": 2.0, "prediction": 2.0}, {"id": 3, "y": 3.0, "prediction": 5.0}]
    reg_result, reg_out = run_metrics(run_cli, tmp_path, reg_rows, "reg", REGRESSION_METRICS)

    assert cls_result.returncode == 0, cls_result.stderr
    assert reg_result.returncode == 0, reg_result.stderr
    assert find_value(read_json(cls_out / "metrics.json"), "accuracy") == pytest.approx(0.75, abs=1e-9)
    assert find_value(read_json(reg_out / "metrics.json"), "mae") == pytest.approx(2 / 3, abs=1e-9)
    assert find_value(read_json(reg_out / "metrics.json"), "rmse") == pytest.approx(math.sqrt(4 / 3), abs=1e-9)


def test_5_1_b_metric_is_unavailable_when_prediction_field_is_missing(tmp_path, run_cli):
    """QRD 5.1.b: with the prediction field removed, accuracy is marked unavailable rather than computed."""
    rows = [{"id": i, "label": "yes"} for i in range(4)]

    result, out = run_metrics(run_cli, tmp_path, rows, "no_predictions")

    assert result.returncode == 0, result.stderr
    assert find_value(read_json(out / "metrics.json"), "accuracy") is None


def test_5_2_a_classification_and_regression_produce_their_expected_metrics(tmp_path, run_cli):
    """QRD 5.2.a: representative classification and regression cases produce their registered metrics."""
    cls_result, cls_out = run_metrics(run_cli, tmp_path, HAND_CALCULATED, "cls_cov")
    reg_rows = [{"id": i, "y": float(i), "prediction": float(i) + 0.5} for i in range(5)]
    reg_result, reg_out = run_metrics(run_cli, tmp_path, reg_rows, "reg_cov", REGRESSION_METRICS)

    assert cls_result.returncode == 0, cls_result.stderr
    assert reg_result.returncode == 0, reg_result.stderr
    assert find_value(read_json(cls_out / "metrics.json"), "accuracy") is not None
    assert find_value(read_json(reg_out / "metrics.json"), "rmse") is not None


@pending("Metrics are hard-coded per task; there is no evaluator registry for a new evaluator to join")
def test_5_2_b_registered_evaluator_works_without_changing_others():
    """QRD 5.2.b: a new evaluator works through the shared interface without modifying unrelated evaluators."""
    raise NotImplementedError


@pending("No comparison stage exists to evaluate two models side by side")
def test_5_3_a_model_comparison_matches_independent_metric_differences():
    """QRD 5.3.a: comparing two models reports metric differences that match independent calculations."""
    raise NotImplementedError


@pending("No comparison stage exists to flag evaluations from different datasets or label sources")
def test_5_3_b_incomparable_evaluations_are_flagged():
    """QRD 5.3.b: results from different datasets or label sources are identified and comparison is warned against."""
    raise NotImplementedError


def test_5_4_a_confusion_matrix_reflects_known_misclassifications(tmp_path, run_cli):
    """QRD 5.4.a: the error summary reports the known misclassification pattern."""
    result, out = run_metrics(run_cli, tmp_path, HAND_CALCULATED, "errors")

    assert result.returncode == 0, result.stderr
    # Sorted labels are [no, yes]; row = true label, column = predicted label.
    assert find_value(read_json(out / "metrics.json"), "confusion_matrix") == [[1, 0], [1, 2]]


@pending("No subgroup configuration exists for group-level metrics")
def test_5_4_b_group_metrics_match_independent_calculation():
    """QRD 5.4.b: group-level metrics for configured subgroups match independently calculated results."""
    raise NotImplementedError


@pending("Metrics output does not yet record a ground-truth versus pseudo-label source designation")
def test_5_5_a_evaluation_source_designation_is_preserved():
    """QRD 5.5.a: evaluations on ground-truth and pseudo-label data keep the correct source designation."""
    raise NotImplementedError


@pending("Dashboard is not implemented; verified and pseudo-label metrics cannot be distinguished on screen")
def test_5_5_b_dashboard_separates_verified_metrics_from_pseudo_label_agreement():
    """QRD 5.5.b: the dashboard distinguishes verified-label metrics from pseudo-label agreement."""
    raise NotImplementedError


def test_5_6_a_tiny_sample_reports_its_limitation(tmp_path, run_cli):
    """QRD 5.6.a: an evaluation with too few records or one class reports that its metrics are limited."""
    rows = [{"id": 1, "label": "yes", "prediction": "yes"}]

    result, out = run_metrics(run_cli, tmp_path, rows, "tiny")

    assert result.returncode == 0, result.stderr
    assert find_value(read_json(out / "metrics.json"), "limitations")


def test_5_6_b_missing_labels_are_unavailable_not_zero(tmp_path, run_cli):
    """QRD 5.6.b: with no valid labels, metrics are explained as unavailable and not reported as zero."""
    rows = [{"id": i, "prediction": "yes"} for i in range(4)]

    result, out = run_metrics(run_cli, tmp_path, rows, "unlabeled_eval")

    assert result.returncode == 0, result.stderr
    assert find_value(read_json(out / "metrics.json"), "accuracy") is None


@pending("BigQuery persistence of evaluation results is not implemented")
def test_5_7_a_evaluation_results_persist_to_bigquery_with_run_and_model_ids():
    """QRD 5.7.a: multiple evaluations remain queryable in BigQuery with correct run and model identifiers."""
    raise NotImplementedError


@pending("BigQuery persistence of evaluation results is not implemented")
def test_5_7_b_historical_comparison_uses_stored_versions_and_conditions():
    """QRD 5.7.b: earlier results retrieved for comparison use the correct stored versions and conditions."""
    raise NotImplementedError
