"""QRD tests for Section 6: Anomaly and Out-of-Distribution Detection.

Traceability: QRD 6.1 through 6.8. Runs pipeline/anomaly_detection.py (Local
Outlier Factor). Distribution-shift tests and OOD detectors are pending because
only the LOF detector exists.
"""
from __future__ import annotations

import json

from conftest import find_value, pending, read_jsonl, run_anomaly, write_jsonl


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


NORMAL_ROWS = [{"id": i, "x": (i % 6) * 0.1, "y": (i // 6) * 0.1} for i in range(30)]
OUTLIER_ROW = {"id": 30, "x": 100.0, "y": 100.0}


def test_6_1_a_seeded_anomaly_is_flagged(tmp_path, run_cli):
    """QRD 6.1.a: a deliberately seeded far-off record is flagged as an anomaly."""
    result, out = run_anomaly(run_cli, tmp_path, NORMAL_ROWS + [OUTLIER_ROW], "seeded")

    assert result.returncode == 0, result.stderr
    flagged = [r["id"] for r in read_jsonl(out / "anomaly_results.jsonl") if r["is_anomaly"]]
    assert 30 in flagged


def test_6_1_b_normal_observations_stay_under_false_positive_baseline(tmp_path, run_cli):
    """QRD 6.1.b: a representative normal dataset flags no more than 5% of records."""
    result, out = run_anomaly(run_cli, tmp_path, NORMAL_ROWS, "normal")

    assert result.returncode == 0, result.stderr
    assert find_value(read_json(out / "anomaly_report.json"), "anomaly_rate") <= 0.05


@pending("Distribution-shift detector is not implemented; only LOF anomaly detection exists")
def test_6_2_a_controlled_feature_shift_is_identified():
    """QRD 6.2.a: a target dataset shifted in known features is identified, with a difference consistent with the method."""
    raise NotImplementedError


@pending("Distribution-shift detector is not implemented; only LOF anomaly detection exists")
def test_6_2_b_same_distribution_samples_report_no_substantial_shift():
    """QRD 6.2.b: equivalent samples from one distribution do not produce a shift beyond tolerance."""
    raise NotImplementedError


@pending("No OOD detector is implemented")
def test_6_3_a_in_distribution_and_shifted_inputs_score_as_expected():
    """QRD 6.3.a: known in-distribution and shifted inputs produce the expected OOD scores or flags."""
    raise NotImplementedError


@pending("No OOD detector is implemented; the unsupported-method limitation cannot be exercised")
def test_6_3_b_unsupported_ood_configuration_reports_limitation():
    """QRD 6.3.b: an incompatible OOD detector or missing reference reports the limitation instead of succeeding."""
    raise NotImplementedError


def test_6_4_a_configured_reference_dataset_is_the_one_used(tmp_path, run_cli):
    """QRD 6.4.a: the detector records the reference dataset it was configured with."""
    reference = write_jsonl(tmp_path / "reference.jsonl", NORMAL_ROWS)

    result, out = run_anomaly(run_cli, tmp_path, NORMAL_ROWS + [OUTLIER_ROW], "with_reference", ("x", "y"), "--reference", reference)

    assert result.returncode == 0, result.stderr
    report = read_json(out / "anomaly_report.json")
    assert str(reference) in json.dumps(report)


def test_6_4_b_reference_with_incompatible_schema_is_blocked(tmp_path, run_cli):
    """QRD 6.4.b: a reference lacking the detector's feature columns blocks the comparison."""
    reference = write_jsonl(tmp_path / "bad_reference.jsonl", [{"id": i, "a": 1.0, "b": 2.0} for i in range(10)])

    result, _ = run_anomaly(run_cli, tmp_path, NORMAL_ROWS, "bad_reference", ("x", "y"), "--reference", reference)

    assert result.returncode != 0
    assert result.stderr.strip()


def test_6_5_a_summary_counts_match_record_level_flags(tmp_path, run_cli):
    """QRD 6.5.a: the saved anomaly count equals the number of records flagged in the results file."""
    result, out = run_anomaly(run_cli, tmp_path, NORMAL_ROWS + [OUTLIER_ROW], "summary")

    assert result.returncode == 0, result.stderr
    flagged = sum(1 for r in read_jsonl(out / "anomaly_results.jsonl") if r["is_anomaly"])
    assert find_value(read_json(out / "anomaly_report.json"), "anomaly_count") == flagged


@pending("Dashboard is not implemented, so finding-to-artifact links cannot be followed")
def test_6_5_b_dashboard_finding_links_to_detection_artifact():
    """QRD 6.5.b: a dashboard finding links to the detection artifact and supporting records."""
    raise NotImplementedError


def test_6_6_a_stricter_threshold_flags_fewer_records(tmp_path, run_cli):
    """QRD 6.6.a: flags follow the documented threshold rule; records are flagged only below the threshold."""
    rows = NORMAL_ROWS + [OUTLIER_ROW]
    loose, loose_out = run_anomaly(run_cli, tmp_path, rows, "loose", ("x", "y"), "--threshold", "-0.5")
    strict, strict_out = run_anomaly(run_cli, tmp_path, rows, "strict", ("x", "y"), "--threshold", "-3.0")

    assert loose.returncode == 0, loose.stderr
    assert strict.returncode == 0, strict.stderr
    loose_count = sum(1 for r in read_jsonl(loose_out / "anomaly_results.jsonl") if r["is_anomaly"])
    strict_count = sum(1 for r in read_jsonl(strict_out / "anomaly_results.jsonl") if r["is_anomaly"])
    assert strict_count <= loose_count


@pending("Only the LOF detector is registered, so method switching cannot be exercised")
def test_6_6_b_detector_selected_by_configuration_is_recorded():
    """QRD 6.6.b: switching between two registered detectors by configuration runs the selected one and records it."""
    raise NotImplementedError


def test_6_7_a_statistical_outlier_is_described_without_asserting_a_real_event(tmp_path, run_cli):
    """QRD 6.7.a: an unusual record is reported as an anomaly without a confirmed fraud or defect claim."""
    result, out = run_anomaly(run_cli, tmp_path, NORMAL_ROWS + [OUTLIER_ROW], "interpretation")

    assert result.returncode == 0, result.stderr
    report = read_json(out / "anomaly_report.json")
    assert find_value(report, "interpretation_limit")
    assert "is fraud" not in json.dumps(report).lower()


@pending("Dashboard wording for anomaly scores is not implemented")
def test_6_7_b_dashboard_labels_anomaly_scores_as_model_generated_findings():
    """QRD 6.7.b: anomaly scores and statistical warnings are labeled as model-generated findings, not verified causes."""
    raise NotImplementedError


@pending("No detector in the pipeline requires reference data, so a missing dependency cannot be triggered")
def test_6_8_a_missing_required_reference_is_reported_and_recorded():
    """QRD 6.8.a: a detector that requires reference data without one reports the dependency and records skipped or failed."""
    raise NotImplementedError


@pending("Dashboard is not implemented; detection-unavailable explanations cannot be checked")
def test_6_8_b_dashboard_explains_unavailable_detection():
    """QRD 6.8.b: the dashboard explains why detection was unavailable and shows no fabricated results."""
    raise NotImplementedError
