"""QRD tests for Section 4: Automated Labeling and Pseudo-Label Generation.

Traceability: QRD 4.1 through 4.7. Runs pipeline/label.py with its example rule
expert. Confidence thresholds, training-side provenance, and dashboard warnings
are pending.
"""
from __future__ import annotations

import json

from conftest import EXAMPLE_EXPERT, pending, read_jsonl, run_label, run_metrics, write_jsonl


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def unlabeled_rows(n: int = 6) -> list[dict]:
    return [{"id": i, "signal": "hit" if i % 2 else "miss"} for i in range(n)]


def partly_labeled_rows(n: int = 6) -> list[dict]:
    rows = unlabeled_rows(n)
    for row in rows[: n // 2]:
        row["label"] = "no"
    return rows


def fully_labeled_rows(n: int = 6) -> list[dict]:
    return [{**row, "label": "no"} for row in unlabeled_rows(n)]


def test_4_1_a_label_state_detection_assigns_pseudo_labels_only_where_needed(tmp_path, run_cli):
    """QRD 4.1.a: fully labeled, partially labeled, and unlabeled datasets each follow the correct label path."""
    full, full_out = run_label(run_cli, tmp_path, fully_labeled_rows(), "full")
    partial, partial_out = run_label(run_cli, tmp_path, partly_labeled_rows(), "partial")
    empty, empty_out = run_label(run_cli, tmp_path, unlabeled_rows(), "empty")

    assert full.returncode == partial.returncode == empty.returncode == 0
    assert read_json(full_out / "label_report.json")["counts"]["pseudo_labels_generated"] == 0
    assert read_json(partial_out / "label_report.json")["counts"]["pseudo_labels_generated"] == 3
    assert read_json(empty_out / "label_report.json")["counts"]["pseudo_labels_generated"] == 6


def test_4_2_a_unlabeled_records_receive_expert_labels_on_matching_records(tmp_path, run_cli):
    """QRD 4.2.a: an unlabeled dataset run through the expert gets labels attached to the correct records."""
    result, out = run_label(run_cli, tmp_path, unlabeled_rows(6))

    assert result.returncode == 0, result.stderr
    labeled = read_jsonl(out / "labeled.jsonl")
    assert [(r["id"], r["label"]) for r in labeled] == [
        (i, "yes" if i % 2 else "no") for i in range(6)
    ]


@pending("Pseudo-label stage has no enabled/disabled switch; skip status is written by the Workflows orchestrator")
def test_4_2_b_disabled_pseudo_labeling_is_marked_skipped():
    """QRD 4.2.b: with pseudo-labeling disabled, the stage is marked skipped and downstream inputs remain valid."""
    raise NotImplementedError


def test_4_3_a_expert_identifier_and_positive_label_come_from_configuration(tmp_path, run_cli):
    """QRD 4.3.a: changing the expert settings in configuration changes the labels produced."""
    config = {"expert": {**EXAMPLE_EXPERT["expert"], "positive_label": "POSITIVE"}}

    result, out = run_label(run_cli, tmp_path, unlabeled_rows(6), "reconfigured", config)

    assert result.returncode == 0, result.stderr
    labels = {r["label"] for r in read_jsonl(out / "labeled.jsonl")}
    assert "POSITIVE" in labels


def test_4_3_b_unavailable_expert_fails_without_writing_labels(tmp_path, run_cli):
    """QRD 4.3.b: an unavailable expert model fails the stage and writes no labeled output."""
    config = {"expert": {"type": "unavailable_model"}}

    result, out = run_label(run_cli, tmp_path, unlabeled_rows(6), "bad_expert", config)

    assert result.returncode != 0
    assert not (out / "labeled.jsonl").exists()


def test_4_4_a_each_pseudo_label_records_source_and_record_identifier(tmp_path, run_cli):
    """QRD 4.4.a: each pseudo-label carries record identifier, source model, version, and label."""
    result, out = run_label(run_cli, tmp_path, unlabeled_rows(4))

    assert result.returncode == 0, result.stderr
    for record in read_jsonl(out / "labeled.jsonl"):
        serialized = json.dumps(record)
        assert "record_id" in serialized
        assert "expert_name" in serialized
        assert "expert_version" in serialized
        assert "label" in record


def test_4_4_b_confidence_is_not_fabricated_when_expert_provides_none(tmp_path, run_cli):
    """QRD 4.4.b: an expert without confidence yields records with no confidence value."""
    result, out = run_label(run_cli, tmp_path, unlabeled_rows(4))

    assert result.returncode == 0, result.stderr
    assert all(record.get("confidence") is None for record in read_jsonl(out / "labeled.jsonl"))


@pending("Needs an expert that reports confidence; the example rule expert does not")
def test_4_4_b_confidence_is_preserved_when_expert_provides_it():
    """QRD 4.4.b (second half): confidence from an expert that provides it is preserved in the output."""
    raise NotImplementedError


def test_4_5_a_ground_truth_labels_are_not_overwritten_by_expert_predictions(tmp_path, run_cli):
    """QRD 4.5.a: ground-truth labels remain unchanged even when the expert disagrees."""
    rows = [{"id": i, "signal": "miss", "label": "yes"} for i in range(4)]

    result, out = run_label(run_cli, tmp_path, rows)

    assert result.returncode == 0, result.stderr
    assert [r["label"] for r in read_jsonl(out / "labeled.jsonl")] == ["yes"] * 4


@pending("Training and dashboard label-source propagation is not implemented")
def test_4_5_b_label_source_propagates_to_training_metrics_and_dashboard():
    """QRD 4.5.b: label source stays identifiable in training data, metrics, and dashboard outputs."""
    raise NotImplementedError


@pending("Example rule expert emits no confidence, so threshold filtering cannot be exercised")
def test_4_6_a_confidence_threshold_splits_labels_as_configured():
    """QRD 4.6.a: labels above and below the confidence threshold are handled as specified."""
    raise NotImplementedError


def test_4_6_b_required_review_keeps_pseudo_labels_out_of_accepted_state(tmp_path, run_cli):
    """QRD 4.6.b: with review required, no pseudo-label is marked accepted without a human decision."""
    result, out = run_label(run_cli, tmp_path, unlabeled_rows(4), "review", None, "--require-review")

    assert result.returncode == 0, result.stderr
    labeled = read_jsonl(out / "labeled.jsonl")
    assert all("review_status" in record for record in labeled)
    assert all(record["review_status"] != "accepted" for record in labeled)


def test_4_6_c_filtering_summary_matches_saved_label_artifact(tmp_path, run_cli):
    """QRD 4.6.c: accepted-label counts in the report match the saved labeled file."""
    result, out = run_label(run_cli, tmp_path, unlabeled_rows(6))

    assert result.returncode == 0, result.stderr
    labeled = read_jsonl(out / "labeled.jsonl")
    accepted = sum(1 for record in labeled if record.get("accepted_for_downstream_use"))
    assert read_json(out / "label_report.json")["counts"]["pseudo_labels_accepted"] == accepted


def test_4_7_a_evaluation_against_pseudo_labels_is_identified_as_expert_agreement(tmp_path, run_cli):
    """QRD 4.7.a: metrics computed against pseudo-labels say they measure agreement with the expert."""
    rows = [
        {"id": 1, "label": "yes", "prediction": "yes", "label_source": "pseudo_label", "expert_name": "example_rule"},
        {"id": 2, "label": "no", "prediction": "yes", "label_source": "pseudo_label", "expert_name": "example_rule"},
    ]

    result, out = run_metrics(run_cli, tmp_path, rows, "pseudo_eval")

    assert result.returncode == 0, result.stderr
    assert "pseudo" in json.dumps(read_json(out / "evaluation_report.json")).lower()


@pending("Dashboard qualification warning is not implemented")
def test_4_7_b_dashboard_warns_when_evaluation_uses_pseudo_labels():
    """QRD 4.7.b: the dashboard shows a qualification whenever evaluation relies on pseudo-labels."""
    raise NotImplementedError
