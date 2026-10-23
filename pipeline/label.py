
#!/usr/bin/env python3
"""
Stage 4: Automated Labeling and Pseudo-Label Generation

Input:
    Stage 2 prepared.jsonl
    Optional Stage 2 manifest.json

Output:
    labeled.jsonl
    label_report.json
    manifest.json

Dependencies:
    pip install google-cloud-storage   # only for gs:// paths

To use a real expert model, subclass ExpertModel and implement predict().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------
# Storage: local files and Google Cloud Storage
# ---------------------------------------------------------------------

def read_bytes(path: str) -> bytes:
    if path.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError(
                "Install google-cloud-storage to use GCS paths."
            ) from exc

        bucket, key = path[5:].split("/", 1)
        return storage.Client().bucket(bucket).blob(key).download_as_bytes()

    return Path(path).read_bytes()


def write_bytes(path: str, data: bytes) -> None:
    if path.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise RuntimeError(
                "Install google-cloud-storage to use GCS paths."
            ) from exc

        bucket, key = path[5:].split("/", 1)
        storage.Client().bucket(bucket).blob(key).upload_from_string(data)
        return

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def join_path(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def read_json(path: str) -> dict:
    value = json.loads(read_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: str) -> list[dict]:
    records = []

    for line_number, line in enumerate(
        read_bytes(path).decode("utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"Expected a JSON object on line {line_number}."
            )

        records.append(record)

    return records


def write_json(path: str, value: Any) -> None:
    data = json.dumps(
        value, indent=2, ensure_ascii=False, default=str
    ) + "\n"
    write_bytes(path, data.encode("utf-8"))


def write_jsonl(path: str, records: list[dict]) -> None:
    data = "".join(
        json.dumps(record, ensure_ascii=False, default=str) + "\n"
        for record in records
    )
    write_bytes(path, data.encode("utf-8"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------
# Expert model interface
# ---------------------------------------------------------------------

class ExpertModel(ABC):
    """
    Common interface for any expert model.

    Implement predict(record) and return either:

        {"label": "positive", "confidence": 0.94}

    or a bare prediction:

        "positive"

    Confidence is optional. Return None if the model cannot provide
    a meaningful confidence estimate.

    You can also return metadata:

        {
            "label": "positive",
            "confidence": 0.94,
            "metadata": {"model_version": "v1"}
        }
    """

    @abstractmethod
    def predict(self, record: dict) -> Any:
        """Generate a prediction for one record."""
        raise NotImplementedError


# ---------------------------------------------------------------------
# Example expert: only for testing the pipeline
# ---------------------------------------------------------------------

class ExampleRuleExpert(ExpertModel):
    """
    Minimal example to demonstrate the interface.

    Replace this with your actual model adapter before using predictions
    for meaningful analysis or downstream model training.
    """

    def __init__(
        self,
        field: str,
        positive_value: Any,
        positive_label: Any,
        negative_label: Any,
    ):
        self.field = field
        self.positive_value = positive_value
        self.positive_label = positive_label
        self.negative_label = negative_label

    def predict(self, record: dict) -> dict:
        value = record.get(self.field)

        label = (
            self.positive_label
            if value == self.positive_value
            else self.negative_label
        )

        return {
            "label": label,
            "confidence": None,
            "metadata": {"adapter": "example_rule"},
        }


def build_expert(config: dict) -> ExpertModel:
    """
    Construct the configured expert.

    Extend this factory to support your real models. For example,
    an adapter could load a PyTorch model or call a Vertex AI endpoint.
    """

    expert_config = config.get("expert", {})
    expert_type = expert_config.get("type", "example_rule")

    if expert_type == "example_rule":
        required = [
            "field",
            "positive_value",
            "positive_label",
            "negative_label",
        ]

        missing = [
            key for key in required if key not in expert_config
        ]

        if missing:
            raise ValueError(
                f"Missing example_rule configuration keys: {missing}"
            )

        return ExampleRuleExpert(
            field=expert_config["field"],
            positive_value=expert_config["positive_value"],
            positive_label=expert_config["positive_label"],
            negative_label=expert_config["negative_label"],
        )

    raise ValueError(
        f"Unsupported expert type: {expert_type!r}. "
        "Add an adapter to build_expert()."
    )


# ---------------------------------------------------------------------
# Prediction normalization and record identifiers
# ---------------------------------------------------------------------

def normalize_prediction(
    result: Any,
) -> tuple[Any, float | None, dict]:
    """Normalize expert outputs into label, confidence, and metadata."""

    if isinstance(result, dict) and "label" in result:
        label = result["label"]
        confidence = result.get("confidence")
        metadata = result.get("metadata") or {}
    else:
        label = result
        confidence = None
        metadata = {}

    if label is None:
        raise ValueError("Expert returned a null label.")

    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(
            confidence, (int, float)
        ):
            raise ValueError(
                "Confidence must be numeric between 0 and 1, or None."
            )

        confidence = float(confidence)

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

    if not isinstance(metadata, dict):
        raise ValueError("Prediction metadata must be a dictionary.")

    return label, confidence, metadata


def get_record_id(
    record: dict,
    index: int,
    id_field: str | None,
) -> str:
    if id_field:
        value = record.get(id_field)

        if value is None:
            raise ValueError(
                f"Record {index} is missing ID field {id_field!r}."
            )

        return str(value)

    for field in ("record_id", "id", "example_id"):
        if record.get(field) is not None:
            return str(record[field])

    serialized = json.dumps(
        record, sort_keys=True, ensure_ascii=False, default=str
    ).encode("utf-8")

    return sha256(serialized)[:20]


# ---------------------------------------------------------------------
# Main Stage 4 pipeline
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    input_bytes = read_bytes(args.input)
    input_hash = sha256(input_bytes)
    records = read_jsonl(args.input)

    # Validate upstream stage completion when a manifest is supplied.
    input_manifest = (
        read_json(args.input_manifest)
        if args.input_manifest
        else {}
    )

    if input_manifest and input_manifest.get("status") != "succeeded":
        raise ValueError(
            "Input manifest must have status='succeeded'; "
            f"received {input_manifest.get('status')!r}."
        )

    config = read_json(args.config) if args.config else {}
    expert = build_expert(config)

    label_field = args.label_field or config.get("label_field", "label")
    id_field = args.id_field or config.get("id_field")

    min_confidence = (
        args.min_confidence
        if args.min_confidence is not None
        else config.get("min_confidence")
    )

    if min_confidence is not None:
        min_confidence = float(min_confidence)

        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1.")

    run_id = args.run_id or str(uuid.uuid4())
    started = time.time()

    output_records = []
    errors = []

    counts = {
        "input_records": len(records),
        "ground_truth_preserved": 0,
        "pseudo_labels_generated": 0,
        "pseudo_labels_accepted": 0,
        "pseudo_labels_rejected": 0,
        "prediction_errors": 0,
    }

    for index, original in enumerate(records):
        record = dict(original)

        try:
            record_id = get_record_id(record, index, id_field)
        except Exception as exc:
            record_id = f"row-{index}"
            errors.append({
                "record_id": record_id,
                "error": str(exc),
            })
            counts["prediction_errors"] += 1
            record["_labeling"] = {
                "label_type": "unlabeled",
                "review_status": "prediction_error",
                "error": str(exc),
                "run_id": run_id,
                "record_id": record_id,
            }
            output_records.append(record)
            continue

        # Existing labels are treated as ground truth and preserved.
        # Never overwrite them unless explicitly requested.
        if record.get(label_field) is not None and not args.overwrite_existing:
            record["_labeling"] = {
                "label_type": "ground_truth",
                "review_status": "not_applicable",
                "record_id": record_id,
            }

            counts["ground_truth_preserved"] += 1
            output_records.append(record)
            continue

        try:
            raw_prediction = expert.predict(record)
            label, confidence, metadata = normalize_prediction(
                raw_prediction
            )

            # If thresholding is enabled, missing confidence is rejected.
            accepted = (
                min_confidence is None
                or (
                    confidence is not None
                    and confidence >= min_confidence
                )
            )

            if accepted:
                record[label_field] = label
                counts["pseudo_labels_accepted"] += 1
            else:
                # Keep the prediction for audit/review, but don't place
                # it in the primary label field used for training.
                record.pop(label_field, None)
                counts["pseudo_labels_rejected"] += 1

            record["_labeling"] = {
                "label_type": "pseudo_label",
                "prediction": label,
                "confidence": confidence,
                "accepted_for_downstream_use": accepted,
                "review_status": (
                    "pending_review"
                    if args.require_review or not accepted
                    else "not_reviewed"
                ),
                "source": {
                    "expert_type": config.get(
                        "expert", {}
                    ).get("type", "example_rule"),
                    "expert_name": config.get(
                        "expert", {}
                    ).get("name"),
                    "expert_version": config.get(
                        "expert", {}
                    ).get("version"),
                    "model_or_endpoint": config.get(
                        "expert", {}
                    ).get("model_or_endpoint"),
                },
                "metadata": metadata,
                "run_id": run_id,
                "record_id": record_id,
            }

            counts["pseudo_labels_generated"] += 1

        except Exception as exc:
            # Keep the record so a single model failure does not discard
            # the rest of the dataset.
            counts["prediction_errors"] += 1
            errors.append({
                "record_id": record_id,
                "record_index": index,
                "error": str(exc),
            })

            record["_labeling"] = {
                "label_type": "unlabeled",
                "review_status": "prediction_error",
                "error": str(exc),
                "run_id": run_id,
                "record_id": record_id,
            }

        output_records.append(record)

    # Write artifacts.
    output_path = args.output or join_path(
        args.output_prefix, "labeled.jsonl"
    )
    report_path = args.report or join_path(
        args.output_prefix, "label_report.json"
    )
    manifest_path = args.manifest or join_path(
        args.output_prefix, "manifest.json"
    )

    write_jsonl(output_path, output_records)

    finished = time.time()

    report = {
        "stage": "stage_4_pseudo_labeling",
        "status": (
            "completed_with_errors" if errors else "completed"
        ),
        "run_id": run_id,
        "input": args.input,
        "input_sha256": input_hash,
        "output_record_count": len(output_records),
        "label_field": label_field,
        "minimum_confidence_threshold": min_confidence,
        "counts": counts,
        "errors": errors,
        "limitations": [
            "Pseudo-labels are model predictions, not verified ground truth.",
            "Confidence may be unavailable or uncalibrated.",
            "When a threshold is configured, predictions without confidence are rejected.",
            "The example rule expert is for testing only.",
            "Existing ground-truth labels are preserved by default.",
        ],
        "duration_seconds": round(finished - started, 3),
    }

    write_json(report_path, report)

    manifest = {
        "stage": "stage_4_pseudo_labeling",
        "status": (
            "completed_with_errors" if errors else "succeeded"
        ),
        "run_id": run_id,
        "input": {
            "path": args.input,
            "sha256": input_hash,
            "manifest_path": args.input_manifest,
            "record_count": len(records),
        },
        "outputs": {
            "labeled_records": output_path,
            "label_report": report_path,
        },
        "configuration": {
            "config_path": args.config,
            "label_field": label_field,
            "id_field": id_field,
            "min_confidence": min_confidence,
            "overwrite_existing": args.overwrite_existing,
            "require_review": args.require_review,
        },
        "counts": counts,
        "next_stage": "stage_5_model_evaluation",
        "created_at_unix": finished,
    }

    write_json(manifest_path, manifest)

    print(json.dumps({
        "status": manifest["status"],
        "outputs": manifest["outputs"],
        "manifest": manifest_path,
        "counts": counts,
    }, indent=2))

    if errors and args.fail_on_prediction_errors:
        return 2

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate auditable pseudo-labels using an expert model."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input prepared JSONL file from Stage 2.",
    )
    parser.add_argument(
        "--input-manifest",
        help="Optional input manifest from Stage 2.",
    )
    parser.add_argument(
        "--output-prefix",
        default="./stage_4",
        help="Output directory or GCS prefix.",
    )
    parser.add_argument("--output", help="Explicit labeled JSONL path.")
    parser.add_argument("--report", help="Explicit report JSON path.")
    parser.add_argument("--manifest", help="Explicit output manifest path.")
    parser.add_argument("--config", help="JSON expert and labeling configuration.")
    parser.add_argument("--label-field", help="Target label field; defaults to 'label'.")
    parser.add_argument("--id-field", help="Optional stable record ID field.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Accept only predictions meeting this confidence threshold.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing labels. Not recommended for ground truth.",
    )
    parser.add_argument(
        "--require-review",
        action="store_true",
        help="Mark accepted pseudo-labels as pending human review.",
    )
    parser.add_argument(
        "--fail-on-prediction-errors",
        action="store_true",
        help="Return exit code 2 if any record prediction fails.",
    )
    parser.add_argument("--run-id", help="Workflow-assigned run ID.")

    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)