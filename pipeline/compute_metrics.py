
#!/usr/bin/env python3
"""Compute classification or regression metrics from labeled JSONL records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    utc_now,
    read_text,
    json_safe,
    get_field,
    read_jsonl,
    write_json,
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def is_missing(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
    )


def record_id(record: dict, index: int, id_field: str | None) -> str:
    if id_field:
        try:
            return str(get_field(record, id_field))
        except KeyError:
            pass
    raw = json.dumps(record, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"record-{index}-{digest}"


# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------

def classification_metrics(
    y_true: list,
    y_pred: list,
    config: dict,
    probability_values: list | None = None,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    average = config.get("average", "weighted")
    zero_division = config.get("zero_division", 0)

    labels = config.get("labels")
    if labels is None:
        labels = sorted(
            set(y_true) | set(y_pred),
            key=lambda value: str(value),
        )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average,
            zero_division=zero_division, labels=labels,
        ),
        "recall": recall_score(
            y_true, y_pred, average=average,
            zero_division=zero_division, labels=labels,
        ),
        "f1": f1_score(
            y_true, y_pred, average=average,
            zero_division=zero_division, labels=labels,
        ),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro",
            zero_division=zero_division, labels=labels,
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro",
            zero_division=zero_division, labels=labels,
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro",
            zero_division=zero_division, labels=labels,
        ),
        "f1_micro": f1_score(
            y_true, y_pred, average="micro",
            zero_division=zero_division, labels=labels,
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted",
            zero_division=zero_division, labels=labels,
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(
                y_true, y_pred, labels=labels
            ).tolist(),
        },
        "per_class": classification_report(
            y_true, y_pred, labels=labels,
            output_dict=True,
            zero_division=zero_division,
        ),
    }

    # Optional ROC-AUC and average precision, when score data is configured.
    positive_label = config.get("positive_label")
    if probability_values is not None:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            unique_true = set(y_true)
            if len(unique_true) == 2:
                if positive_label is None:
                    positive_label = sorted(
                        unique_true, key=lambda v: str(v)
                    )[-1]

                binary_true = [
                    1 if value == positive_label else 0
                    for value in y_true
                ]
                metrics["roc_auc"] = roc_auc_score(
                    binary_true, probability_values
                )
                metrics["average_precision"] = average_precision_score(
                    binary_true, probability_values
                )
                metrics["roc_auc_positive_label"] = positive_label
            else:
                metrics["roc_auc_note"] = (
                    "ROC-AUC is only computed automatically for binary "
                    "classification. Multiclass AUC requires a configured "
                    "class-probability matrix and an explicit strategy."
                )
        except (ValueError, TypeError) as exc:
            metrics["roc_auc_note"] = str(exc)

    return json_safe(metrics)


def regression_metrics(y_true: list, y_pred: list) -> dict:
    import numpy as np
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = predicted - actual
    mse = mean_squared_error(actual, predicted)

    return json_safe({
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(actual, predicted),
        "r2": (
            r2_score(actual, predicted)
            if len(actual) >= 2 else None
        ),
        "mean_error": float(np.mean(errors)),
        "median_absolute_error": float(np.median(np.abs(errors))),
        "max_absolute_error": float(np.max(np.abs(errors))),
    })


# ---------------------------------------------------------------------
# Evaluation orchestration
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    started_at = utc_now()
    started = time.time()
    run_id = args.run_id or str(uuid.uuid4())

    config = json.loads(read_text(args.config))
    task = config.get("task", "classification").lower()
    if task not in {"classification", "regression"}:
        raise ValueError(
            "task must be 'classification' or 'regression'."
        )

    target_field = config.get("target_field", "label")
    prediction_field = config.get("prediction_field", "prediction")
    probability_field = config.get("probability_field")
    id_field = config.get("id_field")
    skip_invalid = bool(config.get("skip_invalid_records", True))

    output_prefix = args.output_prefix.rstrip("/")
    metrics_uri = f"{output_prefix}/metrics.json"
    report_uri = f"{output_prefix}/evaluation_report.json"
    manifest_uri = f"{output_prefix}/manifest.json"

    records = read_jsonl(args.input)
    y_true, y_pred, scores = [], [], []
    included_ids, excluded = [], []
    probability_scores_available = bool(probability_field)

    for index, record in enumerate(records):
        rid = record_id(record, index, id_field)
        try:
            actual = get_field(record, target_field)
            predicted = get_field(record, prediction_field)

            # Inference adapters may wrap the actual prediction in an object.
            prediction_value_path = config.get("prediction_value_path")
            if prediction_value_path:
                predicted = get_field(
                    { "value": predicted },
                    "value." + prediction_value_path,
                )

            if is_missing(actual) or is_missing(predicted):
                raise ValueError("Missing label or prediction.")

            if task == "regression":
                actual = float(actual)
                predicted = float(predicted)
                if not math.isfinite(actual) or not math.isfinite(predicted):
                    raise ValueError("Non-finite regression value.")

            y_true.append(actual)
            y_pred.append(predicted)
            included_ids.append(rid)

            if probability_field:
                try:
                    raw_score = get_field(record, probability_field)
                    positive_label = config.get("positive_label")

                    # Supports {class_name: probability} dictionaries,
                    # including sklearn's probabilities output.
                    if isinstance(raw_score, dict):
                        if positive_label is None:
                            raise ValueError(
                                "positive_label is required when "
                                "probability_field points to a class map."
                            )
                        raw_score = raw_score[str(positive_label)]

                    scores.append(float(raw_score))
                except (KeyError, TypeError, ValueError):
                    probability_scores_available = False

        except (KeyError, TypeError, ValueError) as exc:
            excluded.append({
                "record_id": rid,
                "reason": str(exc),
            })
            if not skip_invalid:
                raise ValueError(
                    f"Invalid record {rid}: {exc}"
                ) from exc

    if not y_true:
        raise ValueError(
            "No valid records remain for evaluation. Check field paths "
            "and ensure labels and predictions are present."
        )

    if task == "classification":
        metrics = classification_metrics(
            y_true,
            y_pred,
            config,
            scores if probability_scores_available
            and len(scores) == len(y_true) else None,
        )
    else:
        metrics = regression_metrics(y_true, y_pred)

    finished_at = utc_now()
    status = "succeeded" if not excluded else "completed_with_warnings"

    metrics_document = {
        "task": task,
        "run_id": run_id,
        "computed_at": finished_at,
        "metrics": metrics,
        "counts": {
            "input_records": len(records),
            "evaluated_records": len(y_true),
            "excluded_records": len(excluded),
        },
        "configuration": {
            "target_field": target_field,
            "prediction_field": prediction_field,
            "prediction_value_path": config.get("prediction_value_path"),
            "average": config.get("average", "weighted"),
        },
    }

    report = {
        "stage": "compute_metrics",
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(time.time() - started, 3),
        "input_uri": args.input,
        "metrics_uri": metrics_uri,
        "counts": metrics_document["counts"],
        "excluded_records": excluded,
        "limitations": [
            "Metrics describe only the records included in this evaluation.",
            "Classification metrics depend on label definitions and averaging strategy.",
            "Pseudo-label-based evaluation is not equivalent to evaluation "
            "against verified ground truth.",
        ],
    }

    manifest = {
        "stage": "compute_metrics",
        "run_id": run_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "inputs": {
            "data": args.input,
            "config": args.config,
        },
        "artifacts": {
            "metrics": metrics_uri,
            "report": report_uri,
        },
        "counts": metrics_document["counts"],
    }

    write_json(metrics_uri, metrics_document)
    write_json(report_uri, report)
    write_json(manifest_uri, manifest)

    print(json.dumps({
        "status": status,
        "run_id": run_id,
        "metrics": metrics_uri,
        "report": report_uri,
        "manifest": manifest_uri,
        "counts": metrics_document["counts"],
    }, indent=2))

    return 0 if not excluded else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics from labeled JSONL records."
    )
    parser.add_argument("--input", required=True,
                        help="JSONL file or gs:// URI containing labels and predictions.")
    parser.add_argument("--config", required=True,
                        help="JSON metrics configuration file.")
    parser.add_argument("--output-prefix", required=True,
                        help="Local output directory or gs:// prefix.")
    parser.add_argument("--run-id",
                        help="Optional pipeline run ID.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"Metric computation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)