
#!/usr/bin/env python3
"""Interactive Streamlit dashboard for modular ML pipeline results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="ML Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------

def read_text_uri(uri: str) -> str:
    """Read a local file or a Google Cloud Storage URI."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        bucket_name, _, blob_name = uri[5:].partition("/")
        if not bucket_name or not blob_name:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return storage.Client().bucket(
            bucket_name
        ).blob(blob_name).download_as_text(encoding="utf-8")

    return Path(uri).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_json(uri: str) -> dict:
    return json.loads(read_text_uri(uri))


@st.cache_data(show_spinner=False)
def load_jsonl(uri: str) -> list[dict]:
    records = []
    for line_number, line in enumerate(read_text_uri(uri).splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number}: {exc}"
            ) from exc
        if isinstance(record, dict):
            records.append(record)
    return records


def get_field(record: dict, field: str) -> Any:
    """Support nested fields such as prediction.prediction."""
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def unwrap_prediction(value: Any) -> Any:
    """Unwrap common prediction shapes emitted by inference adapters."""
    if isinstance(value, dict):
        for key in ("prediction", "label", "text", "output"):
            if key in value:
                return value[key]
    return value


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def numeric_metrics(metrics: dict) -> dict[str, float]:
    """Extract scalar metrics, excluding nested reports and matrices."""
    result = {}
    for key, value in metrics.items():
        number = finite_number(value)
        if number is not None:
            result[key] = number
    return result


def load_run(metrics_uri: str, report_uri: str, manifest_uri: str) -> dict:
    metrics_doc = load_json(metrics_uri)
    report = load_json(report_uri) if report_uri.strip() else {}
    manifest = load_json(manifest_uri) if manifest_uri.strip() else {}

    # Supports the compute_metrics.py output format.
    metric_values = metrics_doc.get("metrics", metrics_doc)

    return {
        "name": metrics_doc.get("run_id")
            or manifest.get("run_id")
            or Path(metrics_uri).name,
        "metrics_uri": metrics_uri,
        "metrics_doc": metrics_doc,
        "metrics": metric_values,
        "report": report,
        "manifest": manifest,
        "task": metrics_doc.get("task", "unknown"),
        "counts": metrics_doc.get("counts", report.get("counts", {})),
        "status": (
            manifest.get("status")
            or report.get("status")
            or "unknown"
        ),
    }


# ---------------------------------------------------------------------
# Reusable chart helpers
# ---------------------------------------------------------------------

def show_metric_cards(run: dict) -> None:
    metrics = run["metrics"]
    counts = run["counts"]

    if run["task"] == "classification":
        preferred = [
            "accuracy", "precision", "recall", "f1",
            "f1_macro", "f1_weighted", "roc_auc",
        ]
    elif run["task"] == "regression":
        preferred = ["mse", "rmse", "mae", "r2", "mean_error"]
    else:
        preferred = list(numeric_metrics(metrics))

    available = [
        (name, finite_number(metrics.get(name)))
        for name in preferred
        if finite_number(metrics.get(name)) is not None
    ][:6]

    if available:
        columns = st.columns(len(available))
        for col, (name, value) in zip(columns, available):
            col.metric(name.replace("_", " ").title(), f"{value:,.4f}")

    st.caption(
        f"Evaluated: {counts.get('evaluated_records', '—')} · "
        f"Excluded: {counts.get('excluded_records', '—')} · "
        f"Input: {counts.get('input_records', '—')}"
    )


def plot_metric_bars(metrics: dict) -> None:
    values = numeric_metrics(metrics)
    # Confusion matrix, loss values nested in objects, and other
    # non-scalar artifacts are automatically excluded.
    if not values:
        st.info("No scalar metrics are available to chart.")
        return

    df = pd.DataFrame({
        "Metric": list(values.keys()),
        "Value": list(values.values()),
    })
    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        title="Evaluation Metrics",
        text_auto=".4f",
    )
    fig.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix(metrics: dict) -> None:
    data = metrics.get("confusion_matrix")
    if not isinstance(data, dict):
        st.info("No confusion matrix was saved for this run.")
        return

    labels = data.get("labels", [])
    matrix = data.get("matrix", [])

    if not matrix or not labels:
        st.info("The confusion matrix is empty.")
        return

    fig = px.imshow(
        matrix,
        x=[str(label) for label in labels],
        y=[str(label) for label in labels],
        text_auto=True,
        aspect="auto",
        labels={
            "x": "Predicted label",
            "y": "Actual label",
            "color": "Count",
        },
        title="Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Rows are actual labels; columns are predicted labels.")


def plot_per_class(metrics: dict) -> None:
    report = metrics.get("per_class")
    if not isinstance(report, dict):
        st.info("No per-class classification report is available.")
        return

    rows = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        if label in {"accuracy"}:
            continue
        for metric in ("precision", "recall", "f1-score"):
            number = finite_number(values.get(metric))
            if number is not None:
                rows.append({
                    "Class": str(label),
                    "Metric": metric.replace("-", " ").title(),
                    "Score": number,
                })

    if rows:
        df = pd.DataFrame(rows)
        fig = px.bar(
            df,
            x="Class",
            y="Score",
            color="Metric",
            barmode="group",
            title="Per-Class Performance",
            range_y=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No per-class scores could be extracted.")


def plot_regression_records(
    records: list[dict],
    target_field: str,
    prediction_field: str,
) -> None:
    rows = []
    for record in records:
        actual = get_field(record, target_field)
        predicted = get_field(record, prediction_field)
        predicted = unwrap_prediction(predicted)

        actual_num = finite_number(actual)
        predicted_num = finite_number(predicted)
        if actual_num is None or predicted_num is None:
            continue

        rows.append({
            "Actual": actual_num,
            "Predicted": predicted_num,
            "Residual": predicted_num - actual_num,
        })

    if not rows:
        st.info(
            "No numeric actual/predicted pairs were found. "
            "Check the configured field paths."
        )
        return

    df = pd.DataFrame(rows)

    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            df,
            x="Actual",
            y="Predicted",
            title="Actual vs. Predicted",
            hover_data=["Residual"],
        )
        lower = min(df["Actual"].min(), df["Predicted"].min())
        upper = max(df["Actual"].max(), df["Predicted"].max())
        fig.add_trace(go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            name="Perfect prediction",
            line={"dash": "dash"},
        ))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.histogram(
            df,
            x="Residual",
            nbins=40,
            title="Residual Distribution",
        )
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)


def plot_run_comparison(runs: list[dict]) -> None:
    rows = []
    for run in runs:
        for name, value in numeric_metrics(run["metrics"]).items():
            rows.append({
                "Run": run["name"],
                "Metric": name,
                "Value": value,
            })

    if not rows:
        st.info("No numeric metrics available for comparison.")
        return

    df = pd.DataFrame(rows)
    common = df.groupby("Metric")["Run"].nunique()
    common_metrics = common[common >= 2].index.tolist()
    df = df[df["Metric"].isin(common_metrics)]

    if df.empty:
        st.info("Runs do not share any comparable scalar metrics.")
        return

    selected_metrics = st.multiselect(
        "Metrics to compare",
        options=sorted(common_metrics),
        default=[
            name for name in ("f1", "f1_macro", "accuracy", "rmse", "mae", "mse")
            if name in common_metrics
        ][:4],
    )

    if selected_metrics:
        fig = px.bar(
            df[df["Metric"].isin(selected_metrics)],
            x="Metric",
            y="Value",
            color="Run",
            barmode="group",
            title="Metric Comparison Across Runs",
            hover_data=["Run"],
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------

st.title("ML Pipeline Dashboard")
st.write(
    "Explore evaluation metrics, compare runs, and inspect model outputs."
)

with st.sidebar:
    st.header("Data Sources")

    metrics_uri = st.text_input(
        "Metrics JSON",
        value="runs/run-001/metrics/metrics.json",
        help="Path to a compute_metrics.py metrics.json artifact.",
    )
    report_uri = st.text_input(
        "Evaluation report JSON (optional)",
        value="runs/run-001/metrics/evaluation_report.json",
    )
    manifest_uri = st.text_input(
        "Manifest JSON (optional)",
        value="runs/run-001/metrics/manifest.json",
    )
    predictions_uri = st.text_input(
        "Predictions JSONL (optional)",
        value="runs/run-001/inference/predictions.jsonl",
    )

    st.divider()
    st.subheader("Record Field Mapping")
    target_field = st.text_input("Actual label / target field", "label")
    prediction_field = st.text_input("Prediction field", "prediction")
    prediction_value_path = st.text_input(
        "Nested prediction path (optional)",
        value="",
        help="For example, enter 'prediction' if the record's prediction "
             "field contains {'prediction': ...}.",
    )

    st.divider()
    comparison_text = st.text_area(
        "Other metrics.json paths (optional)",
        value="",
        help="Enter one metrics JSON path per line to compare runs.",
    )
    load_button = st.button("Load / Refresh Dashboard", type="primary")


# A manual refresh button clears cached artifacts before reloading.
if load_button:
    load_json.clear()
    load_jsonl.clear()

if not metrics_uri.strip():
    st.info("Enter a metrics JSON path in the sidebar to begin.")
    st.stop()

try:
    current_run = load_run(metrics_uri, report_uri, manifest_uri)
except Exception as exc:
    st.error(f"Could not load the metrics artifacts: {exc}")
    st.stop()

# ---------------------------------------------------------------------
# Run overview
# ---------------------------------------------------------------------

st.subheader("Run Overview")

top_left, top_right, top_third, top_fourth = st.columns(4)
top_left.metric("Run ID", str(current_run["name"]))
top_right.metric("Task", str(current_run["task"]).title())
top_third.metric("Status", str(current_run["status"]).replace("_", " ").title())
top_fourth.metric(
    "Records Evaluated",
    str(current_run["counts"].get("evaluated_records", "—")),
)

show_metric_cards(current_run)

with st.expander("Run metadata and configuration"):
    st.json({
        "metrics_uri": current_run["metrics_uri"],
        "counts": current_run["counts"],
        "configuration": current_run["metrics_doc"].get("configuration", {}),
        "manifest": current_run["manifest"],
        "report": current_run["report"],
    })


# ---------------------------------------------------------------------
# Metric charts
# ---------------------------------------------------------------------

st.subheader("Metrics")

chart_tab, table_tab = st.tabs(["Charts", "Metric Values"])
with chart_tab:
    plot_metric_bars(current_run["metrics"])

with table_tab:
    scalar_values = numeric_metrics(current_run["metrics"])
    if scalar_values:
        metric_df = pd.DataFrame(
            [{"Metric": key, "Value": value}
             for key, value in scalar_values.items()]
        )
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
    else:
        st.info("No scalar metrics are available.")


# ---------------------------------------------------------------------
# Task-specific analysis
# ---------------------------------------------------------------------

if current_run["task"] == "classification":
    st.subheader("Classification Analysis")
    class_tab, per_class_tab = st.tabs([
        "Confusion Matrix", "Per-Class Metrics"
    ])
    with class_tab:
        plot_confusion_matrix(current_run["metrics"])
    with per_class_tab:
        plot_per_class(current_run["metrics"])

elif current_run["task"] == "regression":
    st.subheader("Regression Analysis")
    if predictions_uri.strip():
        try:
            records = load_jsonl(predictions_uri)

            if prediction_value_path.strip():
                # Resolve a configured nested prediction value.
                for record in records:
                    value = get_field(record, prediction_field)
                    record["_dashboard_prediction"] = get_field(
                        {"value": value},
                        "value." + prediction_value_path.strip(),
                    )
                plot_regression_records(
                    records,
                    target_field,
                    "_dashboard_prediction",
                )
            else:
                plot_regression_records(
                    records, target_field, prediction_field
                )
        except Exception as exc:
            st.warning(f"Could not load regression prediction records: {exc}")
    else:
        st.info("Enter a predictions JSONL path to view regression plots.")


# ---------------------------------------------------------------------
# Compare runs
# ---------------------------------------------------------------------

st.subheader("Compare Runs")

comparison_runs = [current_run]
comparison_paths = [
    line.strip()
    for line in comparison_text.splitlines()
    if line.strip()
]

for path in comparison_paths:
    try:
        comparison_runs.append(load_run(path, "", ""))
    except Exception as exc:
        st.warning(f"Could not load comparison run {path}: {exc}")

if len(comparison_runs) > 1:
    plot_run_comparison(comparison_runs)
else:
    st.caption("Add one or more metrics.json paths in the sidebar to compare runs.")


# ---------------------------------------------------------------------
# Record-level inspection
# ---------------------------------------------------------------------

st.subheader("Record Inspection")

if not predictions_uri.strip():
    st.info("Enter a predictions JSONL path in the sidebar to inspect records.")
else:
    try:
        records = load_jsonl(predictions_uri)
        if not records:
            st.info("The predictions artifact contains no records.")
        else:
            table_rows = []
            for index, record in enumerate(records):
                inference_meta = record.get("_inference", {})
                labeling_meta = record.get("_labeling", {})
                table_rows.append({
                    "Index": index,
                    "Record ID": inference_meta.get("record_id")
                        or record.get("record_id")
                        or record.get("id")
                        or str(index),
                    "Actual": get_field(record, target_field),
                    "Prediction": unwrap_prediction(
                        get_field(record, prediction_field)
                    ),
                    "Inference Status": inference_meta.get("status", "unknown"),
                    "Label Type": labeling_meta.get("label_type", "unknown"),
                    "Error": inference_meta.get("error", ""),
                })

            table = pd.DataFrame(table_rows)
            status_values = sorted(
                str(value) for value in table["Inference Status"].dropna().unique()
            )
            chosen_statuses = st.multiselect(
                "Filter by inference status",
                status_values,
                default=status_values,
            )
            filtered = table[
                table["Inference Status"].astype(str).isin(chosen_statuses)
            ]

            search = st.text_input("Search records by ID, label, or prediction")
            if search:
                mask = filtered.astype(str).apply(
                    lambda column: column.str.contains(
                        search, case=False, na=False
                    )
                ).any(axis=1)
                filtered = filtered[mask]

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
            )

            if not filtered.empty:
                chosen_id = st.selectbox(
                    "Inspect a record",
                    filtered["Record ID"].astype(str).tolist(),
                )
                selected_row = filtered[
                    filtered["Record ID"].astype(str) == chosen_id
                ].iloc[0]
                selected_index = int(selected_row["Index"])
                with st.expander("Full record JSON"):
                    st.json(records[selected_index])

    except Exception as exc:
        st.warning(f"Could not load record-level data: {exc}")


# ---------------------------------------------------------------------
# Warnings and limitations
# ---------------------------------------------------------------------

st.subheader("Evaluation Warnings")

warnings = current_run["report"].get("limitations", [])
if warnings:
    for warning in warnings:
        st.warning(warning)
else:
    st.caption("No limitations were listed in the evaluation report.")

excluded = current_run["report"].get("excluded_records", [])
if excluded:
    with st.expander(f"Excluded records ({len(excluded)})"):
        st.dataframe(pd.DataFrame(excluded), use_container_width=True)

st.caption(
    "This dashboard visualizes saved artifacts. It does not rerun models "
    "or independently verify ground-truth labels."
)