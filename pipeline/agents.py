#!/usr/bin/env python3
"""
LangChain multi-agent analysis, run by inference.py after predictions are written.

A supervisor agent receives an objective and a summary of the dataset, then routes
subtasks to specialist agents. Each specialist runs a small tool loop: it asks the
model for one JSON action at a time (call a tool, or finish with a finding) until
it finishes or reaches its step limit. Every model reply and tool call is written
to a trace.

Tools are read-only:
  bigquery_select    one SELECT statement against allowed datasets
  read_artifact      a file under an allowed GCS or local prefix
  retrieve_context   RAG retrieval (only when a retrieval block is configured)

Agent config (the "agents" block in an inference config):
{
  "objective": "Summarize data-quality risks and any anomalies in this run.",
  "max_supervisor_rounds": 6,
  "max_steps": 6,
  "max_rows": 50,
  "max_tool_result_chars": 2000,
  "allowed_bigquery_datasets": ["my-project.my_dataset"],
  "allowed_artifact_prefixes": ["gs://my-bucket/runs/run-001/"],
  "model": {"type": "ollama", "model_name": "llama3.1", "base_url": "http://localhost:11434"},
  "specialists": [
    {"name": "data_quality_reviewer",
     "instructions": "Check the quality report for missing values and duplicates.",
     "tools": ["bigquery_select", "read_artifact"]},
    {"name": "evidence_checker",
     "instructions": "Find supporting passages for each claim.",
     "tools": ["retrieve_context"]}
  ]
}

The "model" block is optional; when omitted, the inference adapter is used.
LangChain is imported only when this module is loaded, so the other stages run
without it installed.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool, tool

import rag

from common import (
    utc_now,
)


AGENT_PROMPT_FIELD = "_agent_prompt"
SUPPORTED_TOOLS = ("bigquery_select", "read_artifact", "retrieve_context")
WRITE_OR_DDL_KEYWORDS = re.compile(
    r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|call|export|load)\b",
    re.IGNORECASE,
)
TABLE_REFERENCE = re.compile(r"\b(?:from|join)\s+`?([A-Za-z0-9_.-]+)`?", re.IGNORECASE)


# ---------------------------------------------------------------------
# Model wrapper: any registered language-model adapter as a LangChain chat model
# ---------------------------------------------------------------------

class AdapterChatModel(BaseChatModel):
    """Routes LangChain chat calls through a pipeline model adapter's predict_batch()."""

    adapter: Any

    @property
    def _llm_type(self) -> str:
        return "pipeline-adapter"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n\n".join(f"[{message.type.upper()}]\n{message.content}" for message in messages)
        output = self.adapter.predict_batch([{AGENT_PROMPT_FIELD: prompt}])[0]
        text = output.get("text") if isinstance(output, dict) else None
        if text is None:
            raise RuntimeError("Agent model adapters must return a 'text' field.")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=str(text)))])


# ---------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------

def check_select_only(query: str, allowed_datasets: list[str]) -> None:
    """Raise PermissionError unless the query is one SELECT over allowed datasets."""
    statement = query.strip().rstrip(";").strip()
    if ";" in statement:
        raise PermissionError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", statement, re.IGNORECASE):
        raise PermissionError("Only SELECT statements are allowed.")
    if WRITE_OR_DDL_KEYWORDS.search(statement):
        raise PermissionError("The statement contains a write or DDL keyword.")

    targets = TABLE_REFERENCE.findall(statement)
    for target in targets:
        if not any(target.startswith(dataset + ".") for dataset in allowed_datasets):
            raise PermissionError(f"Table {target} is outside the allowed datasets.")


def make_tools(agent_config: dict, retriever: Any) -> dict[str, BaseTool]:
    allowed_datasets = list(agent_config.get("allowed_bigquery_datasets", []))
    allowed_prefixes = list(agent_config.get("allowed_artifact_prefixes", []))
    max_rows = int(agent_config.get("max_rows", 50))

    @tool
    def bigquery_select(query: str) -> list[dict]:
        """Run one read-only SELECT against allowed BigQuery datasets. Returns up to max_rows rows."""
        from google.cloud import bigquery

        check_select_only(query, allowed_datasets)
        job = bigquery.Client().query(query)
        return [dict(row.items()) for row in job.result(max_results=max_rows)]

    @tool
    def read_artifact(uri: str) -> dict:
        """Read a pipeline artifact (report, manifest, or JSONL) under an allowed prefix."""
        if not any(uri.startswith(prefix) for prefix in allowed_prefixes):
            raise PermissionError("Artifact is outside the allowed prefixes.")
        return {"uri": uri, "content": rag.read_text(uri)}

    tools: dict[str, BaseTool] = {
        "bigquery_select": bigquery_select,
        "read_artifact": read_artifact,
    }

    if retriever is not None:
        @tool
        def retrieve_context(question: str) -> list[dict]:
            """Find passages in the run's vector index that are relevant to a question. Returns chunk IDs."""
            return retriever.retrieve(question)

        tools["retrieve_context"] = retrieve_context

    return tools


def collect_references(value: Any) -> set[str]:
    """Chunk IDs and URIs in a tool result. Findings may cite only these."""
    refs: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("chunk_id", "source_uri", "uri") and isinstance(item, str):
                refs.add(item)
            else:
                refs |= collect_references(item)
    elif isinstance(value, list):
        for item in value:
            refs |= collect_references(item)
    return refs


# ---------------------------------------------------------------------
# Parsing and trace
# ---------------------------------------------------------------------

def parse_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def add_trace(trace: list[dict], actor: str, event: str, **data: Any) -> None:
    trace.append({"seq": len(trace) + 1, "at": utc_now(), "actor": actor, "event": event, **data})


def summarize_dataset(records: list[dict]) -> dict:
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    return {
        "record_count": len(records),
        "fields": fields[:50],
        "sample": [json.dumps(record, default=str)[:300] for record in records[:2]],
    }


# ---------------------------------------------------------------------
# Specialist loop
# ---------------------------------------------------------------------

def run_specialist(
    spec: dict,
    task: str,
    llm: BaseChatModel,
    tools: dict[str, BaseTool],
    trace: list[dict],
    seen_refs: set[str],
    max_steps: int,
    max_result_chars: int,
) -> dict:
    tool_lines = "\n".join(f"- {name}: {item.description}" for name, item in tools.items())
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system}"),
        ("human", "{task}"),
    ])
    system = (
        f"You are the {spec['name']} agent. {spec['instructions']}\n\n"
        "Reply with exactly one JSON object and nothing else.\n"
        'To call a tool: {"tool": "<name>", "args": {...}}\n'
        'To finish: {"final": "<finding text>", "citations": ["<chunk_id or URI>"]}\n'
        f"Available tools:\n{tool_lines}"
    )
    messages: list[BaseMessage] = list(prompt.format_messages(system=system, task=task))

    for step in range(1, max_steps + 1):
        reply = llm.invoke(messages).content
        messages.append(AIMessage(content=reply))
        add_trace(trace, spec["name"], "model_reply", step=step, content=reply)

        action = parse_json_object(reply)
        if action is None:
            messages.append(HumanMessage(content="Your reply was not a JSON object. Reply with exactly one JSON object."))
            continue

        if "final" in action:
            citations = [str(c) for c in action.get("citations", [])]
            verified = [c for c in citations if c in seen_refs]
            unverified = [c for c in citations if c not in seen_refs]
            add_trace(trace, spec["name"], "finished", step=step, unverified_citations=unverified)
            return {
                "agent": spec["name"],
                "task": task,
                "status": "completed",
                "provenance": "model_generated",
                "text": str(action["final"]),
                "citations": verified,
                "unverified_citations": unverified,
            }

        name = action.get("tool")
        if name not in tools:
            add_trace(trace, spec["name"], "tool_refused", step=step, tool=name)
            messages.append(HumanMessage(content=f"Tool {name!r} is not available to you. Use a listed tool or finish."))
            continue

        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        try:
            result: Any = tools[name].invoke(args)
            outcome = "ok"
        except Exception as exc:
            result = {"error": str(exc)}
            outcome = "error"

        seen_refs |= collect_references(result)
        serialized = json.dumps(result, default=str, ensure_ascii=False)
        summary = serialized[:max_result_chars]
        add_trace(
            trace, spec["name"], "tool_call", step=step, tool=name, outcome=outcome,
            args_hash=rag.sha256_text(json.dumps(args, sort_keys=True, default=str)),
            result_summary=summary,
        )
        messages.append(HumanMessage(content=f"Tool result from {name}:\n{summary}"))

    add_trace(trace, spec["name"], "limit_reached", max_steps=max_steps)
    return {
        "agent": spec["name"],
        "task": task,
        "status": "limit_reached",
        "text": None,
        "citations": [],
        "unverified_citations": [],
    }


# ---------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------

def validate_agent_config(agent_config: dict, retriever: Any) -> None:
    specialists = agent_config.get("specialists")
    if not specialists:
        raise ValueError("agents.specialists must list at least one specialist.")

    names = [spec.get("name") for spec in specialists]
    if len(set(names)) != len(names) or not all(names):
        raise ValueError("Each specialist needs a unique name.")

    for spec in specialists:
        for tool_name in spec.get("tools", []):
            if tool_name not in SUPPORTED_TOOLS:
                raise ValueError(f"Unknown tool {tool_name!r} for {spec['name']}.")
            if tool_name == "retrieve_context" and retriever is None:
                raise ValueError(f"{spec['name']} uses retrieve_context, which needs a retrieval block.")


def run_agents(
    agent_config: dict,
    records: list[dict],
    build_model_adapter: Callable[[dict], Any],
    default_model_config: dict,
    retriever: Any = None,
) -> dict:
    """Run the supervisor loop. Returns findings and trace; the caller writes the artifacts."""
    validate_agent_config(agent_config, retriever)

    model_config = agent_config.get("model") or default_model_config
    llm = AdapterChatModel(adapter=build_model_adapter(model_config))
    tools_all = make_tools(agent_config, retriever)
    specialists = {spec["name"]: spec for spec in agent_config["specialists"]}
    max_rounds = int(agent_config.get("max_supervisor_rounds", 6))
    max_steps = int(agent_config.get("max_steps", 6))
    max_chars = int(agent_config.get("max_tool_result_chars", 2000))

    objective = agent_config.get("objective", "Analyze this dataset.")
    roster = "\n".join(f"- {name}: {spec.get('instructions', '')}" for name, spec in specialists.items())
    supervisor_system = (
        f"You coordinate specialist agents to meet this objective: {objective}\n"
        f"Dataset summary: {json.dumps(summarize_dataset(records), default=str)}\n"
        f"Specialists:\n{roster}\n\n"
        "Reply with exactly one JSON object and nothing else:\n"
        '{"next": "<specialist name>", "task": "<what that specialist should do>"}\n'
        'or {"next": "FINISH", "summary": "<final summary of the findings>"}.'
    )
    messages: list[BaseMessage] = [SystemMessage(content=supervisor_system), HumanMessage(content="Start.")]

    trace: list[dict] = []
    findings: list[dict] = []
    seen_refs: set[str] = set()
    status = "limit_reached"
    summary = None
    rounds_used = 0

    for round_number in range(1, max_rounds + 1):
        rounds_used = round_number
        reply = llm.invoke(messages).content
        messages.append(AIMessage(content=reply))
        add_trace(trace, "supervisor", "model_reply", round=round_number, content=reply)

        decision = parse_json_object(reply)
        if decision is None:
            messages.append(HumanMessage(content="Reply with one JSON object as instructed."))
            continue

        if decision.get("next") == "FINISH":
            summary = str(decision.get("summary", ""))
            status = "completed"
            add_trace(trace, "supervisor", "finished", round=round_number)
            break

        spec = specialists.get(decision.get("next"))
        if spec is None:
            messages.append(HumanMessage(content="Unknown specialist. Choose from the roster, or FINISH."))
            continue

        tools = {name: tools_all[name] for name in spec.get("tools", [])}
        add_trace(trace, "supervisor", "delegated", round=round_number, agent=spec["name"])
        finding = run_specialist(
            spec, str(decision.get("task", objective)), llm, tools, trace,
            seen_refs, max_steps, max_chars,
        )
        findings.append(finding)
        messages.append(HumanMessage(
            content=f"{spec['name']} finished with status {finding['status']}:\n{finding['text']}"
        ))

    return {
        "stage": "agents",
        "status": status,
        "objective": objective,
        "summary": summary,
        "findings": findings,
        "counts": {
            "supervisor_rounds": rounds_used,
            "findings": len(findings),
            "tool_calls": sum(1 for e in trace if e["event"] == "tool_call"),
            "unverified_citations": sum(len(f["unverified_citations"]) for f in findings),
        },
        "trace": trace,
        "model_prompt_field": AGENT_PROMPT_FIELD,
        "finished_at": utc_now(),
    }
