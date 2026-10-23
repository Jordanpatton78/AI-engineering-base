"""QRD tests for Section 3.8 (retrieval-augmented generation) and 3.9 (multi-agent analysis).

Most tests run offline. A bag-of-words encoder stands in for the BERT encoder, an
in-memory cosine search stands in for BigQuery VECTOR_SEARCH, and conftest's
FakeOllamaServer stands in for the model endpoint. The BigQuery round trip (3.8.b
against a live table, 12.6.a) runs only when RAG_TEST_TABLE is set.
"""
from __future__ import annotations

import json
import math
import os

import pytest

import rag
from conftest import FakeOllamaServer, ollama_config, read_jsonl, requires_bigquery, run_inference_stage

VOCAB = ["revenue", "churn", "anomaly", "invoice", "customer", "quality", "missing", "march"]


class BagOfWordsEncoder:
    """Deterministic stand-in for the BERT encoder: one dimension per vocabulary word."""

    model_name = "test-bow"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = [w.strip(".,?!").lower() for w in text.split()]
            vectors.append([float(words.count(term)) for term in VOCAB])
        return vectors


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def in_memory_search(rows: list[dict]):
    """Mimics VECTOR_SEARCH with cosine distance over stored rows."""
    def search(table, vector, top_k, run_id, use_brute_force):
        scored = [
            {
                "chunk_id": row["chunk_id"],
                "source_uri": row["source_uri"],
                "record_id": row["record_id"],
                "chunk_text": row["chunk_text"],
                "distance": 1.0 - cosine(vector, row["embedding"]),
            }
            for row in rows if row["run_id"] == run_id
        ]
        return sorted(scored, key=lambda r: r["distance"])[:top_k]
    return search


def fixed_search(hits: list[dict], calls: list | None = None):
    """Returns the same hits for every query. Hits carry chunk_id, chunk_text, and distance."""
    def search(table, vector, top_k, run_id, use_brute_force):
        if calls is not None:
            calls.append(table)
        return [{"source_uri": "gs://docs/a.txt", "record_id": "r1", **hit} for hit in hits][:top_k]
    return search


DOCUMENTS = [
    ("c-revenue", "revenue grew this quarter"),
    ("c-churn", "customer churn increased in march"),
    ("c-invoice", "invoice missing customer id"),
    ("c-quality", "data quality report shows missing values"),
]


def indexed_rows(run_id: str = "run-1") -> list[dict]:
    encoder = BagOfWordsEncoder()
    vectors = encoder.embed([text for _, text in DOCUMENTS])
    return [
        {"chunk_id": chunk_id, "source_uri": "gs://docs/corpus.txt", "record_id": chunk_id,
         "chunk_text": text, "run_id": run_id, "embedding": vector}
        for (chunk_id, text), vector in zip(DOCUMENTS, vectors)
    ]


def write_index_manifest(tmp_path, *, model="test-bow", dimension=len(VOCAB), status="succeeded", run_id="run-1"):
    path = tmp_path / "index_manifest.json"
    path.write_text(json.dumps({
        "status": status,
        "run_id": run_id,
        "embedding_model": model,
        "embedding_dimension": dimension,
        "index_version": "v-test",
    }), encoding="utf-8")
    return path


def retrieval_config(tmp_path, **overrides) -> dict:
    config = {
        "vector_table": "my-project.my_dataset.chunk_vectors",
        "index_manifest": str(write_index_manifest(tmp_path)),
        "embedding_model": "test-bow",
        "query_field": "question",
        "top_k": 2,
        "min_similarity": 0.1,
        "max_context_tokens": 500,
        "require_context": True,
    }
    config.update(overrides)
    return config


def make_retriever(config, search, encoder=None):
    return rag.Retriever(config, encoder=encoder or BagOfWordsEncoder(), searcher=search)


@pytest.fixture
def patched_retriever(monkeypatch):
    """Route inference.py's rag.Retriever(config) through the offline encoder and search."""
    real = rag.Retriever
    rows = indexed_rows()

    def install(search=None, encoder=None):
        monkeypatch.setattr(
            rag, "Retriever",
            lambda config: real(config, encoder=encoder or BagOfWordsEncoder(), searcher=search or in_memory_search(rows)),
        )
    return install


# ---------------------------------------------------------------------
# 3.8 Retrieval-augmented generation
# ---------------------------------------------------------------------

def test_3_8_a_relevant_passage_is_retrieved(tmp_path):
    """QRD 3.8.a: a query matching a passage's content returns that chunk first."""
    retriever = make_retriever(retrieval_config(tmp_path, top_k=3), in_memory_search(indexed_rows()))

    hits = retriever.retrieve("customer churn in march")

    assert hits[0]["chunk_id"] == "c-churn"


def test_3_8_b_bigquery_store_rows_match_chunks_and_dimension(tmp_path, monkeypatch):
    """QRD 3.8.b: every chunk becomes a row with an embedding of the configured dimension and model version."""
    written = []
    monkeypatch.setattr(rag, "load_sources", lambda config: ([
        {"source_uri": "bq://p.d.t", "record_id": "1", "text": "revenue grew this quarter customer churn in march"},
        {"source_uri": "bq://p.d.t", "record_id": "2", "text": "invoice missing customer id"},
    ], []))
    monkeypatch.setattr(rag, "write_vector_rows", lambda table, rows, disposition: written.extend(rows))

    manifest = rag.build_index(
        {"embedding_model": "test-bow", "vector_table": "p.d.vectors", "sources": [],
         "chunk_tokens": 4, "chunk_overlap": 0},
        "run-b", str(tmp_path / "out"), encoder=BagOfWordsEncoder(),
    )

    assert written and len(written) == manifest["chunk_count"]
    assert all(len(row["embedding"]) == len(VOCAB) for row in written)
    assert all(row["embedding_model"] == "test-bow" for row in written)
    assert manifest["embedding_dimension"] == len(VOCAB)


@requires_bigquery
def test_3_8_b_bigquery_round_trip(tmp_path):
    """QRD 3.8.b (live): build from a real table and read back the stored vectors."""
    table = os.environ["RAG_TEST_TABLE"]
    config = {"sources": [{"type": "bigquery", "table": table, "id_field": "record_id", "text_fields": ["text"]}],
              "embedding_model": os.environ.get("RAG_TEST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
              "vector_table": table + "_vectors", "chunk_tokens": 128, "chunk_overlap": 16}
    manifest = rag.build_index(config, "live-run", str(tmp_path / "out"))
    assert manifest["chunk_count"] > 0


def test_3_8_c_gcs_documents_parse_and_unsupported_types_are_listed(tmp_path):
    """QRD 3.8.c: text, Markdown, and JSONL objects parse; other suffixes are not supported and get skipped."""
    text_docs = rag.parse_gcs_object("gs://b/notes.md", ".md", b"# Title\nrevenue grew", ["text"])
    jsonl_docs = rag.parse_gcs_object(
        "gs://b/rows.jsonl", ".jsonl",
        b'{"id": "a", "text": "churn rose"}\n{"id": "b", "text": "invoice missing"}\n', ["text"],
    )

    assert text_docs[0]["text"].startswith("# Title")
    assert [d["record_id"] for d in jsonl_docs] == ["a", "b"]
    assert ".xlsx" not in rag.SUPPORTED_GCS_SUFFIXES
    with pytest.raises(ValueError):
        rag.parse_gcs_object("gs://b/sheet.xlsx", ".xlsx", b"PK", ["text"])


def test_3_8_d_mismatched_embedding_model_refuses_retrieval(tmp_path):
    """QRD 3.8.d: a query encoder that differs from the index's encoder is refused."""
    config = retrieval_config(tmp_path, embedding_model="other-encoder")

    with pytest.raises(rag.RetrievalError, match="Embedding model mismatch"):
        rag.Retriever(config, encoder=BagOfWordsEncoder(), searcher=in_memory_search([]))


def test_3_8_e_context_budget_drops_lowest_scores_and_records_count(tmp_path):
    """QRD 3.8.e: the prompt stays within budget, lower-scored chunks are dropped, and the drop count is recorded."""
    hits = [
        {"chunk_id": "c1", "chunk_text": "a b c d", "distance": 0.1},
        {"chunk_id": "c2", "chunk_text": "e f g h", "distance": 0.2},
        {"chunk_id": "c3", "chunk_text": "i j k l", "distance": 0.3},
    ]
    retriever = make_retriever(retrieval_config(tmp_path, top_k=3, max_context_tokens=9), fixed_search(hits))

    outcome = retriever.build_prompt("what happened")

    assert outcome["retrieval"]["chunk_ids"] == ["c1", "c2"]
    assert outcome["retrieval"]["dropped_for_budget"] == 1
    assert "[c3]" not in outcome["prompt"]


def test_3_8_f_citations_reach_the_model_and_the_manifest(tmp_path, patched_retriever):
    """QRD 3.8.f: each chunk the model sees carries its chunk ID, and the inference record lists them."""
    patched_retriever()
    with FakeOllamaServer(["Churn rose in March [c-churn]."]) as server:
        code, out = run_inference_stage(
            tmp_path, ollama_config(server, retrieval=retrieval_config(tmp_path, query_field="question")),
            [{"id": "q1", "question": "customer churn in march"}],
        )

    assert code == 0
    prompt = server.requests[0]["messages"][-1]["content"]
    assert "[c-churn] customer churn increased in march" in prompt
    record = read_jsonl(out / "predictions.jsonl")[0]
    assert record["_inference"]["retrieval"]["chunk_ids"][0] == "c-churn"
    assert record["_inference"]["retrieval"]["index_version"] == "v-test"


def test_3_8_g_empty_retrieval_stops_the_record_when_context_is_required(tmp_path, patched_retriever):
    """QRD 3.8.g (require_context true): the record fails and no request reaches the model."""
    patched_retriever(search=fixed_search([]))
    with FakeOllamaServer() as server:
        code, out = run_inference_stage(
            tmp_path, ollama_config(server, retrieval=retrieval_config(tmp_path, require_context=True)),
            [{"id": "q1", "question": "anything"}],
        )

    record = read_jsonl(out / "predictions.jsonl")[0]
    assert code == 2
    assert record["_inference"]["status"] == "failed"
    assert server.requests == []


def test_3_8_g_empty_retrieval_continues_with_warning_when_context_is_optional(tmp_path, patched_retriever):
    """QRD 3.8.g (require_context false): the model answers the bare question, and the warning is recorded."""
    patched_retriever(search=fixed_search([]))
    with FakeOllamaServer(["No context was available."]) as server:
        code, out = run_inference_stage(
            tmp_path, ollama_config(server, retrieval=retrieval_config(tmp_path, require_context=False)),
            [{"id": "q1", "question": "anything"}],
        )

    record = read_jsonl(out / "predictions.jsonl")[0]
    assert code == 0
    assert record["_inference"]["status"] == "succeeded"
    assert "warning" in record["_inference"]["retrieval"]
    assert server.requests[0]["messages"][-1]["content"] == "anything"


def test_3_8_h_without_retrieval_the_model_receives_the_raw_text(tmp_path):
    """QRD 3.8.h: with no retrieval block, no retrieval record is written and the prompt is the input text."""
    with FakeOllamaServer(["plain answer"]) as server:
        code, out = run_inference_stage(tmp_path, ollama_config(server), [{"id": "q1", "text": "hello"}])

    record = read_jsonl(out / "predictions.jsonl")[0]
    assert code == 0
    assert server.requests[0]["messages"][-1]["content"] == "hello"
    assert "retrieval" not in record["_inference"]


def test_3_8_i_repeated_retrieval_returns_the_same_chunks_in_order(tmp_path):
    """QRD 3.8.i: the same index, query, and k give the same chunk IDs in the same order."""
    retriever = make_retriever(retrieval_config(tmp_path, top_k=3, min_similarity=0.0), in_memory_search(indexed_rows()))

    first = [hit["chunk_id"] for hit in retriever.retrieve("customer missing values")]
    second = [hit["chunk_id"] for hit in retriever.retrieve("customer missing values")]

    assert first == second and first


def test_3_8_j_retrieval_searches_only_the_configured_table(tmp_path):
    """QRD 3.8.j: searches go only to the configured vector table, and table names are validated."""
    calls: list[str] = []
    retriever = make_retriever(retrieval_config(tmp_path), fixed_search([], calls))

    retriever.retrieve("anything")

    assert calls == ["my-project.my_dataset.chunk_vectors"]
    with pytest.raises(ValueError):
        rag.validate_table("my-project.my_dataset.t; DROP TABLE x")


def test_3_8_k_retrieval_on_a_classifier_is_rejected_before_execution(tmp_path):
    """QRD 3.8.k: a retrieval block on a non-language-model adapter fails validation."""
    config = {
        "adapter": {"type": "sklearn", "model_path": str(tmp_path / "missing.joblib"), "feature_columns": ["x"]},
        "retrieval": retrieval_config(tmp_path),
    }

    code, out = run_inference_stage(tmp_path, config, [{"id": "q1", "x": 1}])

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 1
    assert manifest["status"] == "failed"
    assert "language-model adapter" in json.dumps(manifest)


# ---------------------------------------------------------------------
# 3.9 Multi-agent analysis with LangChain
# ---------------------------------------------------------------------

@pytest.fixture
def agents(tmp_path):
    pytest.importorskip("langchain_core")
    import agents as agents_module

    return agents_module


class ScriptedAdapter:
    """Model adapter test double that returns scripted replies in order."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def predict_batch(self, records: list[dict]) -> list[dict]:
        self.prompts.append(records[0]["_agent_prompt"])
        reply = self.replies.pop(0) if self.replies else '{"next": "FINISH", "summary": "done"}'
        return [{"text": reply}]


def quality_artifact(tmp_path) -> str:
    path = tmp_path / "run-a" / "quality_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"missing_values": 3}', encoding="utf-8")
    return str(path)


def agent_config(tmp_path, specialists=None, **overrides) -> dict:
    config = {
        "objective": "Check data quality for this run.",
        "allowed_artifact_prefixes": [str(tmp_path)],
        "allowed_bigquery_datasets": ["my-project.my_dataset"],
        "max_steps": 3,
        "max_supervisor_rounds": 4,
        "specialists": specialists or [
            {"name": "quality", "instructions": "Read the quality report.", "tools": ["read_artifact"]},
        ],
    }
    config.update(overrides)
    return config


def run_scripted(agents, config, replies):
    adapter = ScriptedAdapter(replies)
    result = agents.run_agents(
        config, [{"id": 1, "x": 1.0}],
        build_model_adapter=lambda model_config: adapter,
        default_model_config={"type": "ollama", "model_name": "fake"},
    )
    return result, adapter


def test_3_9_a_supervisor_delegates_to_specialists_and_finishes(agents, tmp_path):
    """QRD 3.9.a: the supervisor routes a subtask to the named specialist, then finishes."""
    path = quality_artifact(tmp_path)
    replies = [
        '{"next": "quality", "task": "Read the quality report."}',
        json.dumps({"tool": "read_artifact", "args": {"uri": path}}),
        json.dumps({"final": "Three missing values.", "citations": [path]}),
        '{"next": "FINISH", "summary": "Quality issues found."}',
    ]

    result, _ = run_scripted(agents, agent_config(tmp_path), replies)

    assert result["status"] == "completed"
    assert result["findings"][0]["agent"] == "quality"
    assert any(e["event"] == "delegated" and e["agent"] == "quality" for e in result["trace"])
    assert result["summary"] == "Quality issues found."


def test_3_9_b_tool_outside_the_allowlist_is_refused_and_recorded(agents, tmp_path):
    """QRD 3.9.b: a specialist calling a tool it was not given is refused, and the refusal is in the trace."""
    replies = [
        '{"next": "quality", "task": "Check the data."}',
        json.dumps({"tool": "bigquery_select", "args": {"query": "SELECT 1"}}),
        json.dumps({"final": "No data was read.", "citations": []}),
        '{"next": "FINISH", "summary": "Done."}',
    ]

    result, _ = run_scripted(agents, agent_config(tmp_path), replies)

    refused = [e for e in result["trace"] if e["event"] == "tool_refused"]
    assert refused and refused[0]["tool"] == "bigquery_select"
    assert result["findings"][0]["status"] == "completed"


def test_3_9_c_write_and_multi_statement_queries_are_rejected(agents):
    """QRD 3.9.c: only single SELECT statements over allowed datasets pass the read-only check."""
    allowed = ["my-project.my_dataset"]
    agents.check_select_only("SELECT * FROM `my-project.my_dataset.records`", allowed)

    for query in [
        "DELETE FROM `my-project.my_dataset.records` WHERE true",
        "SELECT 1; DROP TABLE `my-project.my_dataset.records`",
        "SELECT * FROM `other-project.other.records`",
    ]:
        with pytest.raises(PermissionError):
            agents.check_select_only(query, allowed)


def test_3_9_d_step_limit_stops_the_agent_and_keeps_the_partial_trace(agents, tmp_path):
    """QRD 3.9.d: reaching max_steps ends the specialist with limit_reached and keeps its trace."""
    path = quality_artifact(tmp_path)
    call = json.dumps({"tool": "read_artifact", "args": {"uri": path}})
    replies = ['{"next": "quality", "task": "Loop."}', call, call, '{"next": "FINISH", "summary": "Stopped."}']

    result, _ = run_scripted(agents, agent_config(tmp_path, max_steps=2), replies)

    assert result["findings"][0]["status"] == "limit_reached"
    assert any(e["event"] == "limit_reached" for e in result["trace"])
    assert sum(1 for e in result["trace"] if e["event"] == "tool_call") == 2


def test_3_9_e_trace_records_replies_and_tool_calls_with_hashes(agents, tmp_path):
    """QRD 3.9.e: every reply and tool call is in the trace; tool calls carry an argument hash and a result summary."""
    path = quality_artifact(tmp_path)
    replies = [
        '{"next": "quality", "task": "Read."}',
        json.dumps({"tool": "read_artifact", "args": {"uri": path}}),
        json.dumps({"final": "ok", "citations": [path]}),
        '{"next": "FINISH", "summary": "ok"}',
    ]

    result, _ = run_scripted(agents, agent_config(tmp_path), replies)

    replies_logged = [e for e in result["trace"] if e["event"] == "model_reply"]
    tool_calls = [e for e in result["trace"] if e["event"] == "tool_call"]
    assert len(replies_logged) == 4
    assert tool_calls[0]["args_hash"] and "missing_values" in tool_calls[0]["result_summary"]
    assert [e["seq"] for e in result["trace"]] == list(range(1, len(result["trace"]) + 1))


def test_3_9_f_only_citations_seen_in_tool_results_are_verified(agents, tmp_path):
    """QRD 3.9.f: a citation to a read artifact is kept; a citation to something never read is marked unverified."""
    path = quality_artifact(tmp_path)
    replies = [
        '{"next": "quality", "task": "Read."}',
        json.dumps({"tool": "read_artifact", "args": {"uri": path}}),
        json.dumps({"final": "Three missing.", "citations": [path, "invented-chunk"]}),
        '{"next": "FINISH", "summary": "ok"}',
    ]

    result, _ = run_scripted(agents, agent_config(tmp_path), replies)

    finding = result["findings"][0]
    assert finding["provenance"] == "model_generated"
    assert finding["citations"] == [path]
    assert finding["unverified_citations"] == ["invented-chunk"]


def test_3_9_g_unreachable_model_endpoint_produces_no_findings(tmp_path):
    """QRD 3.9.g: an unreachable model endpoint fails the agent stage with a reason and no findings."""
    import inference

    config = agent_config(tmp_path)
    llm_config = {"type": "ollama", "model_name": "fake", "base_url": "http://127.0.0.1:9"}

    result = inference.run_agent_stage(config, [{"id": 1}], "run-g", str(tmp_path / "out"), llm_config, None)

    assert result["status"] == "failed"
    findings = json.loads((tmp_path / "out" / "agent_findings.json").read_text(encoding="utf-8"))
    assert not findings.get("findings")
    assert findings["error"]


def test_3_9_h_new_specialist_runs_from_configuration_alone(agents, tmp_path):
    """QRD 3.9.h: a specialist added only in configuration is routed to and runs, with no supervisor change."""
    path = quality_artifact(tmp_path)
    specialists = [
        {"name": "quality", "instructions": "Read the quality report.", "tools": ["read_artifact"]},
        {"name": "anomaly_explainer", "instructions": "Explain anomalies from the artifact.", "tools": ["read_artifact"]},
    ]
    replies = [
        '{"next": "anomaly_explainer", "task": "Explain."}',
        json.dumps({"final": "No anomalies.", "citations": []}),
        '{"next": "FINISH", "summary": "Done."}',
    ]

    result, _ = run_scripted(agents, agent_config(tmp_path, specialists=specialists), replies)

    assert result["findings"][0]["agent"] == "anomaly_explainer"


def test_3_9_i_agent_failure_leaves_prediction_results_unchanged(tmp_path):
    """QRD 3.9.i: an agent stage failure is recorded, and the prediction file is unchanged."""
    with FakeOllamaServer(["prediction answer"]) as server:
        config = ollama_config(server, agents={
            "objective": "Summarize.",
            "model": {"type": "ollama", "model_name": "fake", "base_url": "http://127.0.0.1:9"},
            "specialists": [{"name": "quality", "instructions": "Read.", "tools": ["read_artifact"]}],
        })
        code, out = run_inference_stage(tmp_path, config, [{"id": "q1", "text": "question"}])

    predictions = read_jsonl(out / "predictions.jsonl")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert code == 0
    assert predictions[0]["prediction"]["text"] == "prediction answer"
    assert manifest["agents"]["status"] == "failed"
