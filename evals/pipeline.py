"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each golden sample.
Captures: actual_response (capped at RESPONSE_TRUNCATE chars), actual_contexts (from sources),
and actual_tools_called (detected from thought_process).
"""


import time
import uuid
import copy
import json
import os
import requests
import logfire

from evals.answer_check import failed_indices, is_failed_answer

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/") + "/query"
# The judge scores this text, so it must hold the whole answer (a 250-word answer is ~1,700 chars).
# At 300 chars only the "Sure! Here's a guide…" intro survived and Faithfulness/Correctness were meaningless.
RESPONSE_TRUNCATE = int(os.getenv("EVAL_RESPONSE_CHARS", "2500"))
# Seconds between questions. One technical answer costs ~3,800 tokens (planner + answer) and the main
# model (gpt-oss-120b) allows 8,000 tokens/min, i.e. ~30s per question. Faster than that and requests
# hit 429s and fall back to the smaller model, which would skew the eval.
DELAY_BETWEEN_CALLS = 30
# A backend failure ("internal error") is transient (Groq 429, guardrail timeout) and would be scored as
# a 0 on every metric, so each question is asked up to MAX_ATTEMPTS times, waiting between tries.
MAX_ATTEMPTS = int(os.getenv("EVAL_MAX_ATTEMPTS", "3"))
RETRY_DELAY = int(os.getenv("EVAL_RETRY_DELAY", "60"))




def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    main.py sets:  'Intent: Guardrails Fired'                → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def _ask_once(sample: dict, thread_id: str) -> dict:
    """One /query call → the fields to store on the sample. Never raises."""
    try:
        resp = requests.post(
            API_URL,
            json={"q": sample["question"], "thread_id": thread_id, "use_cache": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_answer = data.get("answer") or ""
        sources = data.get("sources") or []
        fields = {
            "actual_response": raw_answer[:RESPONSE_TRUNCATE],
            "actual_contexts": sources[:5],
            "actual_tools_called": [detect_tool(data.get("thought_process") or [])],
        }
        logfire.info(
            "✅ Response captured",
            tool=fields["actual_tools_called"][0],
            response_chars=len(raw_answer),
            context_chunks=len(sources),
        )
        return fields
    except requests.exceptions.ConnectionError:
        logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
    except Exception as e:
        logfire.error(f"❌ Query failed: {e}")
    return {
        "actual_response": "",
        "actual_contexts": sample.get("relevant_contexts", []),
        "actual_tools_called": ["unknown"],
    }


def _ask_with_retries(sample: dict, thread_id: str) -> dict:
    fields = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        fields = _ask_once(sample, f"{thread_id}_try{attempt}")
        if not is_failed_answer(fields["actual_response"]):
            return fields
        if attempt < MAX_ATTEMPTS:
            logfire.warning(f"⚠️ Backend failed on attempt {attempt}/{MAX_ATTEMPTS}; retrying in {RETRY_DELAY}s")
            time.sleep(RETRY_DELAY)
    return fields


def rerun_failed(dataset: dict, progress_callback=None) -> dict:
    """Copy of `dataset` where only the failed (error/empty) answers are asked again."""
    result = copy.deepcopy(dataset)
    samples = result["rag_samples"]
    todo = failed_indices(samples)
    run_id = uuid.uuid4().hex[:8]
    for k, i in enumerate(todo):
        sample = samples[i]
        if progress_callback:
            progress_callback(k, len(todo), sample["question"], "calling")
        sample.update(_ask_with_retries(sample, f"eval_{run_id}_{i}"))
        if progress_callback:
            progress_callback(k, len(todo), sample["question"], "done", sample["actual_response"])
        if k < len(todo) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)
    return result


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with actual_response, actual_contexts, actual_tools_called filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    # Conversation memory is persisted in Postgres, so a thread id reused across runs carries the
    # previous run's question + answer. The planner then sees the question as "answerable from
    # history", skips retrieval, and every sample scores as direct_answer with zero contexts.
    # A fresh id per run keeps each sample on a clean thread.
    run_id = uuid.uuid4().hex[:8]
    n = len(samples)

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                sample.update(_ask_with_retries(sample, f"eval_{run_id}_{i}"))

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
        
        
def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path) as f:
        return json.load(f)
