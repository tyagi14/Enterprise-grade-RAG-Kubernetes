import copy

import evals.pipeline as pipeline

ERROR_ANSWER = "I apologize, but I encountered an internal error while processing your request. Please try again later."


class _Resp:
    def __init__(self, answer):
        self._answer = answer

    def raise_for_status(self):
        pass

    def json(self):
        return {"answer": self._answer, "thought_process": ["Intent: Technical"], "sources": ["c1", "c2"]}


def _golden(n=3):
    return {"rag_samples": [{"id": i, "question": f"q{i}", "relevant_contexts": []} for i in range(n)], "guardrails_samples": []}


def _install(monkeypatch, answers_by_question):
    """answers_by_question: question -> list of answers returned on successive calls."""
    calls = []

    def fake_post(url, json, timeout):
        q = json["q"]
        calls.append(q)
        return _Resp(answers_by_question[q].pop(0))

    monkeypatch.setattr(pipeline.requests, "post", fake_post)
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    return calls


def test_failed_answer_is_retried_until_it_succeeds(monkeypatch):
    calls = _install(monkeypatch, {"q0": [ERROR_ANSWER, ERROR_ANSWER, "good"], "q1": ["fine"], "q2": ["fine"]})
    out = pipeline.run_pipeline(_golden())
    assert out["rag_samples"][0]["actual_response"] == "good"
    assert calls.count("q0") == 3


def test_gives_up_after_max_attempts_and_keeps_the_failure(monkeypatch):
    calls = _install(monkeypatch, {"q0": [ERROR_ANSWER] * 5, "q1": ["fine"], "q2": ["fine"]})
    out = pipeline.run_pipeline(_golden())
    assert calls.count("q0") == pipeline.MAX_ATTEMPTS
    assert out["rag_samples"][0]["actual_response"] == ERROR_ANSWER


def test_rerun_failed_only_calls_failed_samples_and_returns_a_copy(monkeypatch):
    dataset = _golden()
    for s, a in zip(dataset["rag_samples"], ["ok", ERROR_ANSWER, "ok"]):
        s["actual_response"] = a
        s["actual_contexts"] = ["x"]
        s["actual_tools_called"] = ["retrieve_documents"]
    before = copy.deepcopy(dataset)
    calls = _install(monkeypatch, {"q1": ["now fixed"]})
    out = pipeline.rerun_failed(dataset)
    assert calls == ["q1"]
    assert out["rag_samples"][1]["actual_response"] == "now fixed"
    assert out["rag_samples"][0]["actual_response"] == "ok"
    assert dataset == before
