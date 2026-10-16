from types import SimpleNamespace

import app.agents.nodes.planner as planner


class _StubLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _prompt):
        return SimpleNamespace(content=self._content)


def _run(monkeypatch, llm_output, message="How do I configure SR-IOV?"):
    monkeypatch.setattr(planner, "llm", _StubLLM(llm_output))
    return planner.planner_node({"messages": [{"role": "user", "content": message}], "plan": ["Start"]})


def test_trailing_nul_is_removed_from_search_query(monkeypatch):
    out = _run(monkeypatch, "configure SR-IOV on Intel NICs for Kubernetes\x00")
    assert out["current_query"] == "configure SR-IOV on Intel NICs for Kubernetes"
    assert "\x00" not in "".join(out["plan"]) + out["status"]


def test_nul_after_conversational_still_routes_conversationally(monkeypatch):
    out = _run(monkeypatch, "CONVERSATIONAL\x00", message="thanks")
    assert out["current_query"] == "CONVERSATIONAL"


def test_nul_only_decision_counts_as_empty_and_falls_back(monkeypatch):
    out = _run(monkeypatch, "\x00", message="explain BGP")
    assert out["current_query"] == "explain BGP"
    assert out["plan"][0] == "Intent: Technical (planner fallback)"
