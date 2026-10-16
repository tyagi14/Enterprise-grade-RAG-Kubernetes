from app.agents.nodes.planner import build_planner_prompt
from app.agents.nodes.responder import (
    TRUNCATION_NOTICE,
    build_conversational_prompt,
    mark_if_truncated,
)


def test_planner_prompt_sends_tool_comparisons_and_preferences_to_search():
    p = build_planner_prompt("", "Which would you prefer, Kubernetes or Docker?").lower()
    assert "comparison" in p and "recommendation" in p
    assert "which would you prefer" in p            # named as a technical example


def test_planner_prompt_no_longer_treats_every_opinion_as_conversational():
    p = build_planner_prompt("", "x")
    assert "an opinion," not in p


def test_planner_prompt_keeps_greetings_and_memory_conversational():
    p = build_planner_prompt("User: my name is Sam\n", "what is my name").lower()
    assert "greeting" in p and "conversation history" in p and "conversational" in p


def test_conversational_prompt_forbids_answering_technical_content_from_general_knowledge():
    p = build_conversational_prompt("User: hi\n", "thanks!").lower()
    assert "general knowledge" in p
    assert "documentation" in p                     # steer to the docs instead
    assert "thanks!" in p and "user: hi" in p


def test_truncated_answer_gets_a_notice():
    out = mark_if_truncated("Kubernetes approach", "length")
    assert out.startswith("Kubernetes approach") and out.endswith(TRUNCATION_NOTICE)


def test_complete_answer_is_left_alone():
    assert mark_if_truncated("done", "stop") == "done"
    assert mark_if_truncated("done", None) == "done"
