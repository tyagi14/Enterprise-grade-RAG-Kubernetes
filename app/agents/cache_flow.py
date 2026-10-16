"""
Policy for when the Redis semantic cache may be used.

A cached answer is shared across users, so it must never depend on conversation history:
  - READ/WRITE only on the first turn of a thread (no prior messages), and
  - WRITE only for technical answers that went through retrieval.
Conversational turns ("what is my name?", "expand on that") are never cached.
"""

import logfire

TECHNICAL_MARKER = "Intent: Technical"


def is_first_turn(agent, config: dict) -> bool:
    """True when the thread has no stored messages yet. Fails closed (False) on any error."""
    try:
        state = agent.get_state(config)
        return not (state.values or {}).get("messages")
    except Exception as e:
        logfire.warning(f"⚠️ Could not read thread state, skipping semantic cache: {e}")
        return False


def is_cacheable(final_output: dict) -> bool:
    """True for a finished technical answer that was grounded in retrieved documents."""
    plan = final_output.get("plan") or []
    has_answer = bool((final_output.get("final_answer") or "").strip())
    return has_answer and TECHNICAL_MARKER in plan and bool(final_output.get("documents"))


def remember_exchange(agent, config: dict, question: str, answer: str) -> None:
    """
    A cache hit bypasses the graph, so write the exchange into the thread's memory
    ourselves — otherwise follow-ups ("expand on that") would have no history.
    """
    try:
        agent.update_state(
            config,
            {"messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]},
            as_node="responder",
        )
    except Exception as e:
        logfire.warning(f"⚠️ Could not store cached exchange in thread memory: {e}")
