"""Text clean-up for model output that is stored in the conversation state."""

NUL = "\x00"


def strip_nul(text: str) -> str:
    """
    Remove NUL characters. Postgres JSON(B) rejects the \\u0000 escape, and the planner model
    (gpt-oss via Groq) ends some search queries with a NUL (measured: 6/6 for an SR-IOV question), so
    the LangGraph checkpoint write failed ("unsupported Unicode escape sequence") AFTER the answer
    had already been generated and the user got "internal error".
    """
    return text.replace(NUL, "")
