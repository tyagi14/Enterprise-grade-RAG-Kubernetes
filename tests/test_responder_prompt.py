from app.agents.nodes.responder import build_technical_prompt


def _prompt(context="CONTENT: run kubectl get pods", history="", question="How do I list pods?"):
    return build_technical_prompt(context, history, question)


def test_prompt_contains_context_history_and_question():
    p = _prompt(context="CONTENT: ALPHA", history="User: earlier\n", question="BETA?")
    assert "ALPHA" in p and "earlier" in p and "BETA?" in p


def test_prompt_restricts_answers_to_the_context():
    p = _prompt().lower()
    assert "only" in p and "technical context" in p
    assert "general knowledge" in p          # explicitly forbidden


def test_prompt_says_what_to_do_when_the_context_lacks_the_answer():
    p = _prompt().lower()
    assert "not covered" in p or "does not contain" in p


def test_prompt_asks_for_short_direct_answers_with_exact_commands():
    p = _prompt().lower()
    assert "direct answer" in p and "exact" in p


def test_prompt_no_longer_asks_for_250_words():
    assert "250" not in _prompt()
