import pandas as pd

from evals.answer_check import failed_indices, invalidate_scores, is_failed_answer

ERROR_ANSWER = "I apologize, but I encountered an internal error while processing your request. Please try again later."


def test_error_message_and_empty_answers_are_failures():
    assert is_failed_answer(ERROR_ANSWER)
    assert is_failed_answer("")
    assert is_failed_answer("   ")
    assert is_failed_answer(None)


def test_real_answers_are_not_failures():
    assert not is_failed_answer("Use `kubectl top nodes` to see CPU and memory.")
    assert not is_failed_answer("The documentation does not cover this.")


def test_failed_indices_lists_only_failed_samples():
    samples = [{"actual_response": "ok"}, {"actual_response": ERROR_ANSWER}, {"actual_response": ""}, {}]
    assert failed_indices(samples) == [1, 2, 3]


def test_invalidate_scores_blanks_only_the_given_questions_and_does_not_mutate():
    df = pd.DataFrame({"question": ["q1", "q2", "q3"], "faithfulness": [1.0, 0.0, 0.5]})
    results = {"faithfulness": df}
    out = invalidate_scores(results, ["q2"])
    assert out["faithfulness"]["faithfulness"].isna().tolist() == [False, True, False]
    assert df["faithfulness"].tolist() == [1.0, 0.0, 0.5]


def test_invalidate_scores_handles_empty_and_long_questions():
    long_q = "x" * 100
    df = pd.DataFrame({"question": [long_q[:65]], "answer_relevancy": [0.4]})
    out = invalidate_scores({"answer_relevancy": df}, [long_q])
    assert out["answer_relevancy"]["answer_relevancy"].isna().all()
    assert invalidate_scores(None, ["q"]) is None
