"""
Detects Step 2 answers that are really backend failures, and clears their stale scores.

When the backend cannot answer (Groq 429, guardrail timeout) it replies "I apologize, but I
encountered an internal error…". Stored as a normal answer it scores 0 on every metric and drags the
averages down (4 of 15 samples did exactly that), so such samples are retried and re-scored.
"""

from typing import Optional

import pandas as pd

ERROR_MARKER = "encountered an internal error"
# Metrics whose scores depend on the stored answer (tool correctness is recomputed every run).
ANSWER_DEPENDENT_METRICS = (
    "faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness",
)
QUESTION_KEY_CHARS = 65  # metrics.py stores question[:65] as the row key


def is_failed_answer(response: Optional[str]) -> bool:
    text = (response or "").strip()
    return not text or ERROR_MARKER in text


def failed_indices(samples: list) -> list:
    return [i for i, s in enumerate(samples) if is_failed_answer(s.get("actual_response"))]


def invalidate_scores(results: Optional[dict], questions: list) -> Optional[dict]:
    """Copy of `results` with the answer-dependent scores of `questions` blanked (so Resume re-scores them)."""
    if results is None:
        return None
    keys = {q[:QUESTION_KEY_CHARS] for q in questions}
    out = dict(results)
    for metric in ANSWER_DEPENDENT_METRICS:
        df = results.get(metric)
        if df is None or metric not in df.columns:
            continue
        fresh = df.copy()
        fresh.loc[fresh["question"].isin(keys), metric] = float("nan")
        out[metric] = fresh
    return out
