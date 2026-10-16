"""Is a saved/partial metric run finished, and what is still missing? Used by the evals page."""

from typing import Optional

import pandas as pd

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "answer_correctness": "Answer Correctness",
    "tool_correctness": "Tool Correctness",
}
METRIC_KEYS = tuple(METRIC_LABELS)


def _unscored(df: pd.DataFrame, key: str) -> int:
    return int(df[key].isna().sum())


def is_complete(results: Optional[dict]) -> bool:
    if not results:
        return False
    return all(k in results and _unscored(results[k], k) == 0 for k in METRIC_KEYS)


def describe_progress(results: Optional[dict]) -> str:
    """One line for the UI, e.g. '4 of 6 experiments fully scored; 3 samples unscored (Context Recall).'"""
    results = results or {}
    done = [k for k in METRIC_KEYS if k in results and _unscored(results[k], k) == 0]
    gaps = {METRIC_LABELS[k]: _unscored(results[k], k) for k in METRIC_KEYS if k in results and _unscored(results[k], k)}
    parts = [f"{len(done)} of {len(METRIC_KEYS)} experiments fully scored"]
    if gaps:
        total = sum(gaps.values())
        parts.append(f"{total} sample{'s' if total != 1 else ''} unscored ({', '.join(gaps)})")
    not_started = [METRIC_LABELS[k] for k in METRIC_KEYS if k not in results]
    if not_started:
        parts.append(f"not started: {', '.join(not_started)}")
    return "; ".join(parts) + "."
