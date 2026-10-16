import pandas as pd

from evals.progress import METRIC_KEYS, describe_progress, is_complete


def _df(key, values):
    return pd.DataFrame({"question": [f"q{i}" for i in range(len(values))], key: values})


def _full(nan_in=None):
    out = {k: _df(k, [0.5, 0.6, 0.7]) for k in METRIC_KEYS}
    if nan_in:
        out[nan_in] = _df(nan_in, [0.5, float("nan"), 0.7])
    return out


def test_none_and_empty_are_not_complete():
    assert not is_complete(None)
    assert not is_complete({})


def test_all_six_scored_is_complete():
    assert is_complete(_full())


def test_a_nan_sample_makes_it_incomplete():
    assert not is_complete(_full(nan_in="faithfulness"))


def test_a_missing_experiment_makes_it_incomplete():
    partial = _full()
    del partial["tool_correctness"]
    assert not is_complete(partial)


def test_describe_progress_counts_unscored_samples():
    text = describe_progress(_full(nan_in="context_recall"))
    assert "1 sample" in text and "Context Recall" in text


def test_describe_progress_lists_experiments_not_started():
    partial = {"faithfulness": _df("faithfulness", [0.5, 0.6, 0.7])}
    text = describe_progress(partial)
    assert "1 of 6" in text
