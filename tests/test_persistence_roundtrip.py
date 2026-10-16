import pandas as pd

from evals.persistence import _metrics_to_records, _records_to_metrics


def test_all_unscored_metric_comes_back_as_numeric_nan():
    df = pd.DataFrame({"question": ["a", "b"], "context_precision": [float("nan")] * 2})
    back = _records_to_metrics(_metrics_to_records({"context_precision": df}))["context_precision"]
    assert pd.api.types.is_float_dtype(back["context_precision"])
    assert back["context_precision"].isna().all()
    # The page formats scores with "{:.3f}"; this must not raise on unscored rows.
    back.style.format({"context_precision": "{:.3f}"}).to_html()


def test_partially_scored_metric_keeps_values():
    df = pd.DataFrame({"question": ["a", "b"], "faithfulness": [0.5, float("nan")]})
    back = _records_to_metrics(_metrics_to_records({"faithfulness": df}))["faithfulness"]
    assert back["faithfulness"].tolist()[0] == 0.5 and pd.isna(back["faithfulness"].tolist()[1])
    assert back["question"].tolist() == ["a", "b"]


def test_none_passes_through():
    assert _records_to_metrics(None) is None
