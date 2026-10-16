import asyncio
from types import SimpleNamespace

import pandas as pd

import evals.metrics as M


class _NoWaitPacer:
    async def wait(self, *_args, **_kwargs):
        return 0.0

    def reset_bucket(self):
        pass


class _StubMetric:
    """Scores sample i as float(i); can be told to fail for some questions."""

    def __init__(self, fail_for=()):
        self.calls = []
        self.fail_for = set(fail_for)

    async def abatch_score(self, batch):
        q = batch[0]["user_input"]
        self.calls.append(q)
        if q in self.fail_for:
            raise RuntimeError("judge failed")
        return [SimpleNamespace(value=float(len(q)))]


def _samples(n=4):
    return [{"question": f"question-{i}"} for i in range(n)], [{"user_input": f"question-{i}"} for i in range(n)]


def _run(metric, inputs, samples, prior=None, checkpoint=None):
    M._pacer = _NoWaitPacer()
    M._chain = None
    return asyncio.run(M._batched_score(metric, inputs, samples, None, "T", prior=prior, checkpoint=checkpoint))


def test_already_scored_samples_are_not_scored_again():
    samples, inputs = _samples()
    metric = _StubMetric()
    scores = _run(metric, inputs, samples, prior=[0.5, None, 0.7, None])
    assert metric.calls == ["question-1", "question-3"]
    assert [s.value for s in scores][0] == 0.5 and [s.value for s in scores][2] == 0.7


def test_checkpoint_fires_after_every_sample_with_padded_scores():
    samples, inputs = _samples(3)
    seen = []
    _run(_StubMetric(), inputs, samples, checkpoint=lambda s: seen.append([x.value if x else None for x in s]))
    assert len(seen) == 3
    assert seen[0][1:] == [None, None] and seen[0][0] is not None
    assert all(v is not None for v in seen[-1])


def test_failed_sample_becomes_none_and_is_retried_on_resume():
    samples, inputs = _samples(2)
    metric = _StubMetric(fail_for={"question-1"})
    scores = _run(metric, inputs, samples)
    assert scores[1] is None and scores[0] is not None
    resumed = _StubMetric()
    _run(resumed, inputs, samples, prior=[scores[0].value, None])
    assert resumed.calls == ["question-1"]


def test_prior_values_reads_a_partial_dataframe_by_question():
    samples, _ = _samples(3)
    prior = {"faithfulness": pd.DataFrame({"question": ["question-0", "question-1", "question-2"],
                                           "faithfulness": [0.4, float("nan"), 0.9]})}
    assert M._prior_values(prior, "faithfulness", samples) == [0.4, None, 0.9]
    assert M._prior_values(None, "faithfulness", samples) == [None, None, None]
    assert M._prior_values(prior, "context_recall", samples) == [None, None, None]
