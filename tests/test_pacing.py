import asyncio

from evals.pacing import TokenPacer


class FakeClock:
    """Deterministic time: sleeping advances the clock. Fails fast if the pacer loops forever."""

    def __init__(self, max_sleeps=200):
        self.now = 0.0
        self.sleeps = 0
        self.max_sleeps = max_sleeps

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps += 1
        assert self.sleeps <= self.max_sleeps, "pacer never released the sample (livelock)"
        self.now += seconds


def _pacer(clock, estimate):
    return TokenPacer(limit=8000, initial_estimate=estimate, clock=clock, sleep=clock.sleep)


def test_empty_bucket_waits_then_starts():
    clock = FakeClock()
    pacer = _pacer(clock, estimate=3000)
    pacer.observe(remaining=0, limit=8000, used=0)
    waited = asyncio.run(pacer.wait())
    assert 0 < waited < 60


def test_full_bucket_starts_immediately():
    clock = FakeClock()
    pacer = _pacer(clock, estimate=3000)
    pacer.observe(remaining=8000, limit=8000, used=0)
    assert asyncio.run(pacer.wait()) == 0


def test_estimate_larger_than_bucket_still_starts():
    # estimate + reserve (7,000 + 1,500) exceeds the 8,000-token bucket, which can never hold that much.
    clock = FakeClock()
    pacer = _pacer(clock, estimate=7000)
    pacer.observe(remaining=8000, limit=8000, used=0)
    assert asyncio.run(pacer.wait()) == 0


def test_second_sample_is_spaced_by_the_share_rule():
    clock = FakeClock()
    pacer = _pacer(clock, estimate=3000)
    pacer.observe(remaining=8000, limit=8000, used=0)
    asyncio.run(pacer.wait())
    pacer.observe(remaining=8000, limit=8000, used=0)
    assert asyncio.run(pacer.wait()) > 0


def test_reset_bucket_forgets_the_previous_keys_level():
    # After switching to a different Groq key the old bucket's level no longer applies.
    clock = FakeClock()
    pacer = _pacer(clock, estimate=3000)
    pacer.observe(remaining=0, limit=8000, used=0)
    pacer.reset_bucket()
    assert asyncio.run(pacer.wait()) == 0
