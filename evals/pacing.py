"""
Token-bucket pacing for the RAGAS judge, driven by what Groq reports instead of fixed sleeps.

Why: the old code slept a flat 40s between samples (+62s between experiments) — ~52 of ~58 minutes
of a run was `sleep()`, sized for a 6,000 TPM Llama limit and blind to real usage.

How: Groq refills a per-model token bucket continuously (limit/60 tokens per second) and returns
`x-ratelimit-remaining-tokens` on every response. We record that after each judge call, estimate
how many tokens the next sample costs (moving average of what samples really used), and wait only
as long as needed for the bucket to cover it — while holding back a share of the quota so other
traffic on the same model is not starved.
"""

import asyncio
import json
import time
from typing import Awaitable, Callable, Optional


class TokenPacer:
    def __init__(
        self,
        limit: int = 8000,          # Groq tokens-per-minute for the judge model (corrected by headers)
        share: float = 0.75,        # fraction of the quota the eval may use on average
        reserve: int = 1500,        # tokens always left in the bucket for other callers
        initial_estimate: int = 3000,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.limit = limit
        self.share = share
        self.reserve = reserve
        self.estimate = float(initial_estimate)
        self._clock = clock
        self._sleep = sleep
        self._remaining: Optional[float] = None   # last remaining-tokens header
        self._observed_at = 0.0
        self._used_since_wait = 0
        self._last_start: Optional[float] = None
        self.total_waited = 0.0
        self.total_tokens = 0

    # ── fed by the HTTP response hook ────────────────────────────────────────
    def observe(self, remaining: Optional[int], limit: Optional[int], used: int) -> None:
        if limit:
            self.limit = limit
        if remaining is not None:
            self._remaining = float(remaining)
            self._observed_at = self._clock()
        self._used_since_wait += used
        self.total_tokens += used

    def reset_bucket(self) -> None:
        """Forget the last observed bucket level (the judge switched to a different Groq key/bucket)."""
        self._remaining = None

    def _available_now(self) -> float:
        """Bucket level now: last observation plus continuous refill, capped at the limit."""
        if self._remaining is None:
            return float(self.limit)
        refill = (self._clock() - self._observed_at) * self.limit / 60.0
        return min(float(self.limit), self._remaining + refill)

    def _seconds_needed(self) -> float:
        # Capped at the bucket size: waiting for more tokens than the bucket can hold never ends
        # (a sample estimate of 6,650 + 1,500 reserve > 8,000 made the eval wait forever).
        need = min(self.estimate + self.reserve, float(self.limit))
        deficit = max(0.0, need - self._available_now())
        wait_for_bucket = deficit * 60.0 / self.limit
        # Average-rate cap: never start samples faster than `share` of the quota allows.
        wait_for_share = 0.0
        if self._last_start is not None:
            min_gap = self.estimate * 60.0 / (self.limit * self.share)
            wait_for_share = max(0.0, self._last_start + min_gap - self._clock())
        return max(wait_for_bucket, wait_for_share)

    async def wait(self, label: str = "", status_cb: Optional[Callable[[str], None]] = None) -> float:
        """Call right before starting a sample. Sleeps only as long as necessary; returns seconds waited."""
        # Learn the real cost of the sample that just finished.
        if self._used_since_wait > 0:
            self.estimate = 0.5 * self.estimate + 0.5 * self._used_since_wait
            self._used_since_wait = 0

        waited = 0.0
        while True:
            need = self._seconds_needed()
            if need <= 0.05:
                break
            step = min(need, 10.0)
            if status_cb:
                status_cb(f"⏳ {label}: pacing {need:.0f}s for the Groq token budget "
                          f"(~{self.estimate:.0f} tokens/sample, {self._available_now():.0f} available)...")
            await self._sleep(step)
            waited += step
        self.total_waited += waited
        self._last_start = self._clock()
        return waited


def make_response_hook(pacer: TokenPacer):
    """httpx 'response' event hook: reads Groq's rate-limit headers + usage and feeds the pacer."""
    async def hook(response) -> None:
        try:
            await response.aread()
            used = int((json.loads(response.content).get("usage") or {}).get("total_tokens") or 0)
        except Exception:
            used = 0
        h = response.headers
        remaining = h.get("x-ratelimit-remaining-tokens")
        limit = h.get("x-ratelimit-limit-tokens")
        pacer.observe(
            int(remaining) if remaining and remaining.isdigit() else None,
            int(limit) if limit and limit.isdigit() else None,
            used,
        )
    return hook
