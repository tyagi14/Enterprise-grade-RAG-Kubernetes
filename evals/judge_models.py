"""
Judge model fallback chain.

Groq caps every model at 200,000 tokens/DAY per ORGANISATION (a second API key in the SAME
organisation shares the same budget; a key from a different organisation has its own), and a full
eval needs ~400-500K judge tokens. When the current (model, key) pair runs out of daily budget
mid-run, the judge moves to the next pair and retries the same sample, instead of leaving the rest
of the run as NaN. Order: both keys on one model before falling back to the next model, so the judge
stays the same model for as long as possible.

openai/gpt-oss-120b is deliberately NOT in the default chain: it is the production answer model, and
a long eval would starve live chat.
"""

import os
from typing import NamedTuple, Optional

DEFAULT_CHAIN = ("qwen/qwen3.8-27b", "openai/gpt-oss-20b")


def reasoning_effort_for(model: str) -> str:
    """gpt-oss accepts low/medium/high, qwen accepts none/default. JUDGE_REASONING_EFFORT overrides."""
    override = os.getenv("JUDGE_REASONING_EFFORT")
    if override:
        return override
    return "low" if "gpt-oss" in model else "none"


def max_tokens_for(model: str) -> int:
    """
    Groq counts max_tokens toward the per-minute request size (limit 8,000 on the free tier), so a
    big output reservation on top of ~4-5K prompt tokens (5 whole chunks) makes the request "too
    large" and the sample fails. gpt-oss spends hidden reasoning tokens and needs headroom (an empty
    completion → json_validate_failed); qwen with reasoning off emits only the JSON.
    JUDGE_MAX_TOKENS overrides.
    """
    override = os.getenv("JUDGE_MAX_TOKENS")
    if override:
        return int(override)
    return 4096 if "gpt-oss" in model else 2048


def parse_chain(models_env: str = "", model_env: str = "") -> tuple[str, ...]:
    """
    JUDGE_MODELS (comma list) wins. Otherwise JUDGE_MODEL goes first, followed by the default chain
    without duplicates.
    """
    explicit = tuple(m.strip() for m in models_env.split(",") if m.strip())
    if explicit:
        return explicit
    first = model_env.strip()
    if not first:
        return DEFAULT_CHAIN
    return (first,) + tuple(m for m in DEFAULT_CHAIN if m != first)


def is_daily_limit(error_text: str) -> bool:
    return "tokens per day" in error_text or "(TPD)" in error_text


class JudgeStep(NamedTuple):
    model: str
    key_index: int


def build_steps(models: tuple[str, ...], key_count: int) -> tuple[JudgeStep, ...]:
    return tuple(JudgeStep(model, k) for model in models for k in range(max(1, key_count)))


class KeyHolder:
    """The API key currently used by the judge's HTTP client (read per request by an httpx hook)."""

    def __init__(self, keys: tuple[str, ...]):
        self._keys = keys
        self._index = 0

    @property
    def key(self) -> str:
        return self._keys[self._index]

    def use(self, index: int) -> None:
        self._index = min(index, len(self._keys) - 1)


class JudgeModelChain:
    """Switches the RAGAS judge LLM (its `model` and `model_args` are read on every call) and its key."""

    def __init__(self, llm, steps: tuple[JudgeStep, ...], keys: KeyHolder):
        self._llm = llm
        self._steps = steps
        self._keys = keys
        self._index = 0

    @property
    def current(self) -> JudgeStep:
        return self._steps[self._index]

    def advance(self) -> Optional[str]:
        """Move to the next (model, key). Returns the model name, or None when the chain is exhausted."""
        if self._index + 1 >= len(self._steps):
            return None
        self._index += 1
        step = self._steps[self._index]
        self._keys.use(step.key_index)
        self._llm.model = step.model
        self._llm.model_args = {
            **self._llm.model_args,
            "reasoning_effort": reasoning_effort_for(step.model),
            "max_tokens": max_tokens_for(step.model),
        }
        return step.model
