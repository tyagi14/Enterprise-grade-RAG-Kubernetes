import asyncio
import logging
import logfire
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


# NVIDIA's hosted endpoint is the primary gate, but it is unreliable (models stall, return
# CUDA errors, or get retired — measured: gemma-4-31b-it 10/10 timeouts, nemoguard-8b-topic-control
# HTTP 500). A normal call takes ~1s, so give up on a hung one quickly.
NVIDIA_TIMEOUT_SECONDS = 5
# Fallback gate: the SAME NeMo rail and policy prompt, run on Groq. Used only when every NVIDIA
# call failed, so an NVIDIA outage degrades latency (~5s) instead of taking the app down.
FALLBACK_TIMEOUT_SECONDS = 8

# Primary tier: one LLMRails per configured NVIDIA model, called in parallel (first answer wins).
_nvidia_rails: list[LLMRails] = []
_fallback_rails: LLMRails | None = None


def _rails_config() -> RailsConfig:
    return RailsConfig.from_content(colang_content=COLANG_CONTENT, yaml_content=YAML_CONTENT)


def _build_nvidia_rails(model: str) -> LLMRails:
    llm = ChatNVIDIA(
        api_key=settings.NVIDIA_API_KEY,
        model=model,
        temperature=0,
        timeout=NVIDIA_TIMEOUT_SECONDS,
    )
    return LLMRails(_rails_config(), llm=llm)


def _build_fallback_rails() -> LLMRails | None:
    if not (settings.GUARD_FALLBACK_ENABLED and settings.GROQ_API_KEY):
        return None
    extra = {}
    if settings.GUARD_FALLBACK_REASONING_EFFORT:
        extra["reasoning_effort"] = settings.GUARD_FALLBACK_REASONING_EFFORT   # keep the yes/no check cheap
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GUARD_FALLBACK_MODEL,
        temperature=0,
        timeout=FALLBACK_TIMEOUT_SECONDS,
        max_retries=0,
        **extra,
    )
    return LLMRails(_rails_config(), llm=llm)


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails gates at app startup:
      - primary: one gate per model in NVIDIA_GUARD_MODELS (NVIDIA hosted endpoint, NVIDIA_API_KEY)
      - fallback: the same rail on Groq, used only if the NVIDIA tier fails
    Both use non-reasoning-style yes/no prompting — reasoning models ignore NeMo's *intent
    classification* prompts and answer directly, which would silently disable every rail; the
    `self check input` policy prompt used here is a plain yes/no question and works on both.
    """
    global _nvidia_rails, _fallback_rails

    if not settings.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is not set — cannot start the guardrails gate.")
    if not settings.NVIDIA_GUARD_MODELS:
        raise RuntimeError("NVIDIA_GUARD_MODELS is empty — no guard model configured.")

    # NeMo logs every prompt and internal event at INFO (including user messages) — keep
    # Cloud Run logs to warnings and above.
    logging.getLogger("nemoguardrails").setLevel(logging.CRITICAL)

    _nvidia_rails = [_build_nvidia_rails(model) for model in settings.NVIDIA_GUARD_MODELS]
    _fallback_rails = _build_fallback_rails()
    fallback = f"{settings.GUARD_FALLBACK_MODEL} on Groq" if _fallback_rails else "disabled"
    logfire.info(
        f"🛡️ NeMo Guardrails initialised (primary: {', '.join(settings.NVIDIA_GUARD_MODELS)} on NVIDIA; "
        f"fallback: {fallback})."
    )


def _messages(message: str) -> list[dict]:
    return [{"role": "user", "content": message}]


async def _hedged_generate(message: str):
    """
    Ask every NVIDIA guard model at once and return the first successful answer, cancelling the
    rest. A single hung model then costs nothing. Raises the last error only if all failed.
    """
    pending = {
        asyncio.create_task(rails.generate_async(messages=_messages(message), options={"rails": ["input"]}))
        for rails in _nvidia_rails
    }
    last_error: Exception | None = None
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception() is None:
                for other in pending:
                    other.cancel()
                return task.result()
            last_error = task.exception()
    raise last_error


def _generate(message: str):
    """
    Tier 1: NVIDIA (hedged). Tier 2: Groq fallback. If both fail the error propagates and the
    request fails closed — a message is never waved through unchecked.
    """
    try:
        result = asyncio.run(_hedged_generate(message))
        logfire.info("🛡️ Gate served by NVIDIA.")
        return result
    except Exception as nvidia_error:
        if _fallback_rails is None:
            raise
        logfire.warning(f"⚠️ NVIDIA guardrail tier failed ({nvidia_error}); using Groq fallback gate.")

    result = _fallback_rails.generate(messages=_messages(message), options={"rails": ["input"]})
    logfire.warning("🛡️ Gate served by FALLBACK (Groq) — NVIDIA tier was unavailable.")
    return result


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if not _nvidia_rails:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _generate(message)
        replies = result.response or []
        content = replies[-1].get("content", "") if replies else ""

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
