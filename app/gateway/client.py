import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI

from app.config import settings


# Gateway setup:
#   - Config: the workspace has block_inline_config enabled, so the gateway is driven by the
#     SAVED Portkey config (PORTKEY_CONFIG_ID, a 'pc-...' slug) — inline dicts are rejected with 400.
#   - Models: sent explicitly per request ("@<virtual-key>/<model>"). The saved config does not
#     pick a provider on its own, so a request without a model fails.
#   - Fallback: done here in code (primary → fallback), not in the gateway config.
PRIMARY_MODEL = f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"
FALLBACK_MODEL = f"@{settings.GROQ_SLUG_2}/{settings.GROQ_FAST_MODEL}"
MODEL_CHAIN = (PRIMARY_MODEL, FALLBACK_MODEL)

# Token budget: the Groq plan allows 8,000 tokens/min per model for the whole organisation, and
# an uncapped answer used ~5,200 tokens (and was silently cut off at 3,072). Cap the completion so
# one technical question costs ~3,300 tokens and finishes cleanly. The responder prompt asks for
# ~250 words; this is the safety net.
MAX_COMPLETION_TOKENS = 1200

# max_retries=0: on a 429 the SDKs otherwise sleep and retry for tens of seconds. Fail fast and
# let MODEL_CHAIN fall back to the next model instead.
portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG_ID,
    max_retries=0,
)


# The planner makes a one-word decision, but gpt-oss spends hundreds of hidden "reasoning" tokens
# on it and — measured — sometimes finishes with an EMPTY answer (finish=stop, content='',
# 302 of 311 tokens spent reasoning), which sent plain statements like "My name is Sam." down the
# full retrieval path (~3,800 tokens). Low effort + a cap keeps the decision cheap and non-empty.
PLANNER_REASONING_EFFORT = "low"
PLANNER_MAX_TOKENS = 400


def _build_llm(model: str, feature: str) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=model,
        temperature=0,
        max_retries=0,
        max_tokens=PLANNER_MAX_TOKENS,
        extra_body={"reasoning_effort": PLANNER_REASONING_EFFORT},
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=settings.PORTKEY_CONFIG_ID,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )


def get_langchain_llm(feature: str = "rag"):
    """
    Returns a Portkey-backed ChatOpenAI (primary) with the fallback model attached —
    a drop-in for ChatGroq in LangChain nodes (same .invoke() interface).

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    primary = _build_llm(PRIMARY_MODEL, feature)
    fallback = _build_llm(FALLBACK_MODEL, feature)
    return primary.with_fallbacks([fallback])


def chat_completion(messages: list[dict], temperature: float = 0.1):
    """
    Native Portkey call (so response headers stay readable) that walks MODEL_CHAIN:
    returns the first successful response, re-raises the last error if all models fail.
    """
    last_error: Exception | None = None
    for model in MODEL_CHAIN:
        try:
            return portkey_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
        except Exception as e:
            last_error = e
            logfire.warning(f"⚠️ Gateway model {model} failed, trying next: {e}")
    raise last_error


def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
