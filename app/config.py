import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- GCP Core ---
    PROJECT_ID = os.getenv("PROJECT_ID", "")   # required: set in .env / Cloud Run (Terraform sets it)
    LOCATION = os.getenv("LOCATION", "us-central1")
    GCP_DOC_AI_LOCATION = os.getenv("GCP_DOC_AI_LOCATION", "us")
    GCP_DOC_AI_PROCESSOR_ID = os.getenv("GCP_DOC_AI_PROCESSOR_ID")
    RAW_BUCKET = os.getenv("GCP_RAW_BUCKET", f"{PROJECT_ID}-rag-raw" if PROJECT_ID else "")
    PROCESSED_BUCKET = os.getenv("GCP_PROCESSED_BUCKET", f"{PROJECT_ID}-rag-processed" if PROJECT_ID else "")

    # --- Vector DB (Qdrant) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- LLM (Groq) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MODEL = "openai/gpt-oss-120b"        # main RAG pipeline
    GROQ_FAST_MODEL = "openai/gpt-oss-20b"    # gateway fallback, eval judge

    # --- LLM Gateway (Portkey) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    GROQ_SLUG = "rag"     # primary virtual key: @rag/<GROQ_MODEL>
    GROQ_SLUG_2 = "brag"  # fallback virtual key: @brag/<GROQ_FAST_MODEL>

    # --- Guardrails (NVIDIA) ---
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    # Non-reasoning models that answer the yes/no rail check cleanly, in priority order. Retries
    # alternate through this list, so one hosted model hanging (gemma-4-31b-it does intermittently)
    # does not take the gate down. Override with NVIDIA_GUARD_MODELS="a,b" — no rebuild needed.
    NVIDIA_GUARD_MODELS = [
        m.strip() for m in os.getenv(
            "NVIDIA_GUARD_MODELS",
            "meta/llama-3.2-90b-vision-instruct,meta/llama-3.2-90b-vision-instruct",  # same model twice = hedge its own stalls
        ).split(",") if m.strip()
    ]

    # Fallback gate (same NeMo rail, run on Groq) used only when the NVIDIA tier fails.
    GUARD_FALLBACK_ENABLED = os.getenv("GUARD_FALLBACK_ENABLED", "true").lower() == "true"
    # qwen3.8-27b, not gpt-oss-20b: Groq caps each model at 200,000 tokens/DAY per organisation and
    # gpt-oss-20b is also the answer-fallback and the eval judge, so it runs dry (seen 2026-10-13:
    # the fallback gate itself returned 429 and chat requests failed). qwen has its own daily budget
    # and answers the yes/no policy check with a clean one-word reply when reasoning is off.
    GUARD_FALLBACK_MODEL = os.getenv("GUARD_FALLBACK_MODEL", "qwen/qwen3.8-27b")
    GUARD_FALLBACK_REASONING_EFFORT = os.getenv("GUARD_FALLBACK_REASONING_EFFORT", "none")  # "" = don't send

    # --- Postgres Memory ---
    DB_HOST = os.getenv("DB_HOST")                # /cloudsql/... (Cloud Run) or hostname (local)
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_NAME = os.getenv("DB_NAME", "rag_memory")
    LOCAL_MODE = os.getenv("LOCAL_MODE", "true").lower() == "true"

    # --- Redis Semantic Cache ---
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    USE_SEMANTIC_CACHE = os.getenv("USE_SEMANTIC_CACHE", "false").lower() == "true"

    # --- Observability ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "entreprise_rag")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # --- Evals ---
    JUDGE_GROQ = os.getenv("JUDGE_GROQ")


# Apply LangSmith env vars for automatic LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "entreprise_rag")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()