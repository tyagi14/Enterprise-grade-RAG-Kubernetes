"""
Phase 2 — RAGAS + Tool Correctness metrics.
Uses the JUDGE_GROQ key with the judge model chain (evals/judge_models.py). Samples are scored one at a time and paced by
evals/pacing.py from Groq's own remaining-token counter, instead of fixed sleeps: it waits only as
long as the token bucket needs to refill, and learns what a sample really costs. Contexts are
capped by EVAL_CONTEXT_LIMIT / EVAL_CONTEXT_CHARS (default: all 5 retrieved chunks, whole).
"""


import os
import asyncio
from types import SimpleNamespace
from typing import Callable, Optional
import logfire
import httpx
import pandas as pd
from openai import AsyncOpenAI


from ragas.llms import llm_factory
from evals.pacing import TokenPacer, make_response_hook
from evals.judge_models import JudgeModelChain, KeyHolder, build_steps, is_daily_limit, max_tokens_for, parse_chain, reasoning_effort_for
from ragas.embeddings import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# llama-3.1-8b-instant was retired by Groq. Groq caps EVERY model at 200,000 tokens/DAY per
# organisation (extra API keys in the same organisation share it), and a full eval needs ~150K judge
# tokens. The judge therefore walks a chain (evals/judge_models.py): when one model's daily budget runs
# out it switches to the next and retries the sample. JUDGE_MODELS (comma list) sets the chain, or
# JUDGE_MODEL sets just the first model — both changeable without a rebuild.
JUDGE_MODELS = parse_chain(os.getenv("JUDGE_MODELS", ""), os.getenv("JUDGE_MODEL", ""))
JUDGE_MODEL = JUDGE_MODELS[0]
GENERAL_BATCH_SIZE = 1  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second
# What the judge sees of the retrieval. It used to be 2 chunks x 300 chars (~600 of the ~6,000 chars
# retrieved), so a correct chunk at rank 3+ or past its first 300 chars counted as a miss: Context
# Recall/Precision read 0.0 even when the gold chunk was fully in the top 5 (verified by direct search).
# Now: every retrieved chunk, WHOLE (chunks are ~2,500 chars, and the answer text often sits past
# char 1,200 — Context Recall for the Metrics Server question went 0.0 -> 1.0 with whole chunks).
# Measured cost: ~30-35K judge tokens per sample (~500K for 15), i.e. more than one free Groq model's
# whole day (200K) — the judge chain spreads it over several models. Lower these env vars only if the
# daily Groq budget is the constraint; scores then understate retrieval.
CONTEXT_TRUNCATE = int(os.getenv("EVAL_CONTEXT_CHARS", "3000"))  # chars per context chunk
CONTEXT_LIMIT = int(os.getenv("EVAL_CONTEXT_LIMIT", "5"))        # chunks passed to RAGAS per sample


# Set by run_all_metrics for the duration of a run (one eval runs at a time).
_pacer: TokenPacer = TokenPacer()
_chain: Optional[JudgeModelChain] = None


MAX_JUDGE_KEYS = 6


def _judge_keys() -> tuple[str, ...]:
    """
    Judge API keys in priority order: JUDGE_GROQ, then JUDGE_GROQ_2 ... JUDGE_GROQ_6 (or the
    JUDGE_GROQ_API_KEY[_N] spellings). Each extra key only adds daily budget when it belongs to a
    DIFFERENT Groq organisation — limits are per organisation, not per key.
    """
    # Deliberately no GROQ_API_KEY fallback: that is the PRODUCTION key, and the eval is expensive
    # (~400-500K tokens/run) — it must never silently spend production's daily budget.
    candidates = [os.getenv("JUDGE_GROQ") or os.getenv("JUDGE_GROQ_API_KEY")]
    for n in range(2, MAX_JUDGE_KEYS + 1):
        candidates.append(os.getenv(f"JUDGE_GROQ_{n}") or os.getenv(f"JUDGE_GROQ_API_KEY_{n}"))
    keys = tuple(dict.fromkeys(k for k in candidates if k))
    if not keys:
        raise ValueError("No judge API key set: provide JUDGE_GROQ (and optionally JUDGE_GROQ_2, JUDGE_GROQ_3, ...). The production GROQ_API_KEY is never used for judging.")
    return keys


def _build_judge(pacer: TokenPacer, holder: KeyHolder):
    # The hooks feed Groq's remaining-token header + real usage to the pacer after every judge call,
    # and stamp each request with the key the chain currently points at (so a switch needs no new client).
    async def use_current_key(request) -> None:
        request.headers["Authorization"] = f"Bearer {holder.key}"

    http_client = httpx.AsyncClient(
        event_hooks={"request": [use_current_key], "response": [make_response_hook(pacer)]}, timeout=120
    )
    # max_retries: the openai client honours Retry-After, which rides out Groq's 429s.
    client = AsyncOpenAI(api_key=holder.key, base_url=GROQ_BASE_URL, max_retries=3, http_client=http_client)
    llm = llm_factory(
        JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=max_tokens_for(JUDGE_MODEL),
        reasoning_effort=reasoning_effort_for(JUDGE_MODEL),
    )
    embeddings = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        use_api=False,
    )
    return llm, embeddings


def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated.
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks
    so a single RAGAS LLM call stays small.
    (Live contexts from Qdrant are ~1,500 chars each — without truncation a single
    Faithfulness request costs thousands of tokens against the per-minute limit.)
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    return pd.DataFrame([
        # None = the judge failed on this sample twice; keep the row as NaN so the mean skips it
        {"question": s["question"][:65], metric_key: round(float(r.value), 3) if r is not None else float("nan")}
        for s, r in zip(samples, scores)
    ])


def _prior_values(prior: Optional[dict], metric_key: str, samples: list) -> list:
    """Scores from an earlier (possibly partial) run, aligned to `samples`. None = still to score."""
    df = (prior or {}).get(metric_key)
    if df is None or metric_key not in df.columns:
        return [None] * len(samples)
    by_question = dict(zip(df["question"], df[metric_key]))
    values = []
    for s in samples:
        v = by_question.get(s["question"][:65])
        values.append(None if v is None or pd.isna(v) else float(v))
    return values


def _checkpointer(metric_key: str, samples: list, results: dict, on_progress: Optional[Callable]):
    """Builds the per-sample callback that reports the partially scored experiment."""
    if on_progress is None:
        return None

    def checkpoint(scores: list) -> None:
        on_progress({**results, metric_key: _score_df(metric_key, samples, scores)})
    return checkpoint


async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "",
                         prior: Optional[list] = None, checkpoint: Optional[Callable] = None) -> list:
    """
    Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
    Keeps each burst under the shared Groq token budget.

    A judge call that fails (empty/invalid JSON, 429 after retries) is retried once after a
    cooldown; if it fails again that sample scores None (NaN in the table) instead of aborting
    the whole ~50 minute run.

    prior: scores from an earlier run (None = not scored yet); those samples are reused, not re-scored.
    checkpoint(scores): called after every sample so the caller can persist progress — a browser
    refresh 54 minutes into a run used to throw away everything scored so far.
    """
    all_scores = []
    failed = 0
    last_error = ""
    batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        start = b_idx * GENERAL_BATCH_SIZE
        known = (prior or [None] * len(inputs))[start:start + len(batch)]
        if all(v is not None for v in known):
            all_scores.extend(SimpleNamespace(value=v) for v in known)
            continue
        # Every sample, including the first of a metric, waits only if the token bucket is short.
        await _pacer.wait(f"{label} sample {b_idx + 1}/{len(batches)}", status_cb)
        scores = None
        attempt = 0
        while attempt < 2:
            try:
                scores = await metric.abatch_score(batch)
                break
            except Exception as e:
                last_error = str(e)
                logfire.warning(f"⚠️ {label} sample {b_idx + 1}/{len(batches)} failed (attempt {attempt + 1}): {e}")
                next_model = _chain.advance() if _chain and is_daily_limit(last_error) else None
                if next_model:
                    # Daily budget of the current judge model is gone: not a failed attempt, switch and redo.
                    logfire.warning(f"⚠️ Judge daily token limit reached — switching judge to {next_model}")
                    _pacer.reset_bucket()
                    if status_cb:
                        status_cb(f"🔀 Daily token limit reached — continuing with {next_model} (key {_chain.current.key_index + 1}).")
                    continue
                attempt += 1
                if attempt == 1:
                    await _pacer.wait(f"{label} retry {b_idx + 1}", status_cb)
        if scores is None:
            failed += len(batch)
            scores = [None] * len(batch)
        all_scores.extend(scores)
        if checkpoint:
            checkpoint(all_scores + [None] * (len(inputs) - len(all_scores)))
    if failed and status_cb:
        hint = ""
        if is_daily_limit(last_error):
            hint = " Cause: the Groq DAILY token limit is used up on every judge model in the chain — set JUDGE_MODELS to models with budget left, or wait."
        status_cb(f"⚠️ {label}: {failed} of {len(inputs)} samples could not be scored and are shown as NaN.{hint} Last error: {last_error[:200]}")
    return all_scores


async def run_all_metrics(golden_dataset: dict, status_cb=None, prior: Optional[dict] = None,
                          on_progress: Optional[Callable] = None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    prior: results of an earlier (partial) run — scored samples are reused, only NaN/missing ones are scored.
    on_progress(results: dict): called after every scored sample with everything computed so far.
    """
    global _pacer, _chain
    _pacer = TokenPacer()
    keys = _judge_keys()
    holder = KeyHolder(keys)
    judge_llm, ragas_embeddings = _build_judge(_pacer, holder)
    _chain = JudgeModelChain(judge_llm, build_steps(JUDGE_MODELS, len(keys)), holder)
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results = {}

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(
                Faithfulness(llm=judge_llm), inputs, samples, status_cb, "Faithfulness",
                prior=_prior_values(prior, "faithfulness", samples), checkpoint=_checkpointer("faithfulness", samples, results, on_progress),
            )
            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Relevancy",
                prior=_prior_values(prior, "answer_relevancy", samples),
                checkpoint=_checkpointer("answer_relevancy", samples, results, on_progress),
            )
            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(
                ContextPrecision(llm=judge_llm), inputs, samples, status_cb, "Context Precision",
                prior=_prior_values(prior, "context_precision", samples), checkpoint=_checkpointer("context_precision", samples, results, on_progress),
            )
            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(
                ContextRecall(llm=judge_llm), inputs, samples, status_cb, "Context Recall",
                prior=_prior_values(prior, "context_recall", samples), checkpoint=_checkpointer("context_recall", samples, results, on_progress),
            )
            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        # ── Exp 5: Answer Correctness (split into batches) ────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 5/6 — Answer Correctness batch 1/2...")
        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]
            all_scores = await _batched_score(
                AnswerCorrectness(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Correctness",
                prior=_prior_values(prior, "answer_correctness", samples),
                checkpoint=_checkpointer("answer_correctness", samples, results, on_progress),
            )
            df = _score_df("answer_correctness", samples, all_scores)
            results["answer_correctness"] = df
            logfire.info("🧪 Answer Correctness done", avg=round(df["answer_correctness"].mean(), 3))

        # ── Exp 6: Tool Correctness (no LLM — Jaccard) ───────────────────────
        if status_cb:
            status_cb("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 — Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results
