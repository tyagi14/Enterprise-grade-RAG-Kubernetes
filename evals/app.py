# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL: logfire must be configured before all other imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logfire
logfire.configure(token=os.getenv("LOGFIRE_TOKEN"), service_name="evals")

# ─────────────────────────────────────────────────────────────────────────────
import asyncio
import json
import nest_asyncio
import pandas as pd
import streamlit as st

nest_asyncio.apply()

from evals.answer_check import failed_indices, invalidate_scores
from evals.pipeline import run_pipeline, rerun_failed, load_golden_dataset
from evals.guardrails_eval import run_guardrails_eval, compute_guardrails_metrics
from evals.metrics import run_all_metrics
from evals.persistence import save_run, load_run, clear_run, backend_name
from evals.progress import describe_progress, is_complete

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise RAG — Eval Suite",
    page_icon="🧪",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SCORE_COLORS = {
    "green":  "#d4edda",
    "yellow": "#fff3cd",
    "red":    "#f8d7da",
}


def _badge(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    return "🔴"


def _grade(score: float) -> str:
    if score >= 0.75:
        return "✅ Good"
    elif score >= 0.5:
        return "⚠️ Fair"
    return "❌ Poor"


def _color_score(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= 0.75:
        return f"background-color: {SCORE_COLORS['green']}"
    elif val >= 0.5:
        return f"background-color: {SCORE_COLORS['yellow']}"
    return f"background-color: {SCORE_COLORS['red']}"


def _render_metric_table(df: pd.DataFrame, metric_col: str, title: str):
    avg = df[metric_col].mean()
    if pd.isna(avg):
        st.markdown(f"**{title}** — ⏳ not scored yet")
        return
    st.markdown(f"**{title}** — AVG: {_badge(avg)} `{avg:.2f}` {_grade(avg)}")
    # pandas 3 removed Styler.applymap (renamed Styler.map in 2.1); support both.
    _styler = df.style
    _color = getattr(_styler, "map", None) or _styler.applymap
    styled = _color(_color_score, subset=[metric_col]).format({metric_col: "{:.3f}"})
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _run_async(coro):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────────────────────────────────────
if "golden" not in st.session_state:
    st.session_state.golden = load_golden_dataset()
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False
if "enriched_dataset" not in st.session_state:
    st.session_state.enriched_dataset = None
if "guardrails_results" not in st.session_state:
    st.session_state.guardrails_results = None
if "metric_results" not in st.session_state:
    st.session_state.metric_results = None
if "pipeline_rows" not in st.session_state:
    st.session_state.pipeline_rows = []

# ── Restore the last saved run ────────────────────────────────────────────────
# Streamlit session memory is lost on refresh / new tab / instance restart / redeploy, which used to
# send Step 3 back to "Complete Step 2 first". Reload the saved run once per browser session.
if "restore_checked" not in st.session_state:
    st.session_state.restore_checked = True
    st.session_state.restore_notice = None
    if not st.session_state.pipeline_done:
        _run, _err = load_run()
        if _err:
            st.session_state.restore_notice = ("warning", _err)
        elif _run and _run.get("enriched_dataset") and _run.get("guardrails_results") is not None:
            st.session_state.enriched_dataset = _run["enriched_dataset"]
            st.session_state.guardrails_results = _run["guardrails_results"]
            st.session_state.metric_results = _run.get("metric_results")
            st.session_state.pipeline_done = True
            _mr = _run.get("metric_results")
            _scored = (" with metric scores" if is_complete(_mr) else
                       f" with PARTIAL metric scores ({describe_progress(_mr)}) — click Resume in Step 3") if _mr else ""
            st.session_state.restore_notice = (
                "info", f"♻️ Restored the saved run from {_run.get('saved_at', 'earlier')}{_scored}. "
                        f"Use **Reset & Re-run** in Step 2 to start a fresh one.")

golden = st.session_state.golden
if st.session_state.get("restore_notice"):
    _kind, _msg = st.session_state.restore_notice
    (st.warning if _kind == "warning" else st.info)(_msg)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.title("🧪 Enterprise RAG — Evaluation Suite")
st.caption(
    "Step 1: Review ground truth → Step 2: Run live pipeline → Step 3: Score with RAGAS"
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📋 Step 1 — Ground Truth", "🚀 Step 2 — Live Pipeline", "📊 Step 3 — Eval Metrics"]
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ground Truth
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ground Truth Dataset")
    st.markdown(
        "These are the **golden Q&A pairs** built by parsing your real enterprise documents. "
        "Each entry has a question, a reference answer (ground truth), and the expected tool the RAG agent should call."
    )

    rag_rows = []
    for s in golden["rag_samples"]:
        rag_rows.append({
            "ID": s["id"],
            "Domain": s["domain"].replace("_", " ").title(),
            "Question": s["question"],
            "Reference Answer": s["reference"][:120] + "..." if len(s["reference"]) > 120 else s["reference"],
            "Expected Tool": s["expected_tools"][0] if s["expected_tools"] else "—",
        })
    df_golden = pd.DataFrame(rag_rows)
    st.dataframe(df_golden, use_container_width=True, hide_index=True)
    st.caption(f"✅ {len(rag_rows)} golden RAG samples from 5 enterprise docs")

    st.divider()

    st.subheader("Guardrails Test Cases")
    st.markdown(
        "These inputs test whether the safety rails correctly **block adversarial inputs** "
        "and **let through legitimate questions**."
    )

    g_rows = []
    for g in golden["guardrails_samples"]:
        expected_label = "🛡️ Block" if g["expected_blocked"] else "✅ Pass"
        g_rows.append({
            "ID": g["id"],
            "Input": g["input"],
            "Expected": expected_label,
            "Type": g["type"],
            "Description": g["description"],
        })
    st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)
    st.caption("6 guardrails test cases: 3 adversarial (should block) + 3 legit (should pass)")

    with st.expander("View raw golden_dataset.json"):
        st.json(golden)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Pipeline
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Live Pipeline — Collect Real Responses")
    st.markdown(
        "Sends each golden question to your **running FastAPI app** (`localhost:8000/query`). "
        "Captures the actual response, retrieved contexts, and tool called. "
        "Full answers (up to 2,500 chars) and all 5 retrieved chunks are kept so the RAGAS judge sees the real evidence."
    )
    st.info(
        "⚠️ Make sure your FastAPI backend is running first: `uvicorn app.main:app --reload --port 8000`",
        icon="⚠️",
    )

    col_p1, col_p2, col_p3 = st.columns([1, 1, 2])
    run_pipeline_btn = col_p1.button(
        "▶️ Run Live Pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.pipeline_done,
    )
    reset_btn = col_p2.button(
        "🔄 Reset & Re-run",
        use_container_width=True,
        disabled=not st.session_state.pipeline_done,
    )

    if reset_btn:
        st.session_state.pipeline_done = False
        st.session_state.enriched_dataset = None
        st.session_state.guardrails_results = None
        st.session_state.metric_results = None
        st.session_state.pipeline_rows = []
        _err = clear_run()
        if _err:
            st.warning(_err)
        st.rerun()

    if run_pipeline_btn:
        st.session_state.pipeline_rows = []
        progress_bar = st.progress(0, text="Starting pipeline...")
        live_table_slot = st.empty()
        status_slot = st.empty()

        def pipeline_cb(i, total, question, stage, response=""):
            pct = int((i / total) * 100)
            if stage == "calling":
                progress_bar.progress(pct, text=f"[{i+1}/{total}] Calling /query: {question[:60]}...")
            else:
                short_q = question[:55] + "..." if len(question) > 55 else question
                short_r = response[:80] + "..." if len(response) > 80 else response
                st.session_state.pipeline_rows.append({
                    "#": i + 1,
                    "Question": short_q,
                    "Live Response (truncated)": short_r if short_r else "⚠️ No response",
                    "Status": "✅" if short_r else "❌",
                })
                live_table_slot.dataframe(
                    pd.DataFrame(st.session_state.pipeline_rows),
                    use_container_width=True,
                    hide_index=True,
                )
                progress_bar.progress(
                    int(((i + 1) / total) * 100),
                    text=f"[{i+1}/{total}] ✅ Done",
                )

        with logfire.span("🚀 Streamlit — Run Pipeline Button"):
            enriched = run_pipeline(golden, progress_callback=pipeline_cb)
            st.session_state.enriched_dataset = enriched

        progress_bar.progress(100, text="✅ All responses collected!")
        status_slot.success(f"💾 {len(enriched['rag_samples'])} responses stored in session.")

        # ── Guardrails tests ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Guardrails Tests")
        g_progress = st.progress(0, text="Running guardrails tests...")
        g_status_slot = st.empty()

        def g_cb(i, total, input_text):
            g_progress.progress(
                int((i / total) * 100),
                text=f"[{i+1}/{total}] Testing: {input_text[:60]}...",
            )

        with logfire.span("🛡️ Streamlit — Guardrails Tests"):
            g_results = run_guardrails_eval(enriched["guardrails_samples"], progress_callback=g_cb)
            g_metrics = compute_guardrails_metrics(g_results)
            st.session_state.guardrails_results = g_results
            st.session_state.pipeline_done = True
            _err = save_run(enriched, g_results)
            if _err:
                st.warning(f"⚠️ {_err} The run is kept in this browser session only.")

        g_progress.progress(100, text="✅ Guardrails tests complete!")

        g_rows_live = []
        for r in g_results:
            result_label = {
                "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                "FP": "🛡️ Blocked ❌ (False Positive)", "FN": "✅ Passed ❌ (Missed)",
            }.get(r["result"], r["result"])
            g_rows_live.append({
                "ID": r["id"],
                "Input": r["input"][:70],
                "Expected": "🛡️ Block" if r["expected_blocked"] else "✅ Pass",
                "Actual": "Blocked" if r["actual_blocked"] else "Passed",
                "Result": result_label,
            })
        st.dataframe(pd.DataFrame(g_rows_live), use_container_width=True, hide_index=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Correct", f"{g_metrics['correct']}/{g_metrics['total']}")
        mc2.metric("Precision", f"{g_metrics['precision']:.2f}")
        mc3.metric("Recall", f"{g_metrics['recall']:.2f}")
        mc4.metric("Accuracy", f"{g_metrics['accuracy']:.2f}")

    elif st.session_state.pipeline_done:
        st.success("✅ Pipeline already run. See results below.")

        resp_rows = []
        for s in st.session_state.enriched_dataset["rag_samples"]:
            resp_rows.append({
                "#": s["id"],
                "Domain": s["domain"].replace("_", " ").title(),
                "Question": s["question"][:60],
                "Live Response": s["actual_response"][:100] + "..." if len(s.get("actual_response","")) > 100 else s.get("actual_response",""),
                "Tool Called": s["actual_tools_called"][0] if s.get("actual_tools_called") else "—",
                "Contexts Retrieved": len(s.get("actual_contexts", [])),
            })
        st.dataframe(pd.DataFrame(resp_rows), use_container_width=True, hide_index=True)

        _failed = failed_indices(st.session_state.enriched_dataset["rag_samples"])
        if _failed:
            st.warning(
                f"⚠️ {len(_failed)} answer(s) are backend errors (#{', #'.join(str(i + 1) for i in _failed)}) and "
                "would score 0 on every metric. Re-run just those; their old scores are cleared so Step 3 re-scores them."
            )
            if st.button(f"🔁 Re-run {len(_failed)} failed answer(s)", type="primary"):
                _bar = st.progress(0, text="Re-running failed answers...")

                def _rerun_cb(k, total, question, stage, response=""):
                    _bar.progress(int((k / total) * 100), text=f"[{k + 1}/{total}] {stage}: {question[:60]}...")

                _fixed = rerun_failed(st.session_state.enriched_dataset, progress_callback=_rerun_cb)
                _redone = [_fixed["rag_samples"][i]["question"] for i in _failed]
                st.session_state.enriched_dataset = _fixed
                st.session_state.metric_results = invalidate_scores(st.session_state.metric_results, _redone)
                _err = save_run(_fixed, st.session_state.guardrails_results, st.session_state.metric_results)
                if _err:
                    st.warning(f"⚠️ {_err} The run is kept in this browser session only.")
                st.rerun()

        if st.session_state.guardrails_results:
            st.divider()
            st.subheader("Guardrails Results (from previous run)")
            g_rows_prev = []
            for r in st.session_state.guardrails_results:
                result_label = {
                    "TP": "🛡️ Blocked ✅", "TN": "✅ Passed ✅",
                    "FP": "Blocked ❌ FP", "FN": "Passed ❌ FN",
                }.get(r["result"], r["result"])
                g_rows_prev.append({
                    "ID": r["id"],
                    "Input": r["input"][:70],
                    "Result": result_label,
                })
            st.dataframe(pd.DataFrame(g_rows_prev), use_container_width=True, hide_index=True)
            gm = compute_guardrails_metrics(st.session_state.guardrails_results)
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Correct", f"{gm['correct']}/{gm['total']}")
            mc2.metric("Precision", f"{gm['precision']:.2f}")
            mc3.metric("Recall", f"{gm['recall']:.2f}")
            mc4.metric("Accuracy", f"{gm['accuracy']:.2f}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Eval Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Eval Metrics — RAGAS + Tool Correctness")

    if not st.session_state.pipeline_done:
        st.warning("⚠️ Complete Step 2 (Live Pipeline) first to collect responses.")
    else:
        st.markdown(
            "Runs all **6 metric experiments** on the stored responses. "
            "LLM-based metrics use `JUDGE_GROQ` key — samples are scored one at a time "
            "paced by Groq's own remaining-token counter (not fixed sleeps) to stay within the "
            "**8,000 tokens/min** limit of the judge model. Runtime depends on that limit: "
            "roughly 60-90 min for 15 samples on the free Groq tier (much faster on a paid tier)."
        )
        st.info(
            "Token key used: `JUDGE_GROQ`. Groq limits tokens **per model across the whole organisation**, "
            "so the judge walks a chain of models (`qwen3.8-27b`, then `gpt-oss-20b`) and, when extra keys from "
            "different Groq organisations are configured (`JUDGE_GROQ_2`, `JUDGE_GROQ_3`), tries every key on a model before moving on. "
            "It switches automatically if a daily limit runs out mid-run. The judge now reads whole chunks and full answers, so a sample costs ~30-35K tokens: a full 15-sample run is ~500K tokens, over one free Groq model's 200K/day. "
            "The pacer keeps part of the quota free for other callers.",
            icon="ℹ️",
        )

        _prior = st.session_state.metric_results
        _partial = bool(_prior) and not is_complete(_prior)
        if _partial:
            st.warning(
                f"⏸️ Partial results from an earlier run: {describe_progress(_prior)} "
                "Scored samples are kept — **Resume** only scores what is missing."
            )
        run_metrics_btn = st.button(
            "▶️ Resume Eval Metrics" if _partial else "▶️ Run Eval Metrics",
            type="primary",
            disabled=not st.session_state.pipeline_done,
        )

        if run_metrics_btn:
            status_slot = st.empty()
            results_slots = {}

            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                results_slots[key] = st.empty()

            def status_cb(msg: str):
                status_slot.info(msg)

            save_errors = []

            def on_progress(partial_results: dict):
                # Checkpoint after EVERY scored sample: a refresh/reconnect mid-run (the script is killed)
                # used to lose everything; now the next click resumes from here.
                st.session_state.metric_results = partial_results
                err = save_run(st.session_state.enriched_dataset, st.session_state.guardrails_results, partial_results)
                if err and not save_errors:
                    save_errors.append(err)

            with logfire.span("📊 Streamlit — Run Metrics Button"):
                metric_results = _run_async(
                    run_all_metrics(
                        st.session_state.enriched_dataset, status_cb=status_cb,
                        prior=_prior, on_progress=on_progress,
                    )
                )
                st.session_state.metric_results = metric_results
                _err = save_run(st.session_state.enriched_dataset, st.session_state.guardrails_results, metric_results)
                if _err or save_errors:
                    st.warning(f"⚠️ {_err or save_errors[0]} Scores are kept in this browser session only.")

            if is_complete(metric_results):
                status_slot.success("✅ All 6 experiments complete!")
            else:
                status_slot.warning(
                    f"⚠️ Finished with gaps — {describe_progress(metric_results)} "
                    "Usually a Groq daily limit; click **Resume** later to score only the missing samples."
                )

            for key, title in metric_display_names.items():
                if key in metric_results:
                    with results_slots[key].container():
                        _render_metric_table(metric_results[key], key, title)

        elif st.session_state.metric_results:
            if is_complete(st.session_state.metric_results):
                st.success("✅ Metrics already computed. Showing results below.")
            else:
                st.info("Showing the scores collected so far (blank rows = not scored yet).")
            metric_display_names = {
                "faithfulness":      "Exp 1 — Faithfulness",
                "answer_relevancy":  "Exp 2 — Answer Relevancy",
                "context_precision": "Exp 3 — Context Precision",
                "context_recall":    "Exp 4 — Context Recall",
                "answer_correctness":"Exp 5 — Answer Correctness",
                "tool_correctness":  "Exp 6 — Tool Correctness",
            }
            for key, title in metric_display_names.items():
                if key in st.session_state.metric_results:
                    _render_metric_table(st.session_state.metric_results[key], key, title)

        # ── Final Summary ─────────────────────────────────────────────────────
        if st.session_state.metric_results:
            st.divider()
            st.subheader("Final Summary")

            mr = st.session_state.metric_results
            summary = [
                ("Faithfulness",       mr.get("faithfulness",      pd.DataFrame()).get("faithfulness",      pd.Series()).mean()),
                ("Answer Relevancy",   mr.get("answer_relevancy",  pd.DataFrame()).get("answer_relevancy",  pd.Series()).mean()),
                ("Context Precision",  mr.get("context_precision", pd.DataFrame()).get("context_precision", pd.Series()).mean()),
                ("Context Recall",     mr.get("context_recall",    pd.DataFrame()).get("context_recall",    pd.Series()).mean()),
                ("Answer Correctness", mr.get("answer_correctness",pd.DataFrame()).get("answer_correctness",pd.Series()).mean()),
                ("Tool Correctness",   mr.get("tool_correctness",  pd.DataFrame()).get("tool_correctness",  pd.Series()).mean()),
            ]

            cols = st.columns(len(summary))
            for col, (name, score) in zip(cols, summary):
                if pd.notna(score):
                    col.metric(
                        label=name,
                        value=f"{score:.2f}",
                        delta=_grade(score),
                    )

            if st.session_state.guardrails_results:
                gm = compute_guardrails_metrics(st.session_state.guardrails_results)
                st.metric(
                    label="🛡️ Guardrails Accuracy",
                    value=f"{gm['correct']}/{gm['total']}",
                    delta=f"Precision {gm['precision']:.2f} | Recall {gm['recall']:.2f}",
                )

            summary_df = pd.DataFrame([
                {"Metric": name, "Score": f"{score:.3f}" if pd.notna(score) else "—", "Grade": _grade(score) if pd.notna(score) else "—"}
                for name, score in summary
            ])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
