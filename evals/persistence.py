"""
Persist the eval run so the Streamlit page survives refreshes, new tabs, idle shutdowns and redeploys.

Streamlit's st.session_state lives in one server process's memory, so Step 3 used to fall back to
"Complete Step 2 first" whenever the browser reconnected or Cloud Run restarted the instance. The
run (pipeline answers, guardrail results, metric scores) is now also written to one JSON object:

  - on Cloud Run (or EVAL_STATE_BACKEND=gcs): gs://<GCP_PROCESSED_BUCKET>/eval_state/last_run.json
    (the PROCESSED bucket — the RAW bucket triggers ingestion, so it must never be written here)
  - locally: <repo>/.eval_state/last_run.json

It is a single shared "last run" slot, not per-user. Every function swallows storage errors and
returns a status instead, because a storage hiccup must never take the evaluation page down.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

STATE_BLOB = "eval_state/last_run.json"
LOCAL_PATH = Path(__file__).resolve().parent.parent / ".eval_state" / "last_run.json"


def _use_gcs() -> bool:
    backend = os.getenv("EVAL_STATE_BACKEND", "").lower()
    if backend in ("gcs", "local"):
        return backend == "gcs" and bool(os.getenv("GCP_PROCESSED_BUCKET"))
    return bool(os.getenv("K_SERVICE")) and bool(os.getenv("GCP_PROCESSED_BUCKET"))   # K_SERVICE = Cloud Run


def _blob():
    from google.cloud import storage   # imported lazily: not needed for local runs
    client = storage.Client(project=os.getenv("PROJECT_ID"))
    return client.bucket(os.environ["GCP_PROCESSED_BUCKET"]).blob(STATE_BLOB)


def backend_name() -> str:
    return f"gs://{os.getenv('GCP_PROCESSED_BUCKET')}/{STATE_BLOB}" if _use_gcs() else str(LOCAL_PATH)


def _metrics_to_records(metrics: Optional[dict]) -> Optional[dict]:
    # to_json turns NaN into null, which is valid JSON (a bare NaN is not).
    return None if metrics is None else {k: json.loads(df.to_json(orient="records")) for k, df in metrics.items()}


def _metric_frame(rows: list) -> pd.DataFrame:
    """JSON stores unscored samples as null; a column that is ALL null loads as object dtype, which
    crashes the page's "{:.3f}" formatter. Score columns are therefore forced back to float (NaN)."""
    df = pd.DataFrame(rows)
    return df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in df.columns if c != "question"})


def _records_to_metrics(records: Optional[dict]) -> Optional[dict]:
    return None if records is None else {k: _metric_frame(rows) for k, rows in records.items()}


def save_run(enriched_dataset, guardrails_results, metric_results=None) -> Optional[str]:
    """Store the current run. Returns None on success, else an error message."""
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "enriched_dataset": enriched_dataset,
        "guardrails_results": guardrails_results,
        "metric_results": _metrics_to_records(metric_results),
    }
    try:
        text = json.dumps(payload, default=str)
        if _use_gcs():
            _blob().upload_from_string(text, content_type="application/json")
        else:
            LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOCAL_PATH.write_text(text)
        return None
    except Exception as e:
        return f"Could not save the run to {backend_name()}: {e}"


def load_run() -> tuple[Optional[dict], Optional[str]]:
    """Returns (run, error). run is None when nothing is saved; error is set only on a real failure."""
    try:
        if _use_gcs():
            blob = _blob()
            if not blob.exists():
                return None, None
            text = blob.download_as_text()
        else:
            if not LOCAL_PATH.exists():
                return None, None
            text = LOCAL_PATH.read_text()
        run = json.loads(text)
        run["metric_results"] = _records_to_metrics(run.get("metric_results"))
        return run, None
    except Exception as e:
        return None, f"Could not load the saved run from {backend_name()}: {e}"


def clear_run() -> Optional[str]:
    try:
        if _use_gcs():
            blob = _blob()
            if blob.exists():
                blob.delete()
        elif LOCAL_PATH.exists():
            LOCAL_PATH.unlink()
        return None
    except Exception as e:
        return f"Could not clear the saved run: {e}"
