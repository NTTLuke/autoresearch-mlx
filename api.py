#!/usr/bin/env python3
"""
Autoresearch Dashboard — FastAPI backend.

Serves the React frontend, experiment data, and a benchmark engine that
evaluates multiple LM Studio models as autonomous ML researchers.

Usage:
    uv run python api.py
    uv run python api.py --port 8000 --base-url http://localhost:1234/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from experiments_db import (
    get_best_by_model,
    get_experiment_by_id,
    get_experiments,
    get_experiments_with_weights,
    init_db,
)
from run_autoresearch import (
    DB_FILE,
    create_researcher,
    run_one_experiment,
    sample_music_cli,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autoresearch Dashboard API")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--base-url", default="http://localhost:1234/v1",
                    help="LM Studio API base URL")
    return p.parse_args()


ARGS = _parse_args()

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Autoresearch Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db(DB_FILE)

# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------

@app.get("/api/experiments")
def api_experiments(model_id: str | None = Query(None)):
    rows = get_experiments(model_id=model_id, db_path=DB_FILE)
    return rows


@app.get("/api/models")
def api_models():
    rows = get_best_by_model(db_path=DB_FILE)
    return rows


@app.get("/api/experiments/with-weights")
def api_experiments_with_weights():
    rows = get_experiments_with_weights(db_path=DB_FILE)
    return rows


# ---------------------------------------------------------------------------
# Benchmark engine state  (thread-safe queue, no asyncio dependency)
# ---------------------------------------------------------------------------

class BenchmarkState:
    def __init__(self) -> None:
        self.running = False
        self.stop_requested = False
        self.current_model: str = ""
        self.current_experiment: int = 0
        self.experiments_per_model: int = 0
        self.progress: list[dict[str, Any]] = []
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def status_dict(self) -> dict:
        return {
            "running": self.running,
            "current_model": self.current_model,
            "current_experiment": self.current_experiment,
            "experiments_per_model": self.experiments_per_model,
            "progress": self.progress,
        }

    def emit(self, event: str, data: dict | str) -> None:
        payload = json.dumps({"event": event, "data": data})
        self._log_queue.put(payload)


benchmark = BenchmarkState()


# ---------------------------------------------------------------------------
# Benchmark runner (runs in a background thread)
# ---------------------------------------------------------------------------

def _benchmark_worker(models: list[str], experiments_per_model: int, base_url: str) -> None:
    benchmark.running = True
    benchmark.stop_requested = False
    benchmark.experiments_per_model = experiments_per_model
    benchmark.progress = []

    try:
        for model_id in models:
            if benchmark.stop_requested:
                break

            benchmark.current_model = model_id
            benchmark.emit("model_start", {"model": model_id})

            agent = None if model_id == "random-search" else create_researcher(model_id, base_url)
            prev_result = ""

            model_progress = {
                "model": model_id,
                "completed": 0,
                "total": experiments_per_model,
                "results": [],
            }
            benchmark.progress.append(model_progress)

            for i in range(1, experiments_per_model + 1):
                if benchmark.stop_requested:
                    break

                benchmark.current_experiment = i
                benchmark.emit("experiment_start", {
                    "model": model_id,
                    "experiment": i,
                    "total": experiments_per_model,
                })

                def _progress_cb(msg: str) -> None:
                    benchmark.emit("training_progress", msg)

                def _stop_check() -> bool:
                    return benchmark.stop_requested

                try:
                    result_text = run_one_experiment(
                        agent=agent,
                        model_id=model_id,
                        iteration=i,
                        prev_result=prev_result,
                        goal="minimize val_bpb",
                        base_url=base_url,
                        on_progress=_progress_cb,
                        stop_check=_stop_check,
                    )
                    prev_result = result_text

                    benchmark.emit("experiment_done", {
                        "model": model_id,
                        "experiment": i,
                        "summary": result_text[:500],
                    })
                    model_progress["completed"] = i
                    model_progress["results"].append({
                        "experiment": i,
                        "status": "ok",
                        "summary": result_text[:300],
                    })

                except Exception as exc:
                    benchmark.emit("experiment_error", {
                        "model": model_id,
                        "experiment": i,
                        "error": str(exc),
                    })
                    model_progress["completed"] = i
                    model_progress["results"].append({
                        "experiment": i,
                        "status": "error",
                        "error": str(exc),
                    })
                    prev_result = f"CRASHED: {exc}"

            benchmark.emit("model_done", {"model": model_id})

        benchmark.emit("benchmark_done", {"stopped": benchmark.stop_requested})
    except Exception:
        benchmark.emit("fatal_error", {"error": traceback.format_exc()})
    finally:
        benchmark.running = False
        benchmark.current_model = ""
        benchmark.current_experiment = 0


# ---------------------------------------------------------------------------
# Benchmark endpoints
# ---------------------------------------------------------------------------

class BenchmarkStartRequest(BaseModel):
    models: list[str]
    experiments_per_model: int = 5


@app.post("/api/benchmark/start")
def api_benchmark_start(req: BenchmarkStartRequest):
    if benchmark.running:
        return {"error": "Benchmark already running"}

    benchmark._log_queue = queue.Queue()

    t = threading.Thread(
        target=_benchmark_worker,
        args=(req.models, req.experiments_per_model, ARGS.base_url),
        daemon=True,
    )
    benchmark._thread = t
    t.start()

    return {"status": "started", "models": req.models, "experiments_per_model": req.experiments_per_model}


@app.get("/api/benchmark/status")
def api_benchmark_status():
    return benchmark.status_dict()


@app.post("/api/benchmark/stop")
def api_benchmark_stop():
    if not benchmark.running:
        return {"status": "not_running"}
    benchmark.stop_requested = True
    return {"status": "stop_requested"}


@app.get("/api/benchmark/stream")
async def api_benchmark_stream(request: Request):
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            try:
                msg = benchmark._log_queue.get_nowait()
                yield {"data": msg}
            except queue.Empty:
                await asyncio.sleep(0.5)
                yield {"data": json.dumps({"event": "heartbeat", "data": {}})}

            if not benchmark.running:
                while not benchmark._log_queue.empty():
                    try:
                        yield {"data": benchmark._log_queue.get_nowait()}
                    except queue.Empty:
                        break
                yield {"data": json.dumps({"event": "stream_end", "data": {}})}
                break

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Sampler endpoint
# ---------------------------------------------------------------------------

class SampleRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = 200
    temperature: float = 0.8
    experiment_id: int | None = None


@app.post("/api/sample")
def api_sample(req: SampleRequest):
    weights_path = ""
    if req.experiment_id is not None:
        exp = get_experiment_by_id(req.experiment_id, db_path=DB_FILE)
        if exp and exp.get("weights_path"):
            weights_path = exp["weights_path"]
        else:
            return {"output": f"ERROR: No weights found for experiment #{req.experiment_id}"}
    output = sample_music_cli(req.prompt, req.max_tokens, req.temperature, weights_path)
    return {"output": output}


# ---------------------------------------------------------------------------
# Serve React frontend (production build)
# ---------------------------------------------------------------------------

FRONTEND_BUILD = Path(__file__).resolve().parent / "frontend" / "dist"

if FRONTEND_BUILD.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_BUILD / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        index = FRONTEND_BUILD / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Frontend not built. Run: cd frontend && npm run build</h1>")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print(f"  Autoresearch Dashboard")
    print(f"  API:         http://localhost:{ARGS.port}")
    print(f"  LM Studio:   {ARGS.base_url}")
    print(f"  DB:          {DB_FILE}")
    if FRONTEND_BUILD.exists():
        print(f"  Frontend:    {FRONTEND_BUILD}")
    else:
        print(f"  Frontend:    not built (run: cd frontend && npm run build)")
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port)
