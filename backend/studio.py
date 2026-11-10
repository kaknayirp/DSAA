import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Callable, Iterable, Protocol
from uuid import uuid4

import psutil
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from .studio_schemas import (
    STRATEGIES, BenchmarkRequest, CompareRequest, InferenceRequest,
    InferenceResult, RunSettings, Strategy,
)


Runner = Callable[[InferenceRequest, RunSettings], InferenceResult]
logger = logging.getLogger(__name__)


class BenchmarkEvaluator(Protocol):
    def recall_cases(self, case_count: int, context_blocks: int) -> Iterable[tuple[str, str]]: ...
    def exact_match(self, answer: str | None, expected: str) -> bool: ...
    def summarize(self, rows: list[dict]) -> dict: ...


def now():
    return datetime.now(timezone.utc).isoformat()


class Studio:
    def __init__(self, runners=None, model=None, device=None, evaluator: BenchmarkEvaluator | None = None):
        self.runners: dict[Strategy, Runner] = dict(runners or {})
        self.model = model
        self.device = device
        self.evaluator = evaluator
        self.settings = RunSettings()
        self.compute = Lock()
        self.state = Lock()
        self.runs = {}
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsaa-benchmark")

    def status(self, legacy=False):
        return {
            "live_available": legacy or bool(self.runners),
            "strategies": {strategy: strategy in self.runners for strategy in STRATEGIES},
            "comparison_available": legacy or {"original", "dsaa"} <= self.runners.keys(),
            "benchmark_available": self.evaluator is not None,
            "model": self.model, "device": self.device, "busy": self.compute.locked(),
        }

    def require(self, strategies):
        unavailable = [strategy for strategy in strategies if strategy not in self.runners]
        if unavailable:
            raise HTTPException(503, f"Strategies unavailable: {', '.join(unavailable)}.")

    def run(self, request, settings):
        return self.runners[request.strategy](request, settings)

    def record(self, request, result, settings):
        run_id = uuid4().hex
        record = {
            "id": run_id, "created_at": now(), "strategy": request.strategy, "source": "measured",
            "model": self.model, "device": self.device, "settings": settings.model_dump(),
            **result.model_dump(),
        }
        with self.state:
            if len(self.runs) >= 30:
                self.runs.pop(next(iter(self.runs)))
            self.runs[run_id] = record
        return record

    def benchmark(self, job_id, request, settings, evaluator: BenchmarkEvaluator):
        def update(**fields):
            with self.state:
                self.jobs[job_id] = {**self.jobs[job_id], **fields}

        try:
            with self.compute:
                update(status="running")
                cases = list(evaluator.recall_cases(request.case_count, request.context_blocks))
                if len(cases) != request.case_count:
                    raise ValueError("The evaluator returned a different number of cases than requested.")
                rows = []
                results = {}
                for case, (prompt, expected) in enumerate(cases, 1):
                    for strategy in request.strategies:
                        result = self.run(InferenceRequest(
                            prompt=prompt, strategy=strategy, max_new_tokens=request.max_new_tokens,
                        ), settings)
                        metrics = result.metrics.model_dump()
                        metrics["answer_accuracy"] = float(evaluator.exact_match(metrics["answer"], expected))
                        rows.append({
                            **metrics, "case": case, "strategy": strategy, "prompt": prompt, "expected": expected,
                            "context_tokens": result.context_tokens,
                        })
                        update(completed=len(rows))
                for row in rows:
                    original = next((r for r in rows if r["case"] == row["case"] and r["strategy"] == "original"), None)
                    row["retention_ratio"] = (
                        row["cached_tokens"] / original["cached_tokens"]
                        if original and original["cached_tokens"] else None
                    )
                for strategy in request.strategies:
                    results[strategy] = evaluator.summarize([row for row in rows if row["strategy"] == strategy])
                update(status="completed", results=results, cases=rows, finished_at=now())
        except Exception as error:
            logger.exception("Benchmark %s failed", job_id)
            update(status="failed", error=str(error) if isinstance(error, ValueError)
                   else "Evaluation failed. See server logs.", finished_at=now())

    def router(self):
        router = APIRouter()
        process = psutil.Process()

        @router.get("/api/system/metrics")
        def metrics():
            ram = psutil.virtual_memory()
            gpu_bytes = None
            if self.device and self.device.split(":")[0] == "cuda":
                import torch

                if torch.cuda.is_available():
                    gpu_bytes = torch.cuda.memory_allocated(self.device)
            return {
                "timestamp": now(), "ram_total_bytes": ram.total, "ram_used_bytes": ram.total - ram.available,
                "memory_pressure": ram.percent / 100, "process_ram_bytes": process.memory_info().rss,
                "gpu_memory_bytes": gpu_bytes, "busy": self.compute.locked(),
            }

        @router.websocket("/ws/metrics")
        async def metric_stream(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    await websocket.send_json(metrics())
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=2)
                        if message["type"] == "websocket.disconnect":
                            return
                    except TimeoutError:
                        pass
            except (WebSocketDisconnect, OSError):
                return

        @router.post("/api/inference")
        def inference(request: InferenceRequest):
            self.require([request.strategy])
            if not self.compute.acquire(blocking=False):
                raise HTTPException(409, "An experiment is running. Try again when it finishes.")
            try:
                settings = self.settings
                return self.record(request, self.run(request, settings), settings)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            finally:
                self.compute.release()

        @router.get("/api/cache/{run_id}")
        def cache(run_id: str):
            with self.state:
                run = self.runs.get(run_id)
            if run is None:
                raise HTTPException(404, "Run not found.")
            return {key: value for key, value in run.items() if key != "entries"}

        @router.get("/api/cache/{run_id}/tokens")
        def tokens(run_id: str):
            with self.state:
                run = self.runs.get(run_id)
            if run is None:
                raise HTTPException(404, "Run not found.")
            return run["entries"]

        @router.get("/api/config")
        def config():
            return self.settings

        @router.put("/api/config")
        def set_config(settings: RunSettings):
            self.settings = settings
            return settings

        @router.post("/api/benchmarks", status_code=202)
        def start_benchmark(request: BenchmarkRequest):
            evaluator = self.evaluator
            if evaluator is None:
                raise HTTPException(503, "The benchmark evaluator is not connected.")
            self.require(request.strategies)
            job_id = uuid4().hex
            settings = self.settings
            job = {
                "id": job_id, "status": "queued", "created_at": now(), "completed": 0,
                "total": len(request.strategies) * request.case_count,
                "request": request.model_dump(), "settings": settings.model_dump(),
                "model": self.model, "device": self.device,
            }
            with self.state:
                if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= 3:
                    raise HTTPException(429, "The benchmark queue is full.")
                if len(self.jobs) >= 30:
                    oldest_finished = next(k for k, j in self.jobs.items() if j["status"] in {"completed", "failed"})
                    del self.jobs[oldest_finished]
                self.jobs[job_id] = job
            self.executor.submit(self.benchmark, job_id, request, settings, evaluator)
            return job

        @router.get("/api/benchmarks")
        def benchmarks():
            with self.state:
                return [{k: v for k, v in job.items() if k != "cases"} for job in reversed(self.jobs.values())]

        @router.get("/api/benchmarks/{job_id}")
        def benchmark_status(job_id: str):
            with self.state:
                job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Experiment not found.")
            return {k: v for k, v in job.items() if k != "cases"}

        @router.get("/api/benchmarks/{job_id}/report")
        def report(job_id: str):
            with self.state:
                job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "Experiment not found.")
            if job["status"] != "completed":
                raise HTTPException(409, "The experiment has no completed report.")
            return JSONResponse(job, headers={"Content-Disposition": f'attachment; filename="dsaa-{job_id}.json"'})

        return router

    def compare(self, request: CompareRequest):
        self.require(["original", "dsaa"])
        if not self.compute.acquire(blocking=False):
            raise HTTPException(409, "An experiment is running. Try again when it finishes.")
        try:
            settings = self.settings
            runs = [
                InferenceRequest(**request.model_dump(), strategy=strategy)
                for strategy in ("original", "dsaa")
            ]
            results = [self.run(run, settings) for run in runs]
            return [self.record(run, result, settings) for run, result in zip(runs, results)]
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            self.compute.release()
