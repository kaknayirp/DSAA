import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Literal, Protocol

from fastapi import FastAPI
from pydantic import BaseModel

from .integration import configure_project
from .original import load_original
from .studio import BenchmarkEvaluator, Studio
from .studio_schemas import CacheEntry, CacheMetrics, CompareRequest


class Comparison(BaseModel):
    baseline: CacheMetrics
    dsaa: CacheMetrics
    entries: list[CacheEntry]


class ComparisonResult(Comparison):
    source: Literal["demo", "measured"]
    original_run_id: str | None = None
    dsaa_run_id: str | None = None


class ComparisonRunner(Protocol):
    def __call__(self, prompt: str) -> Comparison: ...


def create_app(
    runner: ComparisonRunner | None = None, *, runners=None, model=None, device=None,
    evaluator: BenchmarkEvaluator | None = None,
    configure: Callable[[Studio], None] | None = None,
) -> FastAPI:
    studio = Studio(runners, model, device, evaluator)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            if configure is not None:
                configure(studio)
            model_path = os.getenv("DSAA_MODEL")
            if model_path and "original" not in studio.runners:
                studio.device = os.getenv("DSAA_DEVICE", "cpu")
                studio.runners["original"] = load_original(model_path, studio.device)
                studio.model = model_path
            yield
        finally:
            studio.executor.shutdown(wait=True)

    app = FastAPI(title="DSAA Studio", lifespan=lifespan)
    app.include_router(studio.router())

    @app.get("/api/health")
    def health():
        return studio.status(legacy=runner is not None)

    @app.get("/api/demo", response_model=ComparisonResult)
    def demo():
        tokens = ["Hello", "!", " My", " name", " is", " Rahul", ".", " I", " study", " AI", "."]
        retained = {2, 3, 4, 5, 8, 9, 10}
        return ComparisonResult(
            source="demo",
            baseline=CacheMetrics(kv_cache_bytes=11 * 32_768, cached_tokens=11),
            dsaa=CacheMetrics(kv_cache_bytes=len(retained) * 32_768, cached_tokens=len(retained)),
            entries=[
                CacheEntry(
                    position=i, token=token, retained=i in retained,
                    attention=0.8 if i in retained else 0.2,
                    semantic=0.9 if i in retained else 0.1,
                    recency=round(i / (len(tokens) - 1), 3),
                    score=round((0.67 if i in retained else 0.13) + 0.2 * i / (len(tokens) - 1), 3),
                    reason="Illustrative retention decision.", saved_bytes=0 if i in retained else 32_768,
                )
                for i, token in enumerate(tokens)
            ],
        )

    @app.post("/api/compare", response_model=ComparisonResult)
    def compare(request: CompareRequest):
        if runner is not None:
            return ComparisonResult(**runner(request.prompt).model_dump(), source="measured")
        original, dsaa = studio.compare(request)
        return ComparisonResult(
            baseline=original["metrics"], dsaa=dsaa["metrics"], entries=dsaa["entries"], source="measured",
            original_run_id=original["id"], dsaa_run_id=dsaa["id"],
        )

    return app


app = create_app(configure=configure_project)
