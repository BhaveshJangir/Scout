"""
FastAPI surface.

Endpoints:
    POST /research      run an agent on a question
    GET  /runs/{id}     fetch a persisted run (full prompts + tool calls)
    GET  /healthz       liveness/readiness probe

Cross-cutting concerns:
    * request_id middleware — every log line in a request shares an id.
    * lifespan — opens/closes the DB connection and the OpenAI client.
    * Pydantic models validate the request and shape the response.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from structlog.contextvars import bind_contextvars, unbind_contextvars

from .agent import ResearchAgent
from .config import Settings, get_settings
from .llm import LLMClient
from .logging_setup import configure_logging, get_logger
from .models import ResearchRequest, ResearchResponse
from .storage import RunStore

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    llm = LLMClient(settings)
    store = RunStore(settings.db_path)
    await store.connect()

    app.state.settings = settings
    app.state.llm = llm
    app.state.store = store
    app.state.agent = ResearchAgent(settings, llm)

    log.info("startup_complete", model=settings.openai_model)
    try:
        yield
    finally:
        await llm.aclose()
        await store.aclose()
        log.info("shutdown_complete")


app = FastAPI(
    title="Research Agent API",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- middleware: request_id tagging ---------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        unbind_contextvars("request_id")
    response.headers["x-request-id"] = request_id
    return response


# ---- routes ---------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest, request: Request) -> ResearchResponse:
    agent: ResearchAgent = request.app.state.agent
    store: RunStore = request.app.state.store

    log.info("research_request", question_len=len(req.question))

    payload = await agent.run(
        req.question,
        max_iterations=req.max_iterations,
        max_cost_usd=req.max_cost_usd,
    )

    # Persist before responding so /runs/{id} works even on a flaky network.
    try:
        await store.save(payload)
    except Exception:  # noqa: BLE001 — never lose the answer over a DB blip
        log.exception("runstore_save_failed")

    return ResearchResponse(**payload)


@app.get("/runs/{run_id}", response_model=ResearchResponse)
async def get_run(run_id: str, request: Request) -> ResearchResponse:
    store: RunStore = request.app.state.store
    payload = await store.get(run_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found")
    return ResearchResponse(**payload)


# ---- generic exception handler -------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "error": str(exc)},
    )


def get_settings_for_uvicorn() -> Settings:
    """Used by main.py to read host/port without importing app state."""
    return get_settings()
