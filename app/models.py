"""
Pydantic models — the contract for the HTTP API and the agent's domain types.

Lesson: keep API models (what the HTTP client sees) separate from domain
models (what your code passes around). They evolve independently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ===========================================================================
# API models
# ===========================================================================


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    max_iterations: int | None = Field(None, ge=1, le=20)
    max_cost_usd: float | None = Field(None, gt=0, le=10)


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    output_preview: str
    duration_ms: int
    error: str | None = None


class CostBreakdown(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class ResearchResponse(BaseModel):
    run_id: str
    question: str
    final_answer: str
    iterations: int
    tool_calls: list[ToolCallRecord]
    cost: CostBreakdown
    duration_ms: int
    stopped_reason: Literal[
        "completed", "max_iterations", "max_cost", "max_wall_time", "error"
    ]
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ===========================================================================
# Tool input models — Pydantic validates LLM-supplied arguments
# ===========================================================================


class WebSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(5, ge=1, le=10)


class FetchUrlArgs(BaseModel):
    url: str = Field(..., min_length=8, max_length=2000)
    max_chars: int = Field(4000, ge=200, le=20000)


class CalculatorArgs(BaseModel):
    expression: str = Field(..., min_length=1, max_length=500)
