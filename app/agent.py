"""
Async ReAct agent with a circuit breaker.

The loop terminates when ANY of the following becomes true:
    * the model returns a final message (no tool calls)         -> "completed"
    * we hit max_iterations                                     -> "max_iterations"
    * cumulative cost.usd exceeds max_cost_usd                  -> "max_cost"
    * wall-clock exceeds max_wall_time_s                        -> "max_wall_time"
    * an unrecoverable exception                                 -> "error"

This shape is the production version of the toy loop in the learning project:
same skeleton, but every external call has a timeout, retries are explicit,
and the loop has hard ceilings so a misbehaving prompt can never cost
arbitrary money or run forever.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from structlog.contextvars import bind_contextvars, unbind_contextvars

from .config import Settings
from .llm import LLMClient
from .logging_setup import get_logger
from .models import CostBreakdown, ToolCallRecord
from .tools import dispatch_tool, openai_tool_schemas

log = get_logger(__name__)


SYSTEM_PROMPT = """You are a careful research assistant.

You have three tools: web_search, fetch_url, calculator.

How to work:
  1. Think about what the question really asks. If broad, decompose.
  2. Use web_search to discover candidate sources, then fetch_url on the
     most promising ones to read them in detail.
  3. Cross-check important factual claims across at least TWO independent
     sources.
  4. Use the calculator for any arithmetic - never compute it yourself.
  5. When you have enough evidence, stop calling tools and write the final
     answer. The final answer must:
       - directly answer the user's question
       - cite sources inline like [1], [2] referring to URLs you actually fetched
       - end with a "Sources" list of those URLs

Be concise. Don't fetch the same URL twice. Refine queries that return junk."""


class ResearchAgent:
    def __init__(self, settings: Settings, llm: LLMClient) -> None:
        self._settings = settings
        self._llm = llm

    async def run(
        self,
        question: str,
        *,
        max_iterations: int | None = None,
        max_cost_usd: float | None = None,
    ) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        bind_contextvars(run_id=run_id)
        try:
            return await self._run_inner(
                run_id=run_id,
                question=question,
                max_iterations=max_iterations or self._settings.max_iterations,
                max_cost_usd=max_cost_usd or self._settings.max_cost_usd,
            )
        finally:
            unbind_contextvars("run_id")

    async def _run_inner(
        self,
        *,
        run_id: str,
        question: str,
        max_iterations: int,
        max_cost_usd: float,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        deadline = started + self._settings.max_wall_time_s

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        cost = CostBreakdown()
        tool_records: list[ToolCallRecord] = []
        final_answer = ""
        stopped_reason = "completed"
        error: str | None = None
        iterations = 0

        try:
            for i in range(1, max_iterations + 1):
                iterations = i

                # ---- circuit breaker checks (BEFORE the expensive call) ----
                if time.perf_counter() > deadline:
                    stopped_reason = "max_wall_time"
                    final_answer = (
                        "[stopped: wall-time budget exceeded] partial findings logged."
                    )
                    break
                if cost.usd >= max_cost_usd:
                    stopped_reason = "max_cost"
                    final_answer = (
                        "[stopped: cost budget exceeded] partial findings logged."
                    )
                    break

                log.info("iteration_start", iter=i)

                resp = await self._llm.chat(
                    messages=messages, tools=openai_tool_schemas(), cost=cost
                )
                msg = resp.choices[0].message

                # Echo assistant message back into history (must include
                # original tool_calls verbatim — the API requires it).
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (msg.tool_calls or [])
                        ]
                        or None,
                    }
                )

                if not msg.tool_calls:
                    final_answer = msg.content or ""
                    stopped_reason = "completed"
                    break

                for tc in msg.tool_calls:
                    try:
                        raw_args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        raw_args = {}

                    output, duration_ms, err = await dispatch_tool(
                        tc.function.name,
                        raw_args,
                        timeout_s=self._settings.tool_timeout_s,
                    )

                    tool_records.append(
                        ToolCallRecord(
                            name=tc.function.name,
                            args=raw_args,
                            output_preview=output[:200],
                            duration_ms=duration_ms,
                            error=err,
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.function.name,
                            "content": output,
                        }
                    )
            else:
                stopped_reason = "max_iterations"
                final_answer = (
                    "[stopped: max iterations reached] partial findings logged."
                )

        except Exception as e:  # noqa: BLE001 — last-resort safety net
            log.exception("agent_run_failed")
            stopped_reason = "error"
            error = str(e)
            if not final_answer:
                final_answer = "[agent error] see logs for details."

        duration_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "run_done",
            stopped_reason=stopped_reason,
            iterations=iterations,
            duration_ms=duration_ms,
            cost_usd=cost.usd,
        )

        return {
            "run_id": run_id,
            "question": question,
            "final_answer": final_answer,
            "iterations": iterations,
            "tool_calls": [tc.model_dump() for tc in tool_records],
            "cost": cost.model_dump(),
            "duration_ms": duration_ms,
            "stopped_reason": stopped_reason,
            "error": error,
        }
