"""
Async tools.

Each tool:
    * has a Pydantic args model (validates LLM-supplied input -> safe types)
    * is async (so we can run blocking I/O in a thread)
    * is wrapped with a hard per-call timeout
    * returns a string (model consumes strings; we serialize as needed)

Centralizing this in one Tool class means the agent loop stays short and
every tool gets the same observability + timeout treatment.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel, ValidationError

from .logging_setup import get_logger
from .models import CalculatorArgs, FetchUrlArgs, WebSearchArgs

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _web_search(args: WebSearchArgs) -> str:
    # DDGS is sync; offload so we don't block the event loop.
    def _run() -> list[dict[str, Any]]:
        with DDGS() as ddgs:
            return list(ddgs.text(args.query, max_results=args.max_results))

    hits = await asyncio.to_thread(_run)
    cleaned = [
        {
            "title": h.get("title", ""),
            "url": h.get("href") or h.get("url", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]
    return json.dumps(cleaned, ensure_ascii=False)


async def _fetch_url(args: FetchUrlArgs) -> str:
    # Reject non-http(s) schemes — defense in depth against SSRF tricks.
    if not (args.url.startswith("http://") or args.url.startswith("https://")):
        return f"ERROR: refused non-http(s) url: {args.url}"

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=httpx.Timeout(10.0),
        headers={"User-Agent": "Mozilla/5.0 (ProductionAgent/1.0)"},
    ) as client:
        try:
            resp = await client.get(args.url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return f"ERROR fetching {args.url}: {e}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = re.sub(r"\n{2,}", "\n\n", soup.get_text(separator="\n")).strip()
    if len(text) > args.max_chars:
        text = text[: args.max_chars] + "\n\n...[truncated]"
    return text


_SAFE_NAMES: dict[str, Any] = {
    k: getattr(math, k) for k in dir(math) if not k.startswith("_")
}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})


async def _calculator(args: CalculatorArgs) -> str:
    expr = args.expression
    if re.search(r"[a-zA-Z_]\s*=", expr) or "__" in expr:
        return "ERROR: unsafe expression"
    try:
        return str(eval(expr, {"__builtins__": {}}, _SAFE_NAMES))
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any], Awaitable[str]]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


TOOLS: dict[str, Tool] = {
    "web_search": Tool(
        name="web_search",
        description=(
            "Search the public web for a query. Returns a JSON list of "
            "results with title, url, snippet. Use to discover sources."
        ),
        args_model=WebSearchArgs,
        fn=_web_search,
    ),
    "fetch_url": Tool(
        name="fetch_url",
        description=(
            "Download a URL and return its visible text content (truncated). "
            "Use after web_search to read promising results in detail."
        ),
        args_model=FetchUrlArgs,
        fn=_fetch_url,
    ),
    "calculator": Tool(
        name="calculator",
        description=(
            "Evaluate a Python math expression. Use for any non-trivial "
            "arithmetic or unit conversion."
        ),
        args_model=CalculatorArgs,
        fn=_calculator,
    ),
}


def openai_tool_schemas() -> list[dict[str, Any]]:
    return [t.to_openai_schema() for t in TOOLS.values()]


# ---------------------------------------------------------------------------
# Dispatch with validation, timeout, and structured logging
# ---------------------------------------------------------------------------


async def dispatch_tool(
    name: str, raw_args: dict[str, Any], timeout_s: float
) -> tuple[str, int, str | None]:
    """Returns (output_string, duration_ms, error_or_None)."""
    started = time.perf_counter()
    tool = TOOLS.get(name)
    if tool is None:
        return f"ERROR: unknown tool '{name}'", 0, "unknown_tool"

    # 1. validate args
    try:
        args = tool.args_model.model_validate(raw_args)
    except ValidationError as e:
        return f"ERROR: invalid args: {e}", 0, "validation_error"

    # 2. run with timeout
    try:
        output = await asyncio.wait_for(tool.fn(args), timeout=timeout_s)
        err: str | None = None
    except asyncio.TimeoutError:
        output = f"ERROR: tool '{name}' timed out after {timeout_s}s"
        err = "timeout"
    except Exception as e:  # noqa: BLE001 — surface to model + logs
        output = f"ERROR: tool '{name}' failed: {e}"
        err = "exception"
        log.exception("tool_failed", tool=name)

    # 3. cap output to keep context window sane
    if len(output) > 6000:
        output = output[:6000] + "\n...[truncated]"

    duration_ms = int((time.perf_counter() - started) * 1000)
    log.info(
        "tool_call",
        tool=name,
        duration_ms=duration_ms,
        ok=err is None,
        error=err,
    )
    return output, duration_ms, err
