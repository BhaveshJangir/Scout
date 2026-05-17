"""
Async OpenAI client wrapper.

Adds three things on top of the raw SDK:
    * a hard timeout
    * exponential-backoff retries on transient errors only
    * usage tracking (tokens in/out + USD cost) into a CostBreakdown

We deliberately do NOT retry on 4xx errors that indicate a bad request —
retrying those just burns money and pollutes logs.
"""

from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings
from .logging_setup import get_logger
from .models import CostBreakdown

log = get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=0,  # we handle retries ourselves with tenacity
        )

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        cost: CostBreakdown,
    ) -> ChatCompletion:
        """Single chat completion with retry + cost accumulation.

        `cost` is mutated in place so the caller can enforce a budget across
        many calls in one run.
        """

        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(
                (RateLimitError, APIConnectionError, APIError)
            ),
        ):
            with attempt:
                resp = await self._client.chat.completions.create(
                    model=self._settings.openai_model,
                    messages=messages,
                    tools=tools or None,
                    tool_choice="auto" if tools else None,
                    temperature=0.2,
                )

        # Update running cost. Some responses don't include usage (e.g. some
        # streaming modes); guard against missing fields.
        usage = getattr(resp, "usage", None)
        if usage:
            cost.input_tokens += usage.prompt_tokens or 0
            cost.output_tokens += usage.completion_tokens or 0
            cost.usd = round(
                cost.input_tokens / 1000 * self._settings.cost_input_per_1k
                + cost.output_tokens / 1000 * self._settings.cost_output_per_1k,
                6,
            )

        log.info(
            "llm_call",
            model=self._settings.openai_model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            cumulative_usd=cost.usd,
        )
        return resp

    async def aclose(self) -> None:
        await self._client.close()
