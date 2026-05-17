"""
Settings loaded from environment variables / .env via pydantic-settings.

Why this pattern:
    * 12-factor: config lives in env, not code.
    * Pydantic validates types at startup so you fail fast on bad config
      instead of crashing in a hot path 20 minutes in.
    * One Settings() instance, imported everywhere via get_settings().
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM ----
    openai_api_key: str = Field(..., description="OpenAI API key (required).")
    openai_model: str = "gpt-4o-mini"
    cost_input_per_1k: float = 0.00015
    cost_output_per_1k: float = 0.0006

    # ---- Agent guardrails (the circuit breaker) ----
    max_iterations: int = 8
    max_wall_time_s: int = 120
    max_cost_usd: float = 0.50
    llm_timeout_s: int = 60
    tool_timeout_s: int = 15

    # ---- HTTP server ----
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # ---- Storage ----
    db_path: str = "./runs.db"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so we don't reparse env on every import."""
    return Settings()  # type: ignore[call-arg]
