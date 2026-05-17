# Research Agent — production build

Async FastAPI service that runs a ReAct research agent with proper guardrails, observability, and persistence. No LangChain, no AutoGPT — just the OpenAI SDK and a small loop you can fully reason about.

## Why no framework

Frameworks like LangChain hide your prompts, change APIs every few months, and add abstractions you have to debug. For production you want every token sent to the model in your repo, every retry policy explicit, and every guardrail visible. This project shows the pattern Anthropic and many production teams advocate: **direct SDK + a thin loop you own**.

## What's in here

```
production-agent/
├── app/
│   ├── api.py            FastAPI app: /research, /runs/{id}, /healthz
│   ├── agent.py          Async ReAct loop with circuit breaker
│   ├── llm.py            AsyncOpenAI wrapper: timeout, retry, cost tracking
│   ├── tools.py          Async tools w/ Pydantic validation + per-call timeout
│   ├── models.py         Pydantic API + tool-arg models
│   ├── storage.py        SQLite persistence of every run
│   ├── config.py         pydantic-settings, validates env at startup
│   └── logging_setup.py  structlog JSON logging w/ run_id, request_id
├── main.py               uvicorn entrypoint
├── requirements.txt
├── Dockerfile            multi-stage, non-root, healthcheck
├── .dockerignore
└── .env.example
```

## Production patterns demonstrated

| Concern | How it's handled | Where |
|--------|------------------|-------|
| Config | env vars validated by Pydantic at startup | `config.py` |
| Logging | structlog → JSON → stdout, every line tagged with `request_id`/`run_id` | `logging_setup.py`, `agent.py`, `api.py` |
| Cost control | per-run `CostBreakdown`, hard `max_cost_usd` ceiling | `llm.py`, `agent.py` |
| Retries | tenacity, exponential backoff, only on transient errors | `llm.py` |
| Timeouts | LLM call timeout + per-tool timeout | `llm.py`, `tools.py` |
| Circuit breaker | `max_iterations` + `max_wall_time_s` + `max_cost_usd` | `agent.py` |
| Tool input validation | Pydantic models reject malformed LLM args before execution | `tools.py` |
| SSRF defense | `fetch_url` rejects non-http(s) schemes | `tools.py` |
| Safe `eval` | restricted globals, blocks `__` and assignment | `tools.py` |
| Persistence | SQLite, every run captured for replay/debugging | `storage.py` |
| Concurrency | `asyncio` + `AsyncOpenAI` + `httpx`, blocking deps offloaded with `asyncio.to_thread` | throughout |
| Container | multi-stage build, non-root user, healthcheck | `Dockerfile` |
| Graceful shutdown | FastAPI lifespan closes the LLM client and DB | `api.py` |
| Error handling | global exception handler, never leaks stack traces in 500s | `api.py` |
| Observability | log every iteration, every tool call, with timings | `agent.py`, `tools.py` |

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env       # edit OPENAI_API_KEY
python main.py
```

Then:

```bash
curl -X POST http://localhost:8000/research \
  -H "content-type: application/json" \
  -d '{"question":"How much electricity does training GPT-4 use, roughly?"}'
```

Response includes `run_id`. Fetch the full transcript later with:

```bash
curl http://localhost:8000/runs/<run_id>
```

## Run in Docker

```bash
docker build -t research-agent .
docker run --rm -p 8000:8000 --env-file .env research-agent
```

## What you get from a run

- Final answer, with inline `[1]`, `[2]` citations.
- Every tool call (name, args, duration, error if any).
- Token usage and USD cost (live cumulative ceiling).
- `stopped_reason` ∈ `completed | max_iterations | max_cost | max_wall_time | error`.
- A `run_id` you can use to fetch the full transcript anytime.

## Where production usually goes next

- **Auth** on the API (API key header → middleware that checks it).
- **Rate limiting** per caller (Redis-backed token bucket).
- **Tracing** with OpenTelemetry — wrap `LLMClient.chat` and `dispatch_tool` in spans, ship to Jaeger/Datadog.
- **Streaming** responses (server-sent events) so clients see partial output.
- **Eval suite** — replay saved runs against new model versions and diff answer quality.
- **Postgres** instead of SQLite — change `storage.py` only.
- **Secrets** from a vault (AWS Secrets Manager, GCP Secret Manager) instead of `.env`.
- **Concurrency cap** — semaphore around `agent.run()` to bound how many concurrent runs spend money at once.
- **Domain allowlist/blocklist** for `fetch_url`.
- **Content moderation** on inputs and outputs if user-facing.

These are deliberately not added to keep the codebase readable. Adding them is straightforward because every cross-cutting concern already has a single home (config, logging, llm, tools, storage).

## Why this is easier to maintain than LangChain

- One agent loop, ~130 lines, every line is yours.
- Prompts live in plain Python strings in `agent.py` and `tools.py` — diffable, reviewable.
- Adding a tool = one Pydantic model + one async function + one entry in `TOOLS`.
- Swap the model provider (OpenAI ↔ Anthropic ↔ vLLM) by changing `llm.py` only.
- All behavior is observable in the logs without instrumentation magic.
