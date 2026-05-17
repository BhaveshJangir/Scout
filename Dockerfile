# ---- build stage ----------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# ---- runtime stage --------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/app/.local/bin:${PATH}"

# Run as non-root.
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Pull installed deps from the build stage.
COPY --from=build /root/.local /home/app/.local

# App code (copied last so dep layer caches across edits).
COPY --chown=app:app app/ ./app/
COPY --chown=app:app main.py ./

EXPOSE 8000

# Healthcheck hits the liveness endpoint.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz', timeout=2).status==200 else 1)"

CMD ["python", "main.py"]
