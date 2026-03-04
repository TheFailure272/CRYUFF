# ──────────────────────────────────────────────────────────────
# C.R.U.Y.F.F. — Backend Dockerfile (multi-stage)
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# System deps for numpy/scipy wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY server/ server/
COPY engine/ engine/
COPY shared/ shared/
COPY workers/ workers/
COPY tests/ tests/

# ── Test stage ───────────────────────────────────────────────
FROM base AS test

RUN python -m pytest tests/ -v --tb=short

# ── Production stage ─────────────────────────────────────────
FROM base AS production

EXPOSE 8000

ENV REDIS_URL=redis://redis:6379/0

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
