# syntax=docker/dockerfile:1

# ---- Build stage --------------------------------------------------------
# Compiles/installs Python dependencies. Several packages here (notably
# chroma-hnswlib, a ChromaDB dependency) have C extensions and don't always
# ship prebuilt wheels for every platform — build-essential is only needed
# here, in this stage, and never ends up in the final image.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# --prefix=/install (not --user): installs into a clean, self-contained
# directory tree that's trivial to copy into the final stage below, and
# avoids the classic Docker gotcha where `pip install --user` writes to
# /root/.local — a directory the non-root runtime user can't traverse into
# because /root itself is 700 by default.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Final stage ---------------------------------------------------------
# Slim runtime image: no compiler toolchain, no build-time-only packages.
FROM python:3.12-slim

# curl is the only runtime system dependency — used by HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installed Python packages from the build stage. /usr/local is
# world-readable/executable by default, so the non-root user below can use
# them with no extra permission fixes.
COPY --from=builder /install /usr/local

COPY app ./app
COPY .env.example .env.example

# Run as a non-root user — standard container security practice; nothing in
# this app needs root privileges at runtime.
RUN useradd --create-home appuser \
    && mkdir -p /app/data/chroma \
    && chown -R appuser:appuser /app
USER appuser

# ChromaDB's persistence directory — docker-compose.yml mounts a named
# volume here so vector data survives container restarts/rebuilds.
VOLUME ["/app/data"]

EXPOSE 8000

# Lets `docker ps` / docker-compose show real health status, and gives
# depends_on: condition: service_healthy (used by other services, if any
# are ever added) something meaningful to wait on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
