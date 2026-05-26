# --- Build stage ---
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml README.md ./
ARG INSTALL_EXTRAS="."
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python -e "${INSTALL_EXTRAS}"

# --- Runtime stage ---
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "services.twin_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
