FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.11-slim AS runtime

LABEL maintainer="zeynepyesilot"

WORKDIR /app

COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY model/ ./model/

RUN mkdir -p /app/saved_models

RUN addgroup --system api && \
    adduser --system --ingroup api api && \
    chown -R api:api /app
USER api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

