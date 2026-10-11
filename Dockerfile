# syntax=docker/dockerfile:1
# API image: FastAPI + XGBoost (CPU build) + ONNX Runtime. No PyTorch, no GPU libraries.
# Build from the repo root. The model comes from deploy/bundle/ (committed; `make publish-bundle`);
# override with --build-arg BUNDLE=<dir> (e.g. a synthetic test bundle).
FROM python:3.11-slim AS builder
ARG BUNDLE=deploy/bundle
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /build
COPY requirements.txt .
# Serving needs no GPU and no Parquet: CPU-only XGBoost, no NVIDIA libraries, no PyArrow.
RUN grep -vE '^(xgboost|nvidia-|pyarrow)' requirements.txt > serve.txt \
 && pip install "xgboost-cpu==$(grep -E '^xgboost==' requirements.txt | cut -d= -f3)" -r serve.txt
# Only ship ONNX Runtime when the bundle contains a deep (ONNX) model.
COPY ${BUNDLE}/manifest.json /tmp/manifest.json
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . \
 && if ! grep -q '"kind": "onnx"' /tmp/manifest.json; then pip uninstall -y onnxruntime; fi \
 && pip uninstall -y pip \
 && find /opt/venv -depth -type d \( -name tests -o -name __pycache__ \) -exec rm -rf {} +

FROM python:3.11-slim
ARG BUNDLE=deploy/bundle
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 \
    OWFD_PROJECT_ROOT=/app \
    OWFD_SERVING_BUNDLE=/app/bundle \
    OWFD_DATABASE_URL=sqlite:////data/owfd.db
WORKDIR /app
COPY configs ./configs
COPY demo_data ./demo_data
COPY ${BUNDLE} ./bundle
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import sys, urllib.request as u; sys.exit(u.urlopen('http://localhost:8000/health').status != 200)"
CMD ["sh", "-c", "uvicorn owfd.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
