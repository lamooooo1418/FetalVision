# FetalVision inference service (CPU ONNX Runtime). Build from the repo root:
#   docker build -t fetalvision-service .
#   docker run --rm -p 8000:8000 fetalvision-service
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FETALVISION_SERVICE_CONFIG=/app/configs/service.yaml FETALVISION_MODELS=/app/models FETALVISION_DATA=/data
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements-service.txt ./
RUN pip install -r requirements-service.txt
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps -e .
COPY configs/service.yaml ./configs/service.yaml
COPY models/*.onnx models/*.json ./models/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "fetalvision.service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
