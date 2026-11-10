FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu' \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/build_artifacts.py ./scripts/build_artifacts.py

ARG DOCS_RAG_SOURCE_COMMIT
ARG DOCS_RAG_SOURCE_DIRTY
RUN python -m scripts.build_artifacts

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]