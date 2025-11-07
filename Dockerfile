# Dockerfile - use python:3.11-slim so binary wheels for scipy/numpy are available
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      gcc \
      gfortran \
      pkg-config \
      libblas-dev \
      liblapack-dev \
      libopenblas-dev \
      libffi-dev \
      libssl-dev \
      ca-certificates \
      curl && \
    rm -rf /var/lib/apt/lists/*

COPY app/ ./app
COPY models/ ./models

# upgrade pip/setuptools/wheel so pip prefers wheels
RUN python -m pip install --upgrade pip setuptools wheel

COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt

ENV MODEL_DIR=/app/models
ENV MODEL_FILE=book_item_model.pkl
ENV NN_FILE=nn_index.pkl
ENV PORT=10000

EXPOSE ${PORT}
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
