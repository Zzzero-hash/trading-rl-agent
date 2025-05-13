ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

RUN groupadd -g 1000 rluser && useradd -m -u 1000 -g rluser rluser
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget python3-dev python3-pip libssl-dev \
    libgl1 libglib2.0-0 libta-lib-dev

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

USER rluser
CMD ["bash"]