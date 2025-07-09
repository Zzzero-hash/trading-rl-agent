# Use CUDA base image (you can adjust CUDA_VARIANT as needed)
ARG CUDA_VARIANT=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
FROM ${CUDA_VARIANT} AS development

# 1. Install system/build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      vim nano curl wget git-lfs \
      htop nvtop \
      build-essential \
      software-properties-common cmake git python3.11 python3.11-dev python3.11-distutils python3.11-venv python3-pip \
      libssl-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 2. Set python3.11 as default and install base Python tools
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    python3 --version && \
    python3 -m pip install --upgrade pip

# 3. Install Python packages
RUN pip install --no-cache-dir \
        jupyter ipython black flake8 pytest debugpy

# Build stages for dependencies, testing, and runtime
FROM development AS deps
WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir -r requirements.txt --ignore-installed blinker

# Test stage using installed dependencies
FROM deps AS test
WORKDIR /workspace
COPY --from=deps /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY . .
ENV PYTHONPATH=/workspace
ENTRYPOINT ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
