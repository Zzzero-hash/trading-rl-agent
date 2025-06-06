ARG CUDA_VARIANT=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_VARIANT} AS base

# Create non-root user
RUN groupadd -g 1000 rluser && useradd -m -u 1000 -g rluser rluser

# Install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential wget autoconf automake libtool pkg-config \
      software-properties-common cmake git python3-dev python3-pip \
      libssl-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Stage 1: install Python dependencies
FROM base AS deps
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.23.5 && \
    # Install all Python deps including Ray with Tune extras
    pip install --no-cache-dir -r requirements.txt --ignore-installed blinker

# Add missing dependencies
RUN pip install --no-cache-dir pyyaml==6.0 yfinance==0.2.61

# Add missing Ray dependency for RLlib
RUN pip install "ray[rllib]" --no-cache-dir

# Stage 2: run tests
FROM base
ENV PYTHONPATH=/workspace
COPY . .
RUN pip install -e . && \
    pip install pytest && \
    pytest --maxfail=1 --disable-warnings -q

# Stage 3: final runtime image
FROM base
WORKDIR /workspace
# copy Python dependencies from deps stage (Debian installs to dist-packages)
COPY --from=deps /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin
# copy application code
COPY . .
COPY src/configs /cfg
USER rluser
ENTRYPOINT ["pytest", "--maxfail=1", "--disable-warnings", "-q"]