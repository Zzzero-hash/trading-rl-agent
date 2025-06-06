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
    # Install all Python deps from requirements.txt
    # This includes NumPy, Ray with all extras, onnxruntime, etc.
    # The --ignore-installed blinker is kept from the original file.
    pip install --no-cache-dir -r requirements.txt --ignore-installed blinker

# Stage 2: run tests
FROM base AS test
ENV PYTHONPATH=/workspace
COPY . .
# Copy sitecustomize for numpy 2.0 compatibility
COPY sitecustomize.py /usr/local/lib/python3.10/dist-packages/sitecustomize.py
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
# Copy sitecustomize for numpy 2.0 compatibility
COPY sitecustomize.py /usr/local/lib/python3.10/dist-packages/sitecustomize.py
COPY src/configs /cfg
USER rluser
ENTRYPOINT ["pytest", "--maxfail=1", "--disable-warnings", "-q"]