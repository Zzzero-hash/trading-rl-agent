ARG CUDA_VARIANT=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
FROM ${CUDA_VARIANT} AS development
ENV TZ=Etc/UTC

# Create non-root user
RUN groupadd -g 1000 rluser && useradd -m -u 1000 -g rluser rluser

# Install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      vim nano curl wget git-lfs \
      htop nvtop \
      build-essential wget autoconf automake libtool pkg-config \
      software-properties-common cmake git python3-dev python3-pip \
      libssl-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    black \
    flake8 \
    pytest \
    debugpy

FROM development AS deps
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --ignore-installed blinker

# Stage 2: run tests
FROM development AS test
ENV PYTHONPATH=/workspace

# Stage 3: final runtime image
FROM development AS runtime
WORKDIR /workspace
# copy Python dependencies from deps stage
COPY --from=deps /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=deps /usr/local/bin /usr/local/bin
# copy application code
COPY . .
# Change ownership of ALL files and directories to rluser with comprehensive permissions
RUN chown -R rluser:rluser /workspace && \
    find /workspace -type d -exec chmod 755 {} \; && \
    find /workspace -type f -name "*.sh" -exec chmod 755 {} \; && \
    find /workspace -type f -name "*.py" -exec chmod 644 {} \; && \
    find /workspace -type f ! -name "*.sh" ! -name "*.py" -exec chmod 644 {} \; && \
    chmod -R u+w /workspace
USER rluser
ENTRYPOINT ["pytest", "--maxfail=1", "--disable-warnings", "-q"]