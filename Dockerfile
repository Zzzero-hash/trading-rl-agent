ARG CUDA_VARIANT=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
FROM ${CUDA_VARIANT} AS development
ENV TZ=Etc/UTC


# Install build tools and build TA-Lib C library from source
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      vim nano curl wget git-lfs \
      htop nvtop \
      build-essential wget autoconf automake libtool pkg-config \
      software-properties-common cmake git python3-dev python3-pip \
      libssl-dev libgl1 libglib2.0-0 && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && ./configure --prefix=/usr && make && make install && cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz && \
    # Symlink to satisfy '-lta-lib' for some builds and run ldconfig
    ln -s /usr/lib/libta_lib.so /usr/lib/libta-lib.so 2>/dev/null || true && \
    ln -s /usr/lib/libta_lib.a /usr/lib/libta-lib.a 2>/dev/null || true && \
    ldconfig && \
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
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include
# Optionally, check that the header exists (for debug)
RUN ls -l /usr/include/ta-lib/ta_defs.h
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --global-option=build_ext --global-option="-I/usr/include" --global-option="-L/usr/lib" -r requirements.txt --ignore-installed blinker

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
# Run the tests before starting the application
ENTRYPOINT ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
