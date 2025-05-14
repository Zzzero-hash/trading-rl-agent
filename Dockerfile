ARG BASE_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

# Create non-root user
RUN groupadd -g 1000 rluser && useradd -m -u 1000 -g rluser rluser

# Install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential wget autoconf automake libtool pkg-config \
      software-properties-common cmake git python3-dev python3-pip \
      libssl-dev libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Build & install TA‑Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Symlink to satisfy '-lta-lib' linker flag
RUN ln -s /usr/local/lib/libta_lib.so /usr/local/lib/libta-lib.so && \
    ln -s /usr/local/lib/libta_lib.a  /usr/local/lib/libta-lib.a && \
    ldconfig

# Install Python dependencies
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.23.5 && \
    pip install --no-cache-dir -r requirements.txt --ignore-installed blinker
COPY . .

USER rluser
CMD ["bash"]