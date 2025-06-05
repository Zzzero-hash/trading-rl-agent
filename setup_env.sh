#!/bin/bash
# Setup a local Python virtual environment with all project dependencies.

set -e

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=".venv"

# Install system build tools if using a Debian based OS
if command -v apt-get >/dev/null; then
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        build-essential wget autoconf automake libtool pkg-config \
        software-properties-common cmake git python3-dev libssl-dev \
        libgl1 libglib2.0-0
fi

# Build and install TA-Lib C library if missing
if ! ldconfig -p | grep -q libta_lib; then
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    pushd ta-lib >/dev/null
    ./configure --prefix=/usr/local
    make
    sudo make install
    popd >/dev/null
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
    sudo ldconfig
fi

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Install numpy first to avoid compiled wheel issues
pip install numpy==1.23.5

# Install remaining requirements including Ray with Tune extras
pip install -r requirements.txt --ignore-installed blinker

# Install the package in editable mode
pip install -e .

echo "Environment setup complete. Activate with 'source $VENV_DIR/bin/activate'."
