#!/bin/bash
# Setup a local Python virtual environment with all project dependencies.

set -e

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

# Build TA-Lib C library if not already installed
if ! ldconfig -p | grep -q libta_lib.so; then
    wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    (cd ta-lib && ./configure && make && make install)
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
fi

# Set library paths for TA-Lib Python wheel
export TA_LIBRARY_PATH=/usr/local/lib
export TA_INCLUDE_PATH=/usr/local/include

# Install numpy first to avoid compiled wheel issues
pip install numpy==1.23.5

# Install remaining requirements including Ray with Tune extras
TA_LIBRARY_PATH=/usr/local/lib TA_INCLUDE_PATH=/usr/local/include \
    pip install -r requirements.txt --ignore-installed blinker

# Install the package in editable mode
pip install -e .

echo "Environment setup complete. Activate with 'source $VENV_DIR/bin/activate'."
