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

# Install numpy first to avoid compiled wheel issues
pip install numpy==1.23.5

# Install remaining requirements including Ray with Tune extras
pip install -r requirements.txt --ignore-installed blinker

# Install the package in editable mode
pip install -e .

echo "Environment setup complete. Activate with 'source $VENV_DIR/bin/activate'."
