#!/bin/bash
# Documentation build wrapper script with locale fixes

# Set proper locale for Sphinx (using available locale)
export LANG=C.utf8
export LC_ALL=C.utf8

# Add local bin to PATH for sphinx tools
export PATH="/home/rluser/.local/bin:$PATH"

# Change to docs directory
cd "$(dirname "$0")/../docs"

# Set SPHINXBUILD to use proper locale
export SPHINXBUILD="env LANG=C.utf8 LC_ALL=C.utf8 sphinx-build"

# Run make with the command passed as argument
make "$@"
