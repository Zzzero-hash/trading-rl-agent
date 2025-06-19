#!/bin/bash
# Documentation build wrapper script with locale fixes

# Set proper locale for Sphinx
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Change to docs directory
cd "$(dirname "$0")/../docs"

# Set SPHINXBUILD to use proper locale
export SPHINXBUILD="env LANG=C.UTF-8 LC_ALL=C.UTF-8 sphinx-build"

# Run make with the command passed as argument
make "$@"
