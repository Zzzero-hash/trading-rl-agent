#!/bin/sh
# Git wrapper script to ensure proper locale is set
# This prevents "setlocale: LC_ALL: cannot change locale" warnings

# Set locale environment variables for Git operations
export LANG=C.utf8
export LC_ALL=C.utf8

# Execute git with all provided arguments
exec git "$@"
