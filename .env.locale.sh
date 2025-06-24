#!/bin/bash
# Universal locale fix script for trading-rl-agent
# This script should be sourced by shell init files

# Unset any conflicting locale variables that VSCode might set
unset LC_ALL 2>/dev/null || true
unset LANG 2>/dev/null || true
unset LANGUAGE 2>/dev/null || true

# Set all locale variables to available C.utf8
export LANG=C.utf8
export LC_ALL=C.utf8
export LC_CTYPE=C.utf8
export LC_NUMERIC=C.utf8
export LC_TIME=C.utf8
export LC_COLLATE=C.utf8
export LC_MONETARY=C.utf8
export LC_MESSAGES=C.utf8
export LC_PAPER=C.utf8
export LC_NAME=C.utf8
export LC_ADDRESS=C.utf8
export LC_TELEPHONE=C.utf8
export LC_MEASUREMENT=C.utf8
export LC_IDENTIFICATION=C.utf8

# For shells that don't handle export properly
LANG=C.utf8
LC_ALL=C.utf8

# Ensure git uses our locale settings
export GIT_COMMITTER_DATE_TIMEZONE_OFFSET=0
export GIT_AUTHOR_DATE_TIMEZONE_OFFSET=0
