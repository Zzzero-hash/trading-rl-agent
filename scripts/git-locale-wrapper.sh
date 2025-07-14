#!/bin/sh
# VSCode Git Environment Wrapper
# This script wraps git commands to ensure proper locale is set
# even when VSCode forces en_US.UTF-8

# Completely override any locale settings VSCode might have set
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
export LANGUAGE=

# Redirect stderr to suppress locale warnings during the git command
exec 2>/dev/null

# Execute the actual git command
exec /usr/bin/git "$@" 2>&1
