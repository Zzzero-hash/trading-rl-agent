#!/bin/bash
# Comprehensive locale fix for trade-agent development environment
# This script should be sourced at the beginning of any shell session

# Set locale to prevent "setlocale: LC_ALL: cannot change locale" warnings
export LANG=C.utf8
export LC_ALL=C.utf8
export LC_CTYPE=C.utf8

# Ensure the locale is available
if ! locale -a 2>/dev/null | grep -q "C.utf8"; then
    # Fallback to C locale if C.utf8 is not available
    export LANG=C
    export LC_ALL=C
    export LC_CTYPE=C
fi

echo "✅ Locale configured: LANG=$LANG, LC_ALL=$LC_ALL"

# Add this to shell profiles if not already present
for profile in ~/.bashrc ~/.profile; do
    if [ -f "$profile" ] && ! grep -q "source.*locale-fix.sh" "$profile" 2>/dev/null; then
        echo "# Source locale fix for trade-agent" >> "$profile"
        echo "[ -f /workspaces/trade-agent/scripts/locale-fix.sh ] && source /workspaces/trade-agent/scripts/locale-fix.sh" >> "$profile"
    fi
done
