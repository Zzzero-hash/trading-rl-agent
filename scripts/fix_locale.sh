#!/bin/bash
# Locale fix for Git operations
# Source this file to fix locale warnings: source scripts/fix_locale.sh

export LANG=C.utf8
export LC_ALL=C.utf8
export LANGUAGE=C.utf8

echo "✅ Locale environment variables set to C.utf8"
echo "   LANG=$LANG"
echo "   LC_ALL=$LC_ALL"
echo "   LANGUAGE=$LANGUAGE"
