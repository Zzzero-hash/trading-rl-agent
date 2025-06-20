#!/bin/bash
# Pre-commit script for cleaning up notebook outputs and checking for large files
# Place this in .git/hooks/pre-commit and make it executable

echo "🧹 Pre-commit cleanup..."

# Check if we have any notebooks to clean
notebooks=$(find . -name "*.ipynb" -not -path "./.git/*" -not -path "./.ipynb_checkpoints/*" 2>/dev/null)

if [ ! -z "$notebooks" ]; then
    echo "📓 Clearing notebook outputs..."

    # Try to clear notebook outputs using nbconvert
    if command -v jupyter &> /dev/null; then
        echo "$notebooks" | xargs jupyter nbconvert --clear-output --inplace 2>/dev/null || {
            echo "⚠️  Warning: Could not clear notebook outputs automatically"
            echo "   Please clear notebook outputs manually before committing"
        }
    else
        echo "⚠️  Warning: jupyter not found"
        echo "   Please clear notebook outputs manually before committing"
    fi
fi

# Check for large files that shouldn't be committed
echo "🔍 Checking for large files..."
large_files=$(find . -size +50M -not -path "./.git/*" -not -path "./ray_results/*" -not -path "./optimization_results/*" 2>/dev/null | head -5)

if [ ! -z "$large_files" ]; then
    echo "⚠️  Warning: Found large files that may not belong in git:"
    echo "$large_files"
    echo ""
    echo "Consider adding these to .gitignore or cleaning them up"
fi

# Check for common temporary files
temp_files=$(find . -name "*.tmp" -o -name "*.temp" -o -name ".DS_Store" -o -name "Thumbs.db" 2>/dev/null)
if [ ! -z "$temp_files" ]; then
    echo "🗑️  Found temporary files that should be cleaned:"
    echo "$temp_files"
fi

echo "✅ Pre-commit checks complete"
