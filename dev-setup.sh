#!/bin/bash
# Development environment setup script
# Configures locale and development tools

# Fix locale issues in dev container - use available locale
export LANG=C.utf8
export LC_ALL=C.utf8

echo "🔧 Setting up development environment..."
echo "Locale set to: $LANG"

# Check if locale is properly configured
if locale -a | grep -q "C.utf8"; then
    echo "✅ Locale C.utf8 is available"
else
    echo "⚠️  Warning: C.utf8 locale not found, using C locale"
    export LANG=C
    export LC_ALL=C
fi

# Clean up any duplicate locale entries in .bashrc
if [ -f ~/.bashrc ]; then
    echo "📝 Cleaning up duplicate locale settings in ~/.bashrc"
    # Remove existing locale settings
    sed -i '/^export LANG=C\.utf8$/d' ~/.bashrc
    sed -i '/^export LC_ALL=C\.utf8$/d' ~/.bashrc
    sed -i '/^# Fix locale.*$/d' ~/.bashrc
    sed -i '/^$/N;/^\n$/d' ~/.bashrc  # Remove empty lines
fi

# Apply locale settings to user profile files
for profile_file in ~/.bashrc ~/.profile; do
    if [ -f "$profile_file" ] || [ "$profile_file" = ~/.bashrc ]; then
        if ! grep -q "export LANG=C.utf8" "$profile_file" 2>/dev/null; then
            echo "📝 Adding locale settings to $profile_file"
            echo "" >> "$profile_file"
            echo "# Fix locale for trading-rl-agent" >> "$profile_file"
            echo "export LANG=C.utf8" >> "$profile_file"
            echo "export LC_ALL=C.utf8" >> "$profile_file"
        fi
    fi
done

# Also set it in the global environment file for system-wide effect
echo "📝 Setting system-wide locale in /etc/environment"
if [ -w /etc/environment ]; then
    if ! grep -q "LANG=C.utf8" /etc/environment 2>/dev/null; then
        echo "LANG=C.utf8" | sudo tee -a /etc/environment > /dev/null
        echo "LC_ALL=C.utf8" | sudo tee -a /etc/environment > /dev/null
    fi
elif [ -f /etc/environment ]; then
    echo "⚠️  Cannot write to /etc/environment (no sudo access)"
fi

# Source this script to apply locale settings to current session
if [ "$0" = "${BASH_SOURCE[0]}" ]; then
    echo "💡 To apply locale settings to current session, run: source $0"
    echo "💡 Or restart your terminal/shell"
fi

# Install and configure pre-commit hooks
if ! command -v pre-commit &> /dev/null; then
    echo "📦 Installing pre-commit"
    pip install pre-commit
fi
if [ ! -f .git/hooks/pre-commit ]; then
    echo "🔧 Installing Git hooks via pre-commit"
    pre-commit install
fi

echo "🎉 Development environment setup complete!"
