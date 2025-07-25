"""
Unified Configuration System for Trading RL Agent.

This module provides a single, consolidated configuration system that supersedes
all other configuration modules in the codebase.
"""

# Re-export the unified configuration system
from .core.unified_config import UnifiedConfig, load_config

# Import legacy configurations for backward compatibility
try:
    LEGACY_CONFIG_AVAILABLE = True
except ImportError:
    LEGACY_CONFIG_AVAILABLE = False

from .core.logging import get_logger

# Global configuration instance
_global_config: UnifiedConfig | None = None

def get_unified_config() -> UnifiedConfig:
    """Get the global unified configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config

def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _global_config
    _global_config = None

# Backward compatibility aliases
get_settings = get_unified_config
load_settings = load_config

__all__ = [
    "UnifiedConfig",
    "get_logger",
    "get_settings",
    "get_unified_config",
    "load_config",
    "load_settings",
    "reset_config"
]
