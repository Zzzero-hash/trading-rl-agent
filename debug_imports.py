#!/usr/bin/env python3

# Debug script to test imports
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/workspaces/trading-rl-agent')

print("Testing direct import from configs.py...")

try:
    # Import the module directly
    import src.agents.configs as configs_module
    print("Module imported successfully")
    print("Available attributes:", dir(configs_module))
    
    # Try to access each class
    if hasattr(configs_module, 'EnsembleConfig'):
        print("✓ EnsembleConfig found")
    else:
        print("✗ EnsembleConfig NOT found")
        
    if hasattr(configs_module, 'SACConfig'):
        print("✓ SACConfig found")
    else:
        print("✗ SACConfig NOT found")
        
    if hasattr(configs_module, 'TD3Config'):
        print("✓ TD3Config found")
    else:
        print("✗ TD3Config NOT found")

except Exception as e:
    print(f"Import failed: {e}")
    
print("\nTesting individual class imports...")

try:
    from src.agents.configs import EnsembleConfig
    print("✓ EnsembleConfig imported successfully")
except Exception as e:
    print(f"✗ EnsembleConfig import failed: {e}")

try:
    from src.agents.configs import SACConfig
    print("✓ SACConfig imported successfully")
except Exception as e:
    print(f"✗ SACConfig import failed: {e}")

try:
    from src.agents.configs import TD3Config
    print("✓ TD3Config imported successfully")
except Exception as e:
    print(f"✗ TD3Config import failed: {e}")
