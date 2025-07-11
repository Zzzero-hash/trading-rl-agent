import importlib
import pkgutil
import sys
from pathlib import Path


def check_imports():
    """Iterates through all Python modules in the 'src' directory and attempts to import them.

    This script is designed to catch any broken imports that may not be covered by the
    test suite. It adds the 'src' directory to the Python path and then recursively
    walks through the package to import each module.

    Any modules that fail to import will be printed to the console.
    """
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    print(f"Checking imports for all modules in: {src_path}")

    for module_info in pkgutil.walk_packages([str(src_path)], prefix=""):
        if module_info.name.startswith("trading_rl_agent.envs.finrl_trading_eng"):
            continue
        try:
            print(f"Importing {module_info.name}...")
            importlib.import_module(module_info.name)
        except Exception as e:
            print(f"Failed to import {module_info.name}: {e}")


if __name__ == "__main__":
    check_imports()
