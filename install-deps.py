#!/usr/bin/env python3
"""
Dependency Installation Helper for Trading RL Agent
Helps users install only the dependencies they need
"""

import subprocess
import sys
from pathlib import Path


class DependencyInstaller:
    """Helper class for managing dependencies"""

    def __init__(self) -> None:
        self.requirements_files = {
            "core": "requirements-core.txt",
            "ml": "requirements-ml.txt",
            "full": "requirements-full.txt",
            "dev": "requirements-dev.txt",
            "production": "requirements-production.txt",
        }

        self.install_options = {
            "1": ("core", "Core - Basic functionality only (~50MB)"),
            "2": ("ml", "ML - Core + Machine Learning (~2.1GB)"),
            "3": ("full", "Full - Production ready with all features (~2.6GB)"),
            "4": ("dev", "Dev - Development tools and testing (~500MB)"),
            "5": ("production", "Production - Optimized for deployment (~2.6GB)"),
        }

    def print_banner(self) -> None:
        """Print installation banner"""
        print("🚀 Trading RL Agent - Dependency Installer")
        print("=" * 50)
        print("Choose your installation profile:")
        print()

    def print_options(self) -> None:
        """Print available installation options"""
        for key, (_, description) in self.install_options.items():
            print(f"  {key}) {description}")
        print()

    def get_user_choice(self) -> str:
        """Get user's installation choice"""
        while True:
            choice = input("Enter your choice [1-5]: ").strip()
            if choice in self.install_options:
                return self.install_options[choice][0]
            print("❌ Invalid choice. Please enter 1-5.")

    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        if sys.version_info < (3, 10):  # noqa: UP036
            print("❌ Python 3.10+ required. Current version:", sys.version)
            return False
        print(f"✅ Python {sys.version.split()[0]} detected")
        return True

    def check_virtual_env(self) -> bool:
        """Check if running in virtual environment"""
        if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
            print("✅ Virtual environment detected")
            return True
        print("⚠️  No virtual environment detected")
        response = input("Continue anyway? (y/N): ").strip().lower()
        return response in ["y", "yes"]

    def install_dependencies(self, profile: str) -> bool:
        """Install dependencies for the specified profile"""
        requirements_file = self.requirements_files.get(profile)
        if not requirements_file or not Path(requirements_file).exists():
            print(f"❌ Requirements file {requirements_file} not found")
            return False

        print(f"\n📦 Installing {profile} dependencies...")
        print(f"📄 Using: {requirements_file}")

        try:
            # Use pip to install requirements
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", requirements_file],
                capture_output=True,
                text=True,
                check=True,
            )

            print("✅ Installation completed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print("❌ Installation failed:")
            print(f"Error: {e.stderr}")
            return False

    def install_with_pip_install(self, profile: str) -> bool:
        """Install using pip install with extras"""
        if profile == "core":
            # Core dependencies
            return self.install_dependencies("core")

        try:
            print(f"\n📦 Installing {profile} dependencies via pip...")

            if profile == "full":
                # Install with all extras
                cmd = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-e",
                    ".",
                    "[dev,ml,production]",
                ]
            else:
                # Install with specific extras
                cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{profile}]"]

            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Installation completed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print("❌ Installation failed:")
            print(f"Error: {e.stderr}")
            return False

    def show_manual_instructions(self) -> None:
        """Show manual installation instructions"""
        print("\n📋 Manual Installation Instructions:")
        print("=" * 40)
        print("1. Core dependencies only:")
        print("   pip install -r requirements-core.txt")
        print()
        print("2. Machine Learning capabilities:")
        print("   pip install -r requirements-ml.txt")
        print()
        print("3. Full production setup:")
        print("   pip install -r requirements-full.txt")
        print()
        print("4. Development tools:")
        print("   pip install -r requirements-dev.txt")
        print()
        print("5. Using pip install with extras:")
        print("   pip install -e .[ml]        # ML dependencies")
        print("   pip install -e .[dev]       # Development tools")
        print("   pip install -e .[production] # Production tools")
        print("   pip install -e .[full]      # Everything")

    def run(self) -> bool:
        """Main installation flow"""
        self.print_banner()

        # Check prerequisites
        if not self.check_python_version():
            return False

        if not self.check_virtual_env():
            return False

        self.print_options()

        # Get user choice
        profile = self.get_user_choice()

        # Ask for installation method
        print(f"\n🔧 Installation method for {profile}:")
        print("1) Use requirements file (recommended)")
        print("2) Use pip install with extras")
        print("3) Show manual instructions")

        method = input("Choose method [1-3]: ").strip()

        if method == "1":
            return self.install_dependencies(profile)
        if method == "2":
            return self.install_with_pip_install(profile)
        if method == "3":
            self.show_manual_instructions()
            return True
        print("❌ Invalid choice")
        return False


def main() -> None:
    """Main entry point"""
    installer = DependencyInstaller()

    try:
        success = installer.run()
        if success:
            print("\n🎉 Setup completed! You can now run:")
            print("   python minimal_test.py  # Test your installation")
            print("   python -m trading_rl_agent.main  # Start the system")
        else:
            print("\n❌ Setup failed. Check the error messages above.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
