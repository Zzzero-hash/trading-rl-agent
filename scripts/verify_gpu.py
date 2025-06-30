#!/usr/bin/env python3
"""GPU Verification Script for Trading RL Agent.

Checks if GPU is available and properly configured in the container.
"""

import os
import subprocess
import sys


def check_nvidia_smi():
    """Check if nvidia-smi is available and working."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ nvidia-smi is working:")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi failed:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timed out")
        return False
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return False


def check_pytorch_cuda():
    """Check if PyTorch can see CUDA."""
    try:
        import torch

        print(f"✅ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print("✅ CUDA is available in PyTorch")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ Number of GPUs: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"✅ GPU {i}: {gpu_name}")

            return True
        else:
            print("❌ CUDA not available in PyTorch")
            return False

    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch CUDA: {e}")
        return False


def check_environment_variables():
    """Check GPU-related environment variables."""
    print("🔍 GPU Environment Variables:")
    gpu_vars = [
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "CUDA_VISIBLE_DEVICES",
    ]

    for var in gpu_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")


def check_pynvml():
    """Check if pynvml (NVIDIA GPU monitoring) is working."""
    try:
        import pynvml

        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✅ pynvml working - Found {device_count} GPU(s)")

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"  GPU {i}: {name}")
            print(
                f"    Memory: {memory_info.used / 1024**2:.0f}MB / {memory_info.total / 1024**2:.0f}MB"
            )

        return True

    except ImportError:
        print("❌ pynvml not installed")
        return False
    except Exception as e:
        print(f"❌ Error with pynvml: {e}")
        return False


def main():
    """Run all GPU verification checks."""
    print("🚀 GPU Verification for Trading RL Agent")
    print("=" * 50)

    checks = [
        ("NVIDIA SMI", check_nvidia_smi),
        ("PyTorch CUDA", check_pytorch_cuda),
        ("Environment Variables", check_environment_variables),
        ("PYNVML", check_pynvml),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n📋 Checking {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    all_passed = True
    for name, result in results:
        if name == "Environment Variables":
            continue  # Skip env vars in summary
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False

    if all_passed:
        print("\n🎉 All GPU checks passed! Your container is ready for GPU training.")
        sys.exit(0)
    else:
        print("\n⚠️  Some GPU checks failed. See details above.")
        print("💡 Troubleshooting tips:")
        print("  1. Ensure NVIDIA drivers are installed on the host")
        print("  2. Install nvidia-container-toolkit on the host")
        print("  3. Restart Docker daemon after installing nvidia-container-toolkit")
        print("  4. Rebuild the dev container")
        sys.exit(1)


if __name__ == "__main__":
    main()
