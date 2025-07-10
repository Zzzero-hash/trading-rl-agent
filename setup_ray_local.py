#!/usr/bin/env python3
"""
Local Ray Setup for CNN+LSTM Training Development

This script sets up Ray locally for development and testing of distributed
CNN+LSTM training before deploying to production clusters.
"""

import subprocess
import sys
import time
from pathlib import Path

import psutil
import ray
import torch


def check_system_resources():
    """Check available system resources."""
    print("🔍 Checking system resources...")

    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    memory = psutil.virtual_memory()

    print(f"  💻 CPU cores: {cpu_count} logical, {cpu_physical} physical")
    print(f"  🧠 Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")

    # GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  🎮 GPUs: {gpu_count} available")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("  🎮 GPUs: None available")

    return {
        "cpu_count": cpu_count,
        "memory_gb": memory.total / (1024**3),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def install_ray_dependencies():
    """Install required Ray dependencies."""
    print("📦 Installing Ray dependencies...")

    required_packages = [
        "ray[tune,train]",
        "optuna",
        "hyperopt",
        "bayesian-optimization",
        "tensorboard",
        "wandb",
    ]

    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            return False

    return True


def setup_ray_cluster(resources):
    """Set up Ray cluster with optimal configuration."""
    print("🚀 Setting up Ray cluster...")

    # Stop any existing Ray processes
    try:
        ray.shutdown()
    except Exception:
        pass

    # Calculate optimal resource allocation
    num_cpus = max(1, resources["cpu_count"] - 1)  # Leave 1 CPU for system
    memory_per_worker = min(2 * 1024**3, resources["memory_gb"] * 0.8 * 1024**3 / 4)  # Max 2GB per worker

    # GPU configuration
    num_gpus = resources["gpu_count"]

    # Object store configuration (25% of available memory)
    object_store_memory = int(resources["memory_gb"] * 0.25 * 1024**3)

    print(f"  🎯 CPUs: {num_cpus}")
    print(f"  🎮 GPUs: {num_gpus}")
    print(f"  🧠 Object store: {object_store_memory / 1024**3:.1f}GB")

    # Initialize Ray
    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        object_store_memory=object_store_memory,
        include_dashboard=True,
        dashboard_host="localhost",
        dashboard_port=8265,
        _temp_dir="/tmp/ray",
        log_to_driver=True,
        configure_logging=True,
        logging_level="INFO",
    )

    print("  ✅ Ray initialized successfully!")
    print("  📊 Dashboard: http://localhost:8265")
    print(f"  🔗 Cluster resources: {ray.cluster_resources()}")

    return True


def test_ray_functionality():
    """Test basic Ray functionality."""
    print("🧪 Testing Ray functionality...")

    @ray.remote
    def cpu_task(x):
        import time

        time.sleep(0.1)
        return x * x

    @ray.remote(num_gpus=0.1 if torch.cuda.is_available() else 0)
    def gpu_task(x):
        import time

        time.sleep(0.1)
        return x * 2

    # Test CPU tasks
    start_time = time.time()
    cpu_futures = [cpu_task.remote(i) for i in range(10)]
    cpu_results = ray.get(cpu_futures)
    cpu_time = time.time() - start_time

    print(f"  ✅ CPU tasks: {len(cpu_results)} completed in {cpu_time:.2f}s")

    # Test GPU tasks if available
    if torch.cuda.is_available():
        start_time = time.time()
        gpu_futures = [gpu_task.remote(i) for i in range(5)]
        gpu_results = ray.get(gpu_futures)
        gpu_time = time.time() - start_time

        print(f"  ✅ GPU tasks: {len(gpu_results)} completed in {gpu_time:.2f}s")
    else:
        print("  ⚠️ GPU tasks: Skipped (no GPU available)")

    return True


def create_ray_training_example():
    """Create a simple example of Ray Tune training."""
    print("📝 Creating Ray training example...")

    example_code = '''
#!/usr/bin/env python3
"""
Example: Ray Tune CNN+LSTM Training

Run this script to test Ray Tune optimization locally:
    python ray_training_example.py
"""

import ray
from ray import tune
import numpy as np
import torch
import torch.nn as nn

# Simple CNN+LSTM model for testing
class SimpleCNNLSTM(nn.Module):
    def __init__(self, input_dim, cnn_filters, lstm_units):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(cnn_filters, lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, sequence)
        x = torch.relu(self.conv(x))
        x = x.transpose(1, 2)  # (batch, sequence, features)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use last timestep
        return x

def train_model(config):
    """Training function for Ray Tune."""
    # Generate synthetic data
    batch_size = 32
    sequence_length = 20
    input_dim = 10

    X = torch.randn(100, sequence_length, input_dim)
    y = torch.randn(100, 1)

    # Create model
    model = SimpleCNNLSTM(
        input_dim=input_dim,
        cnn_filters=config["cnn_filters"],
        lstm_units=config["lstm_units"]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Report to Ray Tune
        tune.report(loss=loss.item(), epoch=epoch)

if __name__ == "__main__":
    # Initialize Ray
    ray.init(address="auto", ignore_reinit_error=True)

    # Define search space
    search_space = {
        "cnn_filters": tune.choice([32, 64, 128]),
        "lstm_units": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-2),
    }

    # Run optimization
    analysis = tune.run(
        train_model,
        config=search_space,
        num_samples=5,
        resources_per_trial={"cpu": 1, "gpu": 0.1 if torch.cuda.is_available() else 0}
    )

    print("Best config:", analysis.get_best_config(metric="loss", mode="min"))
    ray.shutdown()
'''

    example_path = Path("ray_training_example.py")
    with open(example_path, "w") as f:
        f.write(example_code)

    print(f"  ✅ Example saved to: {example_path}")
    print(f"  🚀 Run with: python {example_path}")

    return str(example_path)


def setup_monitoring():
    """Set up monitoring and logging."""
    print("📊 Setting up monitoring...")

    # Create monitoring directories
    monitoring_dirs = [
        "ray_logs",
        "ray_results",
        "tensorboard_logs",
        "checkpoints",
    ]

    for dir_name in monitoring_dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  📁 Created: {dir_name}/")

    # Create monitoring script
    monitor_script = '''#!/usr/bin/env python3
"""
Ray Cluster Monitoring Script
"""

import ray
import time
import psutil

def monitor_cluster():
    """Monitor Ray cluster resources."""
    if not ray.is_initialized():
        print("❌ Ray is not initialized")
        return

    while True:
        try:
            # Ray cluster info
            resources = ray.cluster_resources()
            print(f"\\n🔍 Ray Cluster Status:")
            print(f"  CPUs: {resources.get('CPU', 0)}")
            print(f"  GPUs: {resources.get('GPU', 0)}")
            print(f"  Memory: {resources.get('memory', 0) / 1e9:.1f}GB")
            print(f"  Object Store: {resources.get('object_store_memory', 0) / 1e9:.1f}GB")

            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            print(f"\\n💻 System Status:")
            print(f"  CPU usage: {cpu_percent:.1f}%")
            print(f"  Memory usage: {memory.percent:.1f}%")

            time.sleep(10)

        except KeyboardInterrupt:
            print("\\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_cluster()
'''

    with open("monitor_ray.py", "w") as f:
        f.write(monitor_script)

    print("  ✅ Monitor script: monitor_ray.py")

    return True


def main():
    """Main setup function."""
    print("🚀 Setting up Ray for CNN+LSTM Training")
    print("=" * 50)

    try:
        # Check system resources
        resources = check_system_resources()

        # Install dependencies
        if not install_ray_dependencies():
            print("❌ Failed to install dependencies")
            return False

        # Setup Ray cluster
        if not setup_ray_cluster(resources):
            print("❌ Failed to setup Ray cluster")
            return False

        # Test functionality
        if not test_ray_functionality():
            print("❌ Ray functionality test failed")
            return False

        # Create training example
        example_path = create_ray_training_example()

        # Setup monitoring
        setup_monitoring()

        print("\n🎉 Ray setup completed successfully!")
        print("=" * 50)
        print("📋 Next steps:")
        print("  1. 📊 Visit dashboard: http://localhost:8265")
        print(f"  2. 🧪 Test training: python {example_path}")
        print("  3. 🚀 Run CNN+LSTM: python train_cnn_lstm_ray.py")
        print("  4. 📈 Monitor cluster: python monitor_ray.py")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
