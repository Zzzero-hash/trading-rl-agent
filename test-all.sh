#!/bin/sh
# All tests including Ray integration

# Start Ray cluster (head node) if not already running
if ! ray status >/dev/null 2>&1; then
  echo "🔄 Starting local Ray cluster..."
  ray stop >/dev/null 2>&1
  ray start --head --num-cpus=4 --num-gpus=1
else
  echo "✅ Ray cluster already running."
fi

echo "🎯 Running all tests..."
python3 -m pytest tests/ -v --tb=short
