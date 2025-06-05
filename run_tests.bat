# Run all tests in Docker container
docker run --rm --entrypoint="" trading-rl sh -c "cd /workspace && python3 -m pytest tests/ -v"
